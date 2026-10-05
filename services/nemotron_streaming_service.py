"""Nemotron Speech Streaming (RNN-T, cache-aware) runtime helpers.

One chunked-streaming session class serves both live capture (audio arrives as it is spoken) and file
transcription (the whole file is fed through the same path). Word timestamps come from the per-token
durations of the final ``generate()`` output, so they are available when the session finishes.
"""

from __future__ import annotations

import logging
import queue
import re
import time
from dataclasses import dataclass
from threading import Event, Thread
from types import SimpleNamespace
from typing import Any, Callable, Iterator

import numpy as np

LOGGER = logging.getLogger(__name__)

NEMOTRON_STREAMING_REPO_IDS = {"nvidia/nemotron-speech-streaming-en-0.6b"}
NEMOTRON_SAMPLE_RATE = 16_000
MIN_TRANSFORMERS = (5, 13)
# Word end = min(next word start, last token end + this): keeps a word from spanning a long pause.
MAX_WORD_EXTENSION_SECONDS = 0.5
# Segment breaks for file transcripts: sentence end, a pause, or a length cap.
SEGMENT_PAUSE_SECONDS = 2.0
SEGMENT_MAX_SECONDS = 30.0
_SENTENCE_END_RE = re.compile(r"[.!?][\"')\]]*$")
# File mode feeds audio faster than the model consumes it; keep at most this many chunks queued.
_MAX_QUEUED_CHUNKS = 4

ProgressCallback = Callable[[float], None]
TextCallback = Callable[[str], None]


@dataclass(frozen=True)
class NemotronModelBundle:
    processor: Any
    model: Any
    model_name: str
    device: str


def is_nemotron_streaming(model_name: str | None) -> bool:
    """True for the supported Nemotron streaming checkpoints."""
    return str(model_name or "").strip() in NEMOTRON_STREAMING_REPO_IDS


def _version_tuple(text: str) -> tuple[int, ...]:
    parts = []
    for piece in text.split(".")[:2]:
        match = re.match(r"\d+", piece)
        parts.append(int(match.group()) if match else 0)
    return tuple(parts)


def require_nemotron_runtime() -> None:
    """Raise a RuntimeError with an install hint when the Transformers 5 runtime is missing."""
    try:
        import transformers
    except ImportError as exc:  # pragma: no cover - transformers is a hard requirement
        raise RuntimeError("Nemotron streaming needs the 'transformers' package.") from exc
    if _version_tuple(transformers.__version__) < MIN_TRANSFORMERS:
        raise RuntimeError(
            f"Nemotron streaming needs transformers>=5.13 (installed {transformers.__version__}). "
            "Recreate the virtual environment from requirements.txt."
        )
    try:
        import librosa  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Nemotron streaming needs 'librosa'. Install it from requirements.txt.") from exc


def load_nemotron_model(model_name: str, *, device: str, compute_type: str = "float16") -> NemotronModelBundle:
    """Load the Nemotron processor and model (float16 on CUDA, float32 on CPU)."""
    require_nemotron_runtime()
    import torch
    from transformers import AutoModelForRNNT, AutoProcessor

    use_cuda = device == "cuda"
    dtype = torch.float16 if use_cuda else torch.float32
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForRNNT.from_pretrained(model_name, dtype=dtype).to("cuda" if use_cuda else "cpu").eval()
    return NemotronModelBundle(processor=processor, model=model, model_name=model_name, device=device)


def build_words(tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group subword tokens into words with `text`/`start`/`end` seconds.

    A token starting with a space opens a new word; punctuation joins the previous word. Each word
    ends at the next word's start (at most MAX_WORD_EXTENSION_SECONDS past its last token).
    """
    words: list[dict[str, Any]] = []
    for tok in tokens:
        text = str(tok["token"])
        if not words or text.startswith(" "):
            words.append({"text": text.strip(), "start": float(tok["start"]), "end": float(tok["end"])})
        else:
            words[-1]["text"] += text
            words[-1]["end"] = float(tok["end"])
    words = [w for w in words if re.search(r"\w", w["text"])]
    for index, word in enumerate(words):
        cap = word["end"] + MAX_WORD_EXTENSION_SECONDS
        following = words[index + 1]["start"] if index + 1 < len(words) else cap
        word["end"] = max(word["end"], min(following, cap))
    return words


def words_to_segments(words: list[dict[str, Any]]) -> list[SimpleNamespace]:
    """Split words into segment objects (`start`, `end`, `text`, `words`) shaped like faster-whisper's."""
    segments: list[SimpleNamespace] = []
    current: list[dict[str, Any]] = []

    def flush() -> None:
        if not current:
            return
        segments.append(
            SimpleNamespace(
                start=current[0]["start"],
                end=current[-1]["end"],
                text=" ".join(w["text"] for w in current),
                words=[SimpleNamespace(word=f" {w['text']}", start=w["start"], end=w["end"]) for w in current],
            )
        )
        current.clear()

    for word in words:
        if current and (
            word["start"] - current[-1]["end"] >= SEGMENT_PAUSE_SECONDS
            or word["end"] - current[0]["start"] > SEGMENT_MAX_SECONDS
        ):
            flush()
        current.append(word)
        if _SENTENCE_END_RE.search(word["text"]):
            flush()
    flush()
    return segments


class NemotronStreamSession:
    """One streaming transcription: `feed()` audio, `poll_text()` for new text, `finish()` for timed words."""

    def __init__(self, bundle: NemotronModelBundle, *, stream_text: bool = True) -> None:
        # The text streamer decodes every token in Python (about 5x slower overall); only live mode needs it.
        self._stream_text = stream_text
        self._bundle = bundle
        processor = bundle.processor
        self._first_samples = int(processor.num_samples_first_audio_chunk)
        self._first_frames = int(processor.num_mel_frames_first_audio_chunk)
        self._chunk_samples = int(processor.num_samples_per_audio_chunk)
        self._chunk_frames = int(processor.num_mel_frames_per_audio_chunk)
        self._hop = int(processor.feature_extractor.hop_length)
        self._n_fft = int(processor.feature_extractor.n_fft)
        self._audio = np.zeros(0, dtype=np.float32)
        self._audio_offset = 0
        self._total = 0
        self._started = False
        self._frame = 0
        self._next_start = 0
        self._chunks: queue.Queue[Any] = queue.Queue()
        self._consumed_chunks = 0
        self._queued_chunks = 0
        self._streamer: Any = None
        self._thread: Thread | None = None
        self._output: Any = None
        self._error: BaseException | None = None
        self._closed = False

    @property
    def fed_seconds(self) -> float:
        return self._total / float(NEMOTRON_SAMPLE_RATE)

    @property
    def error(self) -> BaseException | None:
        return self._error

    def pending_chunks(self) -> int:
        return self._queued_chunks - self._consumed_chunks

    def feed(self, samples: np.ndarray) -> None:
        """Append mono 16 kHz float32 audio and queue every chunk that is now complete."""
        if self._closed:
            raise RuntimeError("Nemotron stream is already finished.")
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return
        self._audio = np.concatenate([self._audio, audio])
        self._total += int(audio.size)
        self._emit_ready_chunks()

    def poll_text(self) -> str:
        """Return text decoded since the last call (empty when nothing new)."""
        streamer = self._streamer
        if streamer is None:
            return ""
        pieces: list[str] = []
        while True:
            try:
                item = streamer.text_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, str):
                pieces.append(item)
        return "".join(pieces)

    def finish(self) -> list[dict[str, Any]]:
        """Flush the model, wait for it, and return words (`text`, `start`, `end` seconds)."""
        if self._closed:
            return []
        self._closed = True
        if self._total == 0:
            return []
        # Zero-pad so the tail audio and the model's right-context lookahead are fully decoded.
        pad = self._chunk_samples + (0 if self._started else self._first_samples)
        self._append_silence(pad)
        self._emit_ready_chunks()
        self._chunks.put(None)
        if self._thread is not None:
            self._thread.join()
        if self._error is not None:
            raise RuntimeError(f"Nemotron streaming failed: {self._error}") from self._error
        if self._output is None:
            return []
        processor = self._bundle.processor
        _, stamps = processor.decode(self._output.sequences, durations=self._output.durations, skip_special_tokens=True)
        return build_words(stamps[0])

    def abort(self) -> None:
        """Stop the decoding thread without waiting for a result."""
        self._closed = True
        self._chunks.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _append_silence(self, samples: int) -> None:
        self._audio = np.concatenate([self._audio, np.zeros(samples, dtype=np.float32)])
        self._total += samples

    def _slice(self, start: int, end: int) -> np.ndarray:
        return self._audio[start - self._audio_offset : end - self._audio_offset]

    def _trim(self) -> None:
        keep_from = max(0, self._next_start)
        drop = keep_from - self._audio_offset
        if drop > 0:
            self._audio = self._audio[drop:]
            self._audio_offset = keep_from

    def _features(self, samples: np.ndarray, *, first: bool) -> Any:
        bundle = self._bundle
        return bundle.processor(
            samples,
            sampling_rate=NEMOTRON_SAMPLE_RATE,
            is_streaming=True,
            is_first_audio_chunk=first,
            return_tensors="pt",
        ).to(bundle.model.device, dtype=bundle.model.dtype)

    def _emit_ready_chunks(self) -> None:
        if not self._started:
            if self._total < self._first_samples:
                return
            first = self._features(self._slice(0, self._first_samples), first=True)
            self._started = True
            self._frame = self._first_frames
            self._next_start = self._frame * self._hop - self._n_fft // 2
            self._chunks.put(first.input_features[:, : self._first_frames, :])
            self._queued_chunks += 1
            self._start_generate(first)
        while self._next_start + self._chunk_samples <= self._total:
            end = self._next_start + self._chunk_samples
            inputs = self._features(self._slice(self._next_start, end), first=False)
            self._chunks.put(inputs.input_features)
            self._queued_chunks += 1
            self._frame += self._chunk_frames
            self._next_start = self._frame * self._hop - self._n_fft // 2
        self._trim()

    def _chunk_iterator(self) -> Iterator[Any]:
        while True:
            chunk = self._chunks.get()
            if chunk is None:
                return
            self._consumed_chunks += 1
            yield chunk

    def _start_generate(self, first: Any) -> None:
        kwargs = {**first, "input_features": self._chunk_iterator(), "return_dict_in_generate": True}
        if self._stream_text:
            from transformers import TextIteratorStreamer

            self._streamer = TextIteratorStreamer(self._bundle.processor.tokenizer, skip_special_tokens=True)
            kwargs["streamer"] = self._streamer

        def run() -> None:
            try:
                self._output = self._bundle.model.generate(**kwargs)
            except BaseException as exc:  # surfaced by finish()
                LOGGER.exception("Nemotron streaming generate failed")
                self._error = exc

        self._thread = Thread(target=run, name="nemotron-generate", daemon=True)
        self._thread.start()


def transcribe_nemotron_audio(
    bundle: NemotronModelBundle,
    audio_np: np.ndarray,
    *,
    cancel_event: Event | None = None,
    on_progress: ProgressCallback | None = None,
    on_text: TextCallback | None = None,
) -> list[SimpleNamespace]:
    """Stream a whole 16 kHz mono waveform through Nemotron and return timed segments (empty if cancelled)."""
    session = NemotronStreamSession(bundle, stream_text=on_text is not None)
    total = max(1, int(audio_np.size))
    step = NEMOTRON_SAMPLE_RATE  # one second per feed keeps cancel/progress responsive
    streamed = ""
    try:
        for offset in range(0, int(audio_np.size), step):
            if cancel_event is not None and cancel_event.is_set():
                session.abort()
                return []
            session.feed(audio_np[offset : offset + step])
            while session.pending_chunks() > _MAX_QUEUED_CHUNKS and session.error is None:
                time.sleep(0.01)
            if session.error is not None:
                break
            if on_progress:
                on_progress(min(99.0, (offset + step) / total * 100.0))
            text = session.poll_text()
            if text and on_text:
                streamed += text
                on_text(streamed.strip())
        words = session.finish()
    except BaseException:
        session.abort()
        raise
    if cancel_event is not None and cancel_event.is_set():
        return []
    if on_progress:
        on_progress(100.0)
    return words_to_segments(words)
