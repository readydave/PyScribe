#!/usr/bin/env python3
"""Spike: are Nemotron streaming token timestamps good enough for word-level speaker assignment?

Streams audio through Nemotron exactly as the live backend would (chunked feature generator), takes the
per-token durations from the final generate() output, builds words, and compares each word's start time with
faster-whisper word timestamps on the same audio.

Usage (env with transformers>=5.13, faster-whisper, soundfile):
    .venv-tf5/bin/python scripts/spikes/nemotron_timestamps.py --audio <file> --seconds 180
"""

from __future__ import annotations

import argparse
import difflib
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from threading import Thread

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def norm(word: str) -> str:
    return re.sub(r"[^\w']", "", word.lower())


def build_words(tokens: list[dict]) -> list[dict]:
    """Group subword tokens into words: a token starting with a space opens a new word; punctuation joins."""
    words: list[dict] = []
    for tok in tokens:
        text = tok["token"]
        if not words or text.startswith(" "):
            words.append({"text": text.strip(), "start": tok["start"], "end": tok["end"]})
        else:
            words[-1]["text"] += text
            words[-1]["end"] = tok["end"]
    return [w for w in words if norm(w["text"])]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--seconds", type=float, default=180.0)
    parser.add_argument("--model", default="nvidia/nemotron-speech-streaming-en-0.6b")
    parser.add_argument("--whisper", default="large-v3-turbo")
    args = parser.parse_args()

    import soundfile as sf
    import torch
    from faster_whisper import WhisperModel
    from transformers import AutoModelForRNNT, AutoProcessor

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(args.audio), "-ac", "1", "-ar", "16000",
                        "-t", str(args.seconds), str(wav)], check=True)
        audio, sr = sf.read(wav, dtype="float32")

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForRNNT.from_pretrained(args.model, dtype=torch.float16).to("cuda").eval()
    first_n = processor.num_samples_first_audio_chunk
    first = processor(audio[:first_n], sampling_rate=sr, is_streaming=True, is_first_audio_chunk=True,
                      return_tensors="pt").to(model.device, dtype=model.dtype)

    def features():
        yield first.input_features[:, : processor.num_mel_frames_first_audio_chunk, :]
        frame = processor.num_mel_frames_first_audio_chunk
        hop, n_fft = processor.feature_extractor.hop_length, processor.feature_extractor.n_fft
        start = frame * hop - n_fft // 2
        while (end := start + processor.num_samples_per_audio_chunk) < audio.shape[0]:
            inputs = processor(audio[start:end], sampling_rate=sr, is_streaming=True, is_first_audio_chunk=False,
                               return_tensors="pt").to(model.device, dtype=model.dtype)
            yield inputs.input_features
            frame += processor.num_mel_frames_per_audio_chunk
            start = frame * hop - n_fft // 2

    result: dict = {}

    def run() -> None:
        result["out"] = model.generate(**{**first, "input_features": features()},
                                       return_dict_in_generate=True)

    thread = Thread(target=run)
    thread.start()
    thread.join()
    out = result["out"]
    text, stamps = processor.decode(out.sequences, durations=out.durations, skip_special_tokens=True)
    words = build_words(stamps[0])
    print(f"audio={len(audio) / sr:.0f}s nemotron_words={len(words)} frame={processor._encoder_frame_ms:.0f}ms")
    print("text:", (text if isinstance(text, str) else text[0])[:200])

    del model
    torch.cuda.empty_cache()
    whisper = WhisperModel(args.whisper, device="cuda", compute_type="float16")
    segs, _ = whisper.transcribe(audio, word_timestamps=True, language="en", vad_filter=True,
                                 no_repeat_ngram_size=4)
    ref = [{"text": w.word.strip(), "start": w.start, "end": w.end} for s in segs for w in s.words if norm(w.word)]
    print(f"whisper_words={len(ref)}")

    matcher = difflib.SequenceMatcher(a=[norm(w["text"]) for w in words], b=[norm(w["text"]) for w in ref],
                                      autojunk=False)
    diffs: list[float] = []
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            diffs.append(words[block.a + k]["start"] - ref[block.b + k]["start"])
    if not diffs:
        print("no matching words")
        return 1
    absd = sorted(abs(d) for d in diffs)
    matched = len(diffs)
    print(f"matched={matched} ({matched / len(words) * 100:.0f}% of nemotron words)")
    print(f"start offset nemotron-whisper: median={statistics.median(diffs) * 1000:+.0f} ms")
    print(f"abs diff: median={statistics.median(absd) * 1000:.0f} ms p90={absd[int(len(absd) * 0.9) - 1] * 1000:.0f} ms "
          f"p99={absd[int(len(absd) * 0.99) - 1] * 1000:.0f} ms max={absd[-1] * 1000:.0f} ms")
    within = lambda ms: sum(d <= ms / 1000 for d in absd) / len(absd) * 100  # noqa: E731
    print(f"within 160 ms: {within(160):.0f}%  within 320 ms: {within(320):.0f}%  within 640 ms: {within(640):.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
