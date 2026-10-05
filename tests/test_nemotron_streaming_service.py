"""Tests for the Nemotron streaming service (word building, chunking, version gate)."""

from __future__ import annotations

import queue
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from services.nemotron_streaming_service import (
    NemotronModelBundle,
    NemotronStreamSession,
    build_words,
    is_nemotron_streaming,
    require_nemotron_runtime,
    transcribe_nemotron_audio,
    words_to_segments,
)


def _tokens(*items: tuple[str, float, float]) -> list[dict]:
    return [{"token": text, "start": start, "end": end} for text, start, end in items]


class WordBuildingTests(unittest.TestCase):
    def test_subwords_and_punctuation_join_into_words(self) -> None:
        words = build_words(_tokens((" Hel", 0.0, 0.08), ("lo", 0.08, 0.16), (",", 0.16, 0.24), (" world", 0.4, 0.48)))
        self.assertEqual([w["text"] for w in words], ["Hello,", "world"])
        self.assertAlmostEqual(words[0]["start"], 0.0)

    def test_word_end_extends_to_next_start_but_not_across_long_pauses(self) -> None:
        words = build_words(_tokens((" a", 0.0, 0.08), (" b", 0.3, 0.38), (" c", 5.0, 5.08)))
        self.assertAlmostEqual(words[0]["end"], 0.3)  # next word starts soon: extend to it
        self.assertAlmostEqual(words[1]["end"], 0.38 + 0.5)  # long pause: capped extension
        self.assertAlmostEqual(words[2]["end"], 5.08 + 0.5)  # last word

    def test_tokens_without_word_characters_are_dropped(self) -> None:
        self.assertEqual(build_words(_tokens((" ", 0.0, 0.08), (" ok", 0.1, 0.18))), [
            {"text": "ok", "start": 0.1, "end": 0.18 + 0.5}
        ])

    def test_segments_split_on_sentence_end_and_pause(self) -> None:
        words = [
            {"text": "Hello.", "start": 0.0, "end": 0.4},
            {"text": "How", "start": 0.5, "end": 0.7},
            {"text": "are", "start": 0.7, "end": 0.9},
            {"text": "you", "start": 3.0, "end": 3.2},
        ]
        segments = words_to_segments(words)
        self.assertEqual([s.text for s in segments], ["Hello.", "How are", "you"])
        self.assertEqual(segments[1].words[0].word, " How")
        self.assertAlmostEqual(segments[1].end, 0.9)


class RuntimeGateTests(unittest.TestCase):
    def test_model_id_detection(self) -> None:
        self.assertTrue(is_nemotron_streaming("nvidia/nemotron-speech-streaming-en-0.6b"))
        self.assertFalse(is_nemotron_streaming("tiny"))

    def test_old_transformers_is_rejected_with_install_hint(self) -> None:
        with patch("transformers.__version__", "4.57.1"):
            with self.assertRaisesRegex(RuntimeError, "transformers>=5.13"):
                require_nemotron_runtime()


class _Features(dict):
    def __init__(self, frames: int) -> None:
        super().__init__(input_features=np.zeros((1, frames, 4), dtype=np.float32))
        self.input_features = self["input_features"]

    def to(self, *args, **kwargs) -> "_Features":
        return self


class _FakeProcessor:
    num_samples_first_audio_chunk = 1000
    num_mel_frames_first_audio_chunk = 10
    num_samples_per_audio_chunk = 800
    num_mel_frames_per_audio_chunk = 8
    feature_extractor = SimpleNamespace(hop_length=100, n_fft=200)
    tokenizer = object()

    def __init__(self) -> None:
        self.calls: list[tuple[int, bool]] = []

    def __call__(self, samples, *, is_first_audio_chunk, **kwargs) -> _Features:
        self.calls.append((len(samples), is_first_audio_chunk))
        return _Features(10 if is_first_audio_chunk else 8)

    def decode(self, sequences, *, durations, skip_special_tokens):
        return "", [_tokens((" hi", 0.0, 0.08), (" there.", 0.3, 0.38))]


class _FakeStreamer:
    def __init__(self, tokenizer, **kwargs) -> None:
        self.text_queue: queue.Queue = queue.Queue()


class _FakeModel:
    device = "cpu"
    dtype = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        chunks = list(kwargs["input_features"])
        if "streamer" in kwargs:
            kwargs["streamer"].text_queue.put(f"{len(chunks)} chunks ")
        return SimpleNamespace(sequences=[[1]], durations=[[1]])


@unittest.skipUnless(__import__("importlib").util.find_spec("transformers"), "transformers not installed")
class StreamSessionTests(unittest.TestCase):
    def _session(self) -> tuple[NemotronStreamSession, _FakeProcessor]:
        processor = _FakeProcessor()
        bundle = NemotronModelBundle(processor=processor, model=_FakeModel(), model_name="m", device="cpu")
        return NemotronStreamSession(bundle), processor

    def test_nothing_runs_until_the_first_chunk_is_complete(self) -> None:
        with patch("transformers.TextIteratorStreamer", _FakeStreamer):
            session, processor = self._session()
            session.feed(np.zeros(999, dtype=np.float32))
            self.assertEqual(processor.calls, [])
            session.abort()

    def test_chunks_follow_first_chunk_then_fixed_windows(self) -> None:
        with patch("transformers.TextIteratorStreamer", _FakeStreamer):
            session, processor = self._session()
            session.feed(np.zeros(3000, dtype=np.float32))
            # first chunk 1000 samples, next windows start at 10*100-100=900 and then every 800
            self.assertEqual(processor.calls[0], (1000, True))
            self.assertEqual([c for c in processor.calls[1:]], [(800, False)] * 2)  # windows [900,1700) and [1700,2500); [2500,3300) is incomplete
            session.abort()

    def test_finish_pads_flushes_and_returns_timed_words(self) -> None:
        with patch("transformers.TextIteratorStreamer", _FakeStreamer):
            session, _ = self._session()
            session.feed(np.zeros(1500, dtype=np.float32))
            words = session.finish()
            self.assertEqual([w["text"] for w in words], ["hi", "there."])
            self.assertIn("chunks", session.poll_text())
            self.assertEqual(session.finish(), [])

    def test_finish_without_audio_returns_nothing(self) -> None:
        with patch("transformers.TextIteratorStreamer", _FakeStreamer):
            session, _ = self._session()
            self.assertEqual(session.finish(), [])

    def test_streamer_is_only_attached_when_text_streaming_is_requested(self) -> None:
        with patch("transformers.TextIteratorStreamer", _FakeStreamer):
            for stream_text in (True, False):
                processor, model = _FakeProcessor(), _FakeModel()
                session = NemotronStreamSession(
                    NemotronModelBundle(processor=processor, model=model, model_name="m", device="cpu"),
                    stream_text=stream_text,
                )
                session.feed(np.zeros(1500, dtype=np.float32))
                session.finish()
                self.assertEqual("streamer" in model.kwargs, stream_text)
                self.assertEqual(session.poll_text() != "", stream_text)

    def test_transcribe_audio_returns_segments_and_reports_progress(self) -> None:
        progress: list[float] = []
        with patch("transformers.TextIteratorStreamer", _FakeStreamer):
            processor = _FakeProcessor()
            bundle = NemotronModelBundle(processor=processor, model=_FakeModel(), model_name="m", device="cpu")
            segments = transcribe_nemotron_audio(bundle, np.zeros(40_000, dtype=np.float32), on_progress=progress.append)
        self.assertEqual([s.text for s in segments], ["hi there."])
        self.assertEqual(progress[-1], 100.0)


if __name__ == "__main__":
    unittest.main()
