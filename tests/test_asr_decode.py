"""Tests for faster-whisper decoding helpers (no GPU or models required)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from services import asr_decode
from services.live_vram_service import GpuMemoryInfo


def _gpu(free_gb: float) -> GpuMemoryInfo:
    return GpuMemoryInfo(total_gb=24.0, free_gb=free_gb, used_gb=24.0 - free_gb, source="nvml")


class ResolveBatchSizeTests(unittest.TestCase):
    def test_cpu_defaults_to_sequential(self) -> None:
        self.assertEqual(asr_decode.resolve_batch_size("cpu"), 1)

    def test_explicit_request_wins_and_is_clamped(self) -> None:
        self.assertEqual(asr_decode.resolve_batch_size("cpu", 8), 8)
        self.assertEqual(asr_decode.resolve_batch_size("cuda", 0), 1)

    def test_gpu_tiers_follow_free_vram(self) -> None:
        for free_gb, expected in ((20.0, 16), (6.0, 8), (4.0, 4), (2.0, 1)):
            with patch("services.asr_decode.get_gpu_memory_info", return_value=_gpu(free_gb)):
                self.assertEqual(asr_decode.resolve_batch_size("cuda"), expected, free_gb)

    def test_unknown_gpu_memory_is_sequential(self) -> None:
        with patch("services.asr_decode.get_gpu_memory_info", return_value=None):
            self.assertEqual(asr_decode.resolve_batch_size("cuda"), 1)


class OpenSegmentStreamTests(unittest.TestCase):
    def test_sequential_passes_hotwords_and_vad(self) -> None:
        calls: list[dict] = []

        class _Model:
            def transcribe(self, audio, **kwargs):
                calls.append(kwargs)
                return iter([SimpleNamespace(text="a"), SimpleNamespace(text="b")]), None

        stream = asr_decode.open_segment_stream(_Model(), [0.0], "en", hotwords=" Nemotron ", word_timestamps=True)
        self.assertEqual([s.text for s in stream], ["a", "b"])
        self.assertEqual(calls[0]["hotwords"], "Nemotron")
        self.assertTrue(calls[0]["vad_filter"])
        self.assertTrue(calls[0]["word_timestamps"])

    def test_empty_hotwords_become_none(self) -> None:
        calls: list[dict] = []

        class _Model:
            def transcribe(self, audio, **kwargs):
                calls.append(kwargs)
                return iter(()), None

        list(asr_decode.open_segment_stream(_Model(), [0.0], None, hotwords="  "))
        self.assertIsNone(calls[0]["hotwords"])

    def test_oom_on_first_batch_falls_back_to_sequential(self) -> None:
        class _Model:
            def transcribe(self, audio, **kwargs):
                return iter([SimpleNamespace(text="seq")]), None

        class _Batched:
            def __init__(self, model) -> None:
                pass

            def transcribe(self, audio, **kwargs):
                raise RuntimeError("CUDA failed with error out of memory")

        with patch("faster_whisper.BatchedInferencePipeline", _Batched), patch(
            "services.asr_decode._free_cuda_cache"
        ) as free_cache:
            stream = asr_decode.open_segment_stream(_Model(), [0.0], "en", batch_size=8)
            self.assertEqual([s.text for s in stream], ["seq"])
        free_cache.assert_called_once()

    def test_oom_mid_stream_resumes_after_last_segment(self) -> None:
        seq_calls: list[dict] = []

        class _Model:
            def transcribe(self, audio, **kwargs):
                seq_calls.append(kwargs)
                return iter([SimpleNamespace(text="tail", end=9.0)]), None

        def _failing_stream():
            yield SimpleNamespace(text="head", end=4.5)
            raise RuntimeError("CUDA failed with error out of memory")

        class _Batched:
            def __init__(self, model) -> None:
                pass

            def transcribe(self, audio, **kwargs):
                return _failing_stream(), None

        with patch("faster_whisper.BatchedInferencePipeline", _Batched), patch("services.asr_decode._free_cuda_cache"):
            stream = asr_decode.open_segment_stream(_Model(), [0.0], "en", batch_size=8)
            self.assertEqual([s.text for s in stream], ["head", "tail"])
        self.assertEqual(seq_calls[0]["clip_timestamps"], [4.5])

    def test_sequential_retry_tolerates_one_stale_context_error(self) -> None:
        attempts: list[int] = []

        class _Model:
            def transcribe(self, audio, **kwargs):
                attempts.append(1)
                if len(attempts) == 1:
                    raise RuntimeError("parallel_for failed: cudaErrorInvalidDevice: invalid device ordinal")
                return iter([SimpleNamespace(text="ok", end=1.0)]), None

        class _Batched:
            def __init__(self, model) -> None:
                pass

            def transcribe(self, audio, **kwargs):
                raise RuntimeError("CUDA failed with error out of memory")

        with patch("faster_whisper.BatchedInferencePipeline", _Batched), patch("services.asr_decode._free_cuda_cache"):
            stream = asr_decode.open_segment_stream(_Model(), [0.0], "en", batch_size=4)
            self.assertEqual([s.text for s in stream], ["ok"])
        self.assertEqual(len(attempts), 2)

    def test_non_oom_error_is_not_swallowed(self) -> None:
        class _Batched:
            def __init__(self, model) -> None:
                pass

            def transcribe(self, audio, **kwargs):
                raise RuntimeError("cuDNN error")

        with patch("faster_whisper.BatchedInferencePipeline", _Batched):
            with self.assertRaises(RuntimeError):
                list(asr_decode.open_segment_stream(object(), [0.0], "en", batch_size=8))


class SegmentWordsTests(unittest.TestCase):
    def test_extracts_words(self) -> None:
        seg = SimpleNamespace(words=[SimpleNamespace(start=0.0, end=0.5, word=" hi", probability=0.91234)])
        self.assertEqual(
            asr_decode.segment_words(seg),
            [{"start": 0.0, "end": 0.5, "word": " hi", "probability": 0.9123}],
        )

    def test_missing_words_is_empty(self) -> None:
        self.assertEqual(asr_decode.segment_words(SimpleNamespace(words=None)), [])
        self.assertEqual(asr_decode.segment_words(SimpleNamespace()), [])


if __name__ == "__main__":
    unittest.main()
