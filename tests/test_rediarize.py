"""Speakers-only re-run on saved audio (services.transcription_service.rediarize_segments)."""

from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

from services import transcription_service as ts

SEGMENTS = [
    {"start": 0.0, "end": 4.0, "speaker": "S1", "text": "Hello."},
    {"start": 4.0, "end": 8.0, "speaker": "S1", "text": "Hi there."},
    {"start": 8.0, "end": 12.0, "speaker": "S1", "text": "Bye."},
]
TURNS = [
    {"start": 0.0, "end": 4.0, "speaker": "S1"},
    {"start": 4.0, "end": 8.0, "speaker": "S2"},
    {"start": 8.0, "end": 12.0, "speaker": "S3"},
]


class RediarizeSegmentsTests(unittest.TestCase):
    def setUp(self) -> None:
        for target, value in (
            ("get_ffmpeg_cmd", "ffmpeg"),
            ("convert_to_16k_mono", "/tmp/fake.wav"),
        ):
            patcher = patch.object(ts, target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_relabels_with_exact_count_and_keeps_text(self) -> None:
        with patch.object(ts, "_run_diarization_backend", return_value=TURNS) as run:
            out = ts.rediarize_segments("a.wav", SEGMENTS, num_speakers=3, backend="accurate", device="cpu")
        self.assertEqual(run.call_args.kwargs["max_speakers"], 3)
        self.assertEqual([s["speaker"] for s in out], ["S1", "S2", "S3"])
        self.assertEqual([s["text"] for s in out], ["Hello.", "Hi there.", "Bye."])
        self.assertEqual({s["speaker"] for s in SEGMENTS}, {"S1"}, "input segments must not be mutated")

    def test_no_turns_raises_instead_of_wiping_labels(self) -> None:
        with patch.object(ts, "_run_diarization_backend", return_value=[]):
            with self.assertRaises(RuntimeError):
                ts.rediarize_segments("a.wav", SEGMENTS, num_speakers=2)

    def test_rejects_zero_speakers(self) -> None:
        with self.assertRaises(ValueError):
            ts.rediarize_segments("a.wav", SEGMENTS, num_speakers=0)

    def test_cancel_event_interrupts(self) -> None:
        cancel = threading.Event()
        cancel.set()

        def fake_backend(**kwargs: object) -> list[dict]:
            kwargs["progress_cb"](10.0)  # type: ignore[operator]
            return TURNS

        with patch.object(ts, "_run_diarization_backend", side_effect=fake_backend):
            with self.assertRaises(InterruptedError):
                ts.rediarize_segments("a.wav", SEGMENTS, num_speakers=3, cancel_event=cancel)

    def test_cuda_failure_retries_on_cpu(self) -> None:
        calls: list[str] = []

        def fake_backend(**kwargs: object) -> list[dict]:
            calls.append(str(kwargs["device"]))
            if kwargs["device"] == "cuda":
                raise RuntimeError("cuda boom")
            return TURNS

        with (
            patch.object(ts, "_run_diarization_backend", side_effect=fake_backend),
            patch.object(ts, "_should_retry_diarization_on_cpu", return_value=True),
            patch.object(ts, "_collect_cuda_diagnostics", return_value=""),
        ):
            out = ts.rediarize_segments("a.wav", SEGMENTS, num_speakers=3, device="cuda")
        self.assertEqual(calls, ["cuda", "cpu"])
        self.assertEqual(len(out), 3)


if __name__ == "__main__":
    unittest.main()
