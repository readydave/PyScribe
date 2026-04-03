"""Tests for live transcription helpers and session orchestration."""

from __future__ import annotations

import json
import queue
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from services.live_transcription_service import (
    LIVE_SAMPLE_RATE,
    LiveAudioDevice,
    LiveSessionController,
    LiveSessionOptions,
    choose_live_audio_devices,
    classify_live_audio_device,
    live_model_supported,
    reconcile_live_transcript,
)


class _FakeRequestQueue:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def put(self, item: dict) -> None:
        self.items.append(dict(item))

    def close(self) -> None:
        return None

    def join_thread(self) -> None:
        return None


class LiveTranscriptionServiceTests(unittest.TestCase):
    @staticmethod
    def _options(tmp_dir: str, *, source_mode: str = "microphone", keep_audio: bool = True) -> LiveSessionOptions:
        return LiveSessionOptions(
            model_name="deepdml/faster-whisper-large-v3-turbo-ct2",
            device="cpu",
            compute_type="int8",
            language=None,
            source_mode=source_mode,
            input_device_id="device-1",
            input_device_name="Built-in Input",
            output_root=tmp_dir,
            keep_audio_on_success=keep_audio,
            use_diarization=False,
            diar_backend="off",
            max_speakers=None,
        )

    def test_classify_live_audio_device_detects_loopback_markers(self) -> None:
        self.assertEqual(classify_live_audio_device("Monitor of Built-in Audio"), "loopback")
        self.assertEqual(classify_live_audio_device("USB Stereo Mix"), "loopback")
        self.assertEqual(classify_live_audio_device("Headset Microphone"), "microphone")

    def test_choose_live_audio_devices_filters_by_source_mode(self) -> None:
        devices = [
            LiveAudioDevice(id="mic-1", name="Mic", kind="microphone", available=True),
            LiveAudioDevice(id="loop-1", name="Monitor", kind="loopback", available=True),
            LiveAudioDevice(id="mic-2", name="Muted", kind="microphone", available=False),
        ]

        self.assertEqual([device.id for device in choose_live_audio_devices(devices, source_mode="microphone")], ["mic-1"])
        self.assertEqual([device.id for device in choose_live_audio_devices(devices, source_mode="loopback")], ["loop-1"])

    def test_reconcile_live_transcript_replaces_draft_tail_without_duplication(self) -> None:
        committed = [{"start": 0.0, "end": 2.0, "text": "hello"}]
        first_segments = [
            {"start": 2.1, "end": 3.8, "text": "draft one"},
            {"start": 4.2, "end": 5.6, "text": "draft two"},
        ]
        committed_after_first, draft_after_first = reconcile_live_transcript(
            committed,
            first_segments,
            window_start_seconds=2.0,
            window_end_seconds=6.0,
            stabilization_tail_seconds=2.0,
        )
        self.assertEqual([segment["text"] for segment in committed_after_first], ["hello", "draft one"])
        self.assertEqual([segment["text"] for segment in draft_after_first], ["draft two"])

        second_segments = [
            {"start": 4.1, "end": 4.4, "text": "draft two corrected"},
            {"start": 5.1, "end": 6.0, "text": "latest"},
        ]
        committed_after_second, draft_after_second = reconcile_live_transcript(
            committed_after_first,
            second_segments,
            window_start_seconds=4.0,
            window_end_seconds=6.0,
            stabilization_tail_seconds=1.5,
        )
        self.assertEqual([segment["text"] for segment in committed_after_second], ["hello", "draft one", "draft two corrected"])
        self.assertEqual([segment["text"] for segment in draft_after_second], ["latest"])

    def test_live_model_supported_rejects_granite(self) -> None:
        self.assertTrue(live_model_supported("deepdml/faster-whisper-large-v3-turbo-ct2"))
        self.assertFalse(live_model_supported("ibm-granite/granite-4.0-1b-speech"))

    def test_live_session_emits_incremental_transcript_events(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(LiveSessionController, "_start_asr_process", return_value=None):
            controller = LiveSessionController(self._options(temp_dir))
            controller.start()
            controller._request_queue = _FakeRequestQueue()
            controller._event_queue = queue.Queue()

            audio = np.zeros(int(LIVE_SAMPLE_RATE * 3.1), dtype=np.float32)
            controller.append_audio_chunk(audio, np.zeros(audio.size, dtype=np.int16).tobytes())
            self.assertEqual(controller._request_queue.items[0]["type"], "decode")

            controller._event_queue.put(
                {
                    "type": "result",
                    "request_id": 1,
                    "segments": [{"start": 0.0, "end": 2.4, "text": "hello world"}],
                    "window_start_seconds": 0.0,
                    "window_end_seconds": 3.1,
                    "stabilization_tail_seconds": 0.5,
                    "final": False,
                }
            )
            events = controller.poll_events()
            transcript_events = [event for event in events if event.get("type") == "transcript"]
            self.assertEqual(len(transcript_events), 1)
            self.assertEqual(transcript_events[0]["value"], "hello world")
            controller.shutdown()

    def test_live_session_error_event_marks_metadata_failed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(LiveSessionController, "_start_asr_process", return_value=None):
            controller = LiveSessionController(self._options(temp_dir, source_mode="loopback"))
            controller.start()
            controller._request_queue = _FakeRequestQueue()
            controller._event_queue = queue.Queue()
            controller._event_queue.put({"type": "error", "value": "worker crashed"})

            events = controller.poll_events()
            metadata = json.loads(Path(controller.metadata_path).read_text(encoding="utf-8"))

            self.assertEqual(events[0]["type"], "error")
            self.assertEqual(metadata["status"], "failed")
            self.assertEqual(metadata["source_mode"], "loopback")
            controller.shutdown()

    def test_live_session_finalize_success_can_delete_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(LiveSessionController, "_start_asr_process", return_value=None):
            controller = LiveSessionController(self._options(temp_dir, keep_audio=False))
            controller.start()
            controller._request_queue = _FakeRequestQueue()
            controller._event_queue = queue.Queue()
            controller.append_audio_chunk(np.zeros(LIVE_SAMPLE_RATE, dtype=np.float32), np.zeros(LIVE_SAMPLE_RATE, dtype=np.int16).tobytes())
            controller.close_capture()
            self.assertTrue(controller.capture_path.exists())

            controller.finalize_success("final transcript")
            metadata = json.loads(Path(controller.metadata_path).read_text(encoding="utf-8"))

            self.assertEqual(metadata["status"], "completed")
            self.assertFalse(controller.capture_path.exists())
            self.assertTrue(controller.final_transcript_path.exists())
            controller.shutdown()

    def test_live_session_finalize_cancelled_keeps_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(LiveSessionController, "_start_asr_process", return_value=None):
            controller = LiveSessionController(self._options(temp_dir))
            controller.start()
            controller._request_queue = _FakeRequestQueue()
            controller._event_queue = queue.Queue()
            controller.append_audio_chunk(np.zeros(LIVE_SAMPLE_RATE, dtype=np.float32), np.zeros(LIVE_SAMPLE_RATE, dtype=np.int16).tobytes())
            controller.close_capture()
            controller.finalize_cancelled()
            metadata = json.loads(Path(controller.metadata_path).read_text(encoding="utf-8"))

            self.assertEqual(metadata["status"], "cancelled")
            self.assertTrue(controller.capture_path.exists())
            controller.shutdown()


if __name__ == "__main__":
    unittest.main()
