"""Tests for backend-aware transcription service flows."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from services.granite_speech_service import GraniteSpeechModelBundle
from services.model_service import resolve_transcription_model
from services.transcription_service import TranscriptionResult, transcribe_media_file, transcribe_prepared_audio


@dataclass
class _FakeVisualResult:
    report: str
    available: bool
    cancelled: bool
    elapsed_seconds: float


class TranscriptionServiceTests(unittest.TestCase):
    def test_transcribe_prepared_audio_granite_returns_transcript_without_segments(self) -> None:
        bundle = GraniteSpeechModelBundle(
            processor=object(),
            model=object(),
            model_name="ibm-granite/granite-4.0-1b-speech",
            device="cpu",
        )
        spec = resolve_transcription_model("ibm-granite/granite-4.0-1b-speech")
        captured_text: list[str] = []
        captured_progress: list[float] = []

        with patch("services.transcription_service._probe_duration_seconds", return_value=12.5), patch(
            "services.transcription_service.load_audio_waveform",
            return_value=[0.1, 0.2],
        ), patch(
            "services.transcription_service.transcribe_granite_audio",
            return_value="granite transcript",
        ):
            result = transcribe_prepared_audio(
                wav_path="prepared.wav",
                model=bundle,
                model_spec=spec,
                language=None,
                on_text=captured_text.append,
                on_progress=captured_progress.append,
            )

        self.assertEqual(result.transcript, "granite transcript")
        self.assertEqual(result.transcript_only, "granite transcript")
        self.assertEqual(result.segments, [])
        self.assertFalse(result.cancelled)
        self.assertEqual(captured_text, ["granite transcript"])
        self.assertEqual(captured_progress, [])

    def test_transcribe_media_file_disables_granite_diarization_and_appends_visual_report(self) -> None:
        statuses: list[str] = []
        spec = resolve_transcription_model("ibm-granite/granite-4.0-1b-speech")
        fake_multimodal = SimpleNamespace(
            analyze_video_stream=lambda *args, **kwargs: _FakeVisualResult(
                report="OCR summary",
                available=True,
                cancelled=False,
                elapsed_seconds=1.25,
            )
        )

        with patch.dict(sys.modules, {"services.multimodal_service": fake_multimodal}), patch(
            "services.transcription_service.get_ffmpeg_cmd",
            return_value="ffmpeg",
        ), patch(
            "services.transcription_service.convert_to_16k_mono",
            return_value="prepared.wav",
        ), patch(
            "services.transcription_service.ensure_model_cached",
            return_value="C:\\cache\\granite",
        ), patch(
            "services.transcription_service.load_model",
            return_value=object(),
        ) as load_model_mock, patch(
            "services.transcription_service.transcribe_prepared_audio",
            return_value=TranscriptionResult(
                transcript="granite transcript",
                transcript_only="granite transcript",
                visual_report="",
                segments=[],
                cancelled=False,
                duration_seconds=12.0,
                transcription_seconds=2.0,
                diarization_seconds=0.0,
                visual_analysis_seconds=0.0,
            ),
        ) as transcribe_mock:
            result = transcribe_media_file(
                media_path="clip.mp4",
                model_name="ibm-granite/granite-4.0-1b-speech",
                run_mode="full",
                device="cpu",
                compute_type="int8",
                cancel_event=threading.Event(),
                use_diarization=True,
                diar_backend="accurate",
                use_visual_analysis=True,
                on_status=statuses.append,
            )

        self.assertIn("OCR summary", result.transcript)
        self.assertEqual(result.transcript_only, "granite transcript")
        self.assertTrue(any("Diarization unavailable" in status for status in statuses))
        self.assertEqual(load_model_mock.call_args.kwargs["model_spec"], spec)
        self.assertFalse(transcribe_mock.call_args.kwargs["use_diarization"])


if __name__ == "__main__":
    unittest.main()
