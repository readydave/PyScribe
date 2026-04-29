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
from services.transcription_service import (
    TranscriptionResult,
    _can_spawn_isolated_diarization,
    _should_isolate_diarization_backend,
    _should_retry_diarization_on_cpu,
    transcribe_media_file,
    transcribe_prepared_audio,
)


@dataclass
class _FakeVisualResult:
    report: str
    available: bool
    cancelled: bool
    elapsed_seconds: float


class TranscriptionServiceTests(unittest.TestCase):
    def test_should_retry_diarization_on_cpu_handles_cudnn_mismatch(self) -> None:
        exc = RuntimeError("cuDNN error: CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH")

        self.assertTrue(_should_retry_diarization_on_cpu(exc, backend="accurate", device="cuda"))
        self.assertFalse(_should_retry_diarization_on_cpu(exc, backend="accurate", device="cpu"))

    def test_should_isolate_diarization_backend_for_pyannote_backends(self) -> None:
        self.assertTrue(_should_isolate_diarization_backend("accurate"))
        self.assertTrue(_should_isolate_diarization_backend("fast"))
        self.assertFalse(_should_isolate_diarization_backend("off"))

    def test_can_spawn_isolated_diarization_requires_real_main_file(self) -> None:
        with patch.object(sys.modules["__main__"], "__file__", "/tmp/runner.py"):
            self.assertTrue(_can_spawn_isolated_diarization())
        with patch.object(sys.modules["__main__"], "__file__", "<stdin>"):
            self.assertFalse(_can_spawn_isolated_diarization())

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

    def test_transcribe_prepared_audio_retries_diarization_on_cpu_after_cuda_runtime_error(self) -> None:
        statuses: list[str] = []

        class _FakeSegment:
            def __init__(self, start: float, end: float, text: str) -> None:
                self.start = start
                self.end = end
                self.text = text

        class _FakeModel:
            def transcribe(self, *args, **kwargs):
                return iter([_FakeSegment(0.0, 1.0, "hello there")]), None

        attempts: list[str] = []

        def _fake_run(**kwargs):
            attempts.append(kwargs["device"])
            if kwargs["device"] == "cuda":
                raise RuntimeError("cuDNN error: CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH")
            return [{"start": 0.0, "end": 1.0, "speaker": "S1"}]

        spec = resolve_transcription_model("deepdml/faster-whisper-large-v3-turbo-ct2")

        with patch("services.transcription_service._probe_duration_seconds", return_value=8.0), patch(
            "services.transcription_service.load_audio_waveform",
            return_value=[0.1, 0.2],
        ), patch(
            "services.transcription_service._run_diarization_backend",
            side_effect=_fake_run,
        ):
            result = transcribe_prepared_audio(
                wav_path="prepared.wav",
                model=_FakeModel(),
                model_spec=spec,
                language=None,
                use_diarization=True,
                diar_backend="accurate",
                device="cuda",
                max_speakers=2,
                on_status=statuses.append,
            )

        self.assertEqual(attempts, ["cuda", "cpu"])
        self.assertIn("Diarization CUDA runtime unavailable. Retrying diarization on CPU", "\n".join(statuses))
        self.assertTrue(result.transcript.startswith("[S1] hello there"))

    def test_transcribe_prepared_audio_empty_diarization_keeps_plain_transcript(self) -> None:
        statuses: list[str] = []
        diar_progress: list[float] = []

        class _FakeSegment:
            def __init__(self, start: float, end: float, text: str) -> None:
                self.start = start
                self.end = end
                self.text = text

        class _FakeModel:
            def transcribe(self, *args, **kwargs):
                return iter(
                    [
                        _FakeSegment(0.0, 1.0, "this yes last night"),
                        _FakeSegment(1.0, 2.0, "yeah"),
                    ]
                ), None

        spec = resolve_transcription_model("deepdml/faster-whisper-large-v3-turbo-ct2")

        with patch("services.transcription_service._probe_duration_seconds", return_value=8.0), patch(
            "services.transcription_service.load_audio_waveform",
            return_value=[0.1, 0.2],
        ), patch(
            "services.transcription_service._run_diarization_backend",
            return_value=[],
        ):
            result = transcribe_prepared_audio(
                wav_path="prepared.wav",
                model=_FakeModel(),
                model_spec=spec,
                language=None,
                use_diarization=True,
                diar_backend="accurate",
                device="cpu",
                on_status=statuses.append,
                on_diar_progress=diar_progress.append,
            )

        self.assertEqual(result.transcript, "this yes last night yeah")
        self.assertNotIn("[S?]", result.transcript)
        self.assertEqual(result.segments[0].get("speaker"), None)
        self.assertIn("Diarization produced no speaker segments", "\n".join(statuses))
        self.assertEqual(diar_progress[-1], 0)

    def test_transcribe_prepared_audio_successful_diarization_formats_speaker_labels(self) -> None:
        class _FakeSegment:
            def __init__(self, start: float, end: float, text: str) -> None:
                self.start = start
                self.end = end
                self.text = text

        class _FakeModel:
            def transcribe(self, *args, **kwargs):
                return iter(
                    [
                        _FakeSegment(0.0, 1.0, "hello there"),
                        _FakeSegment(1.0, 2.0, "general update"),
                    ]
                ), None

        spec = resolve_transcription_model("deepdml/faster-whisper-large-v3-turbo-ct2")

        with patch("services.transcription_service._probe_duration_seconds", return_value=8.0), patch(
            "services.transcription_service.load_audio_waveform",
            return_value=[0.1, 0.2],
        ), patch(
            "services.transcription_service._run_diarization_backend",
            return_value=[
                {"start": 0.0, "end": 1.0, "speaker": "S1"},
                {"start": 1.0, "end": 2.0, "speaker": "S2"},
            ],
        ):
            result = transcribe_prepared_audio(
                wav_path="prepared.wav",
                model=_FakeModel(),
                model_spec=spec,
                language=None,
                use_diarization=True,
                diar_backend="accurate",
                device="cpu",
            )

        self.assertEqual(result.transcript, "[S1] hello there\n[S2] general update")
        self.assertNotIn("[S?]", result.transcript)


if __name__ == "__main__":
    unittest.main()
