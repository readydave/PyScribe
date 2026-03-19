"""Tests for the backend-neutral benchmark worker."""

from __future__ import annotations

import os
import threading
import unittest
from unittest.mock import patch

try:
    from PySide6.QtWidgets import QApplication
except ImportError:  # pragma: no cover - optional in lightweight environments
    QApplication = None

from services.transcription_service import TranscriptionResult
from services.model_service import RuntimeInfo, resolve_transcription_model

if QApplication is not None:
    from ui_qt.benchmark_dialog import BenchmarkWorker
else:  # pragma: no cover - optional in lightweight environments
    BenchmarkWorker = None


@unittest.skipIf(QApplication is None or BenchmarkWorker is None, "PySide6 is not installed")
class BenchmarkDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls._app = QApplication.instance() or QApplication([])

    def test_benchmark_worker_uses_backend_neutral_transcription_path(self) -> None:
        runtime = RuntimeInfo(
            device="cpu",
            compute_type="int8",
            gpu_name="N/A",
            vram_gb=0.0,
            cpu_count=8,
        )
        worker = BenchmarkWorker(
            selected_models=["ibm-granite/granite-4.0-1b-speech"],
            runtime=runtime,
            cancel_event=threading.Event(),
            audio_language="en",
        )
        result_lines: list[str] = []
        worker.result_line.connect(result_lines.append)

        with patch("ui_qt.benchmark_dialog.os.path.exists", return_value=True), patch(
            "ui_qt.benchmark_dialog.get_ffmpeg_cmd",
            return_value="ffmpeg",
        ), patch(
            "ui_qt.benchmark_dialog.convert_to_16k_mono",
            return_value="prepared.wav",
        ), patch(
            "ui_qt.benchmark_dialog.ensure_model_cached",
            return_value="C:\\cache\\granite",
        ), patch(
            "ui_qt.benchmark_dialog.resolve_transcription_model",
            return_value=resolve_transcription_model("ibm-granite/granite-4.0-1b-speech"),
        ), patch(
            "ui_qt.benchmark_dialog.load_model",
            return_value=object(),
        ), patch(
            "ui_qt.benchmark_dialog.transcribe_prepared_audio",
            return_value=TranscriptionResult(
                transcript="Granite benchmark transcript",
                transcript_only="Granite benchmark transcript",
                visual_report="",
                segments=[],
                cancelled=False,
                duration_seconds=8.0,
                transcription_seconds=1.2,
                diarization_seconds=0.0,
                visual_analysis_seconds=0.0,
            ),
        ) as transcribe_mock:
            worker.run()

        self.assertEqual(transcribe_mock.call_count, 1)
        self.assertTrue(result_lines)
        self.assertIn("Granite benchmark transcript", result_lines[0])


if __name__ == "__main__":
    unittest.main()
