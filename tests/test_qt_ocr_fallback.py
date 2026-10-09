"""Qt OCR fallback setting: persisted on change and before a job starts (the worker reads it from disk)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from services import AppConfig
from services.model_service import RuntimeInfo
from ui_qt.main_window import MainWindow


class OcrFallbackControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _build_window(self, config: AppConfig) -> tuple[MainWindow, list[str]]:
        saved: list[str] = []
        runtime = RuntimeInfo(
            device="cpu", compute_type="int8", gpu_name="N/A", vram_gb=0.0, cpu_count=8
        )
        patches = [
            patch("ui_qt.main_window.detect_runtime", return_value=runtime),
            patch("ui_qt.main_window.load_config", return_value=config),
            patch("ui_qt.main_window.save_config", side_effect=lambda cfg, *a, **k: saved.append(cfg.visual_ocr_fallback)),
            patch("ui_qt.main_window.list_live_audio_inputs", return_value=[]),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        window = MainWindow()
        self.addCleanup(window.close)
        return window, saved

    def test_combo_defaults_to_auto_and_loads_saved_value(self) -> None:
        win, _ = self._build_window(AppConfig())
        self.assertEqual(win.visual_fallback_combo.currentData(), "auto")
        win2, _ = self._build_window(AppConfig(visual_ocr_fallback="pytesseract"))
        self.assertEqual(win2.visual_fallback_combo.currentData(), "pytesseract")

    def test_changing_the_combo_saves_immediately(self) -> None:
        win, saved = self._build_window(AppConfig())
        win.visual_fallback_combo.setCurrentIndex(win.visual_fallback_combo.findData("pytesseract"))
        self.assertEqual(saved[-1], "pytesseract")
        self.assertEqual(win.config.visual_ocr_fallback, "pytesseract")

    def test_value_is_saved_before_the_job_launches(self) -> None:
        win, saved = self._build_window(AppConfig(run_mode="visual_only", use_visual_analysis=True))
        win.visual_fallback_combo.blockSignals(True)  # prove the start path saves it, not the change handler
        win.visual_fallback_combo.setCurrentIndex(win.visual_fallback_combo.findData("rapidocr"))
        win.visual_fallback_combo.blockSignals(False)
        with tempfile.TemporaryDirectory() as tmp:
            media = Path(tmp) / "talk.mp4"
            media.write_bytes(b"x")
            win.media_path = str(media)
            saved_at_launch: list[str] = []
            with (
                patch.object(win, "_launch_transcription_worker", side_effect=lambda **k: saved_at_launch.append(saved[-1])),
                patch("ui_qt.main_window.check_ocr_backend_ready", return_value=(True, None)),
                patch.object(win, "_confirm_visual_backend_download", return_value=True),
                patch("ui_qt.main_window.QMessageBox.warning", side_effect=AssertionError("unexpected dialog")),
            ):
                win.start_transcription()
        self.assertEqual(saved_at_launch, ["rapidocr"])


if __name__ == "__main__":
    unittest.main()
