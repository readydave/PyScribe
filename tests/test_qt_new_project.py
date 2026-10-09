"""New Project button: clears the current job state."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from PySide6.QtWidgets import QApplication, QMessageBox

from services.config_service import AppConfig
from services.model_service import RuntimeInfo
from ui_qt.main_window import MainWindow


class NewProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls._app = QApplication.instance() or QApplication([])

    def _build_window(self) -> MainWindow:
        runtime = RuntimeInfo(device="cpu", compute_type="int8", gpu_name="N/A", vram_gb=0.0, cpu_count=8)
        for item in [
            patch("ui_qt.main_window.detect_runtime", return_value=runtime),
            patch("ui_qt.main_window.load_config", return_value=AppConfig()),
            patch("ui_qt.main_window.save_config"),
            patch("ui_qt.main_window.list_live_audio_inputs", return_value=[]),
        ]:
            item.start()
            self.addCleanup(item.stop)
        window = MainWindow()
        self.addCleanup(window.close)
        return window

    def _fill(self, win: MainWindow) -> None:
        win.media_path = "/tmp/a.wav"
        win.path_label.setText("/tmp/a.wav")
        win.transcript_text = win.transcript_only_text = "hello"
        win.text_area.setPlainText("hello")
        win.save_btn.setEnabled(True)
        win.copy_btn.setEnabled(True)
        win.progress_bar.setValue(100)

    def test_confirmed_reset_clears_job_state(self) -> None:
        win = self._build_window()
        self._fill(win)
        with patch("ui_qt.main_window.QMessageBox.question", return_value=QMessageBox.Yes):
            win._on_new_project()
        self.assertIsNone(win.media_path)
        self.assertEqual(win.path_label.text(), "No file selected")
        self.assertEqual(win.text_area.toPlainText(), "")
        self.assertEqual(win.transcript_text, "")
        self.assertFalse(win.save_btn.isEnabled())
        self.assertEqual(win.progress_bar.value(), 0)
        self.assertEqual(win.status_label.text(), "Ready")

    def test_declined_reset_keeps_everything(self) -> None:
        win = self._build_window()
        self._fill(win)
        with patch("ui_qt.main_window.QMessageBox.question", return_value=QMessageBox.No):
            win._on_new_project()
        self.assertEqual(win.media_path, "/tmp/a.wav")
        self.assertEqual(win.text_area.toPlainText(), "hello")

    def test_no_prompt_when_nothing_to_discard(self) -> None:
        win = self._build_window()
        win.media_path = "/tmp/a.wav"
        with patch("ui_qt.main_window.QMessageBox.question") as ask:
            win._on_new_project()
        ask.assert_not_called()
        self.assertIsNone(win.media_path)

    def test_blocked_while_job_running(self) -> None:
        win = self._build_window()
        self._fill(win)
        with patch.object(win, "_is_transcription_running", return_value=True), patch(
            "ui_qt.main_window.QMessageBox.information"
        ) as info, patch("ui_qt.main_window.QMessageBox.question") as ask:
            win._on_new_project()
        info.assert_called_once()
        ask.assert_not_called()
        self.assertEqual(win.media_path, "/tmp/a.wav")


if __name__ == "__main__":
    unittest.main()
