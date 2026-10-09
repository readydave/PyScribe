"""Load and Save rows in the Qt progress timeline."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from PySide6.QtWidgets import QApplication

from services.config_service import AppConfig
from services.model_service import RuntimeInfo
from ui_qt.job_stages import Stage
from ui_qt.main_window import MainWindow


class TimelineStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls._app = QApplication.instance() or QApplication([])

    def _build_window(self) -> MainWindow:
        runtime = RuntimeInfo(device="cpu", compute_type="int8", gpu_name="N/A", vram_gb=0.0, cpu_count=8)
        patches = [
            patch("ui_qt.main_window.detect_runtime", return_value=runtime),
            patch("ui_qt.main_window.load_config", return_value=AppConfig()),
            patch("ui_qt.main_window.save_config"),
            patch("ui_qt.main_window.list_live_audio_inputs", return_value=[]),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        window = MainWindow()
        window.show()
        QApplication.processEvents()
        self.addCleanup(window.close)
        return window

    def test_load_and_save_rows_hidden_when_idle(self) -> None:
        win = self._build_window()
        self.assertFalse(win.load_progress_bar.isVisibleTo(win))
        self.assertFalse(win.save_progress_bar.isVisibleTo(win))

    def test_stage_event_drives_load_row(self) -> None:
        win = self._build_window()
        win.job_tracker.reset({Stage.LOAD, Stage.TRANSCRIBE})
        win._on_stage_event("load:start")
        self.assertEqual(win.job_tracker.info(Stage.LOAD).state, "active")
        self.assertEqual(win.load_progress_bar.maximum(), 0)
        win._on_stage_event("load:done")
        self.assertEqual(win.job_tracker.info(Stage.LOAD).state, "done")
        self.assertEqual(win.load_progress_bar.maximum(), 100)
        self.assertEqual(win.load_progress_bar.value(), 100)

    def test_unknown_stage_event_is_ignored(self) -> None:
        win = self._build_window()
        win.job_tracker.reset({Stage.LOAD})
        win._on_stage_event("bogus:start")
        self.assertEqual(win.job_tracker.info(Stage.LOAD).state, "pending")

    def test_auto_save_runs_inside_save_stage(self) -> None:
        win = self._build_window()
        win.job_tracker.reset({Stage.TRANSCRIBE, Stage.SAVE})
        seen: list[str] = []
        with patch.object(
            win, "_auto_save_completed_parts",
            side_effect=lambda **_: seen.append(win.job_tracker.info(Stage.SAVE).state),
        ):
            win._run_auto_save_stage(transcript="a", transcript_only="a", visual_report="")
        self.assertEqual(seen, ["active"])
        self.assertEqual(win.job_tracker.info(Stage.SAVE).state, "done")

    def test_save_stage_skipped_when_disabled(self) -> None:
        win = self._build_window()
        win.job_tracker.reset({Stage.TRANSCRIBE})
        with patch.object(win, "_auto_save_completed_parts") as auto:
            win._run_auto_save_stage(transcript="a", transcript_only="a", visual_report="")
        auto.assert_called_once()
        self.assertEqual(win.job_tracker.info(Stage.SAVE).state, "disabled")


if __name__ == "__main__":
    unittest.main()
