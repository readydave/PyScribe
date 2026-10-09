from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from PySide6.QtWidgets import QApplication

from services.config_service import AppConfig
from services.model_service import RuntimeInfo
from ui_qt.main_window import TRANSCRIPT_MIN_WIDTH, MainWindow


class ActionRowFitTests(unittest.TestCase):
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
        self.addCleanup(window.close)
        return window

    def _settle(self, win: MainWindow, width: int, height: int) -> None:
        win.resize(width, height)
        for _ in range(8):
            QApplication.processEvents()

    def _action_buttons(self, win: MainWindow) -> list:
        return [
            win.transcribe_btn, win.stop_live_btn, win.pause_live_btn, win.cancel_btn, win.force_stop_btn,
            win.save_btn, win.rename_with_title_btn, win.open_btn, win.copy_btn,
        ]

    def _assert_fits(self, win: MainWindow) -> None:
        host_rect = win.dock_host.rect()
        for btn in self._action_buttons(win):
            if not btn.isVisible():
                continue
            rect = btn.rect()
            rect.moveTopLeft(btn.mapTo(win.dock_host, rect.topLeft()))
            self.assertTrue(host_rect.contains(rect), f"{btn.text()} is clipped: {rect} not in {host_rect}")
            self.assertGreaterEqual(btn.width(), btn.sizeHint().width(), f"{btn.text()} is narrower than its size hint")

    def test_buttons_fit_at_minimum_size_in_every_mode(self) -> None:
        win = self._build_window()
        for live in (0, 1):
            win.input_mode_combo.setCurrentIndex(live)
            for collapsed in (False, True):
                if win._sidebar_collapsed != collapsed:
                    win._toggle_sidebar_collapsed()
                self._settle(win, 840, 560)
                with self.subTest(live=bool(live), collapsed=collapsed):
                    self._assert_fits(win)
                    self.assertTrue(win.force_stop_btn.isVisible())

    def test_row_wraps_only_when_narrow(self) -> None:
        win = self._build_window()
        win.input_mode_combo.setCurrentIndex(1)
        self._settle(win, 1400, 800)
        job_buttons = [b for b in (win.transcribe_btn, win.stop_live_btn, win.pause_live_btn, win.cancel_btn, win.force_stop_btn) if b.isVisible()]
        self.assertEqual({b.y() for b in job_buttons}, {job_buttons[0].y()})
        self._settle(win, 840, 560)
        self.assertGreater(len({b.y() for b in job_buttons}), 1)

    def test_wrapping_keeps_button_order(self) -> None:
        win = self._build_window()
        win.input_mode_combo.setCurrentIndex(1)
        self._settle(win, 840, 560)
        order = [win.transcribe_btn, win.stop_live_btn, win.pause_live_btn, win.cancel_btn, win.force_stop_btn]
        positions = [(b.y(), b.x()) for b in order if b.isVisible()]
        self.assertEqual(positions, sorted(positions))

    def test_transcript_floor_is_the_named_constant(self) -> None:
        win = self._build_window()
        self.assertEqual(win.dock_host.centralWidget().minimumWidth(), TRANSCRIPT_MIN_WIDTH)


if __name__ == "__main__":
    unittest.main()
