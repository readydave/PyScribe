from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from PySide6.QtWidgets import QApplication, QDockWidget

from services.config_service import AppConfig
from services.model_service import RuntimeInfo
from ui_qt.main_window import MainWindow


class DockLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls._app = QApplication.instance() or QApplication([])

    def _build_window(self, config: AppConfig | None = None) -> tuple[MainWindow, list[AppConfig]]:
        saved: list[AppConfig] = []
        runtime = RuntimeInfo(device="cpu", compute_type="int8", gpu_name="N/A", vram_gb=0.0, cpu_count=8)
        patches = [
            patch("ui_qt.main_window.detect_runtime", return_value=runtime),
            patch("ui_qt.main_window.load_config", return_value=config or AppConfig()),
            patch("ui_qt.main_window.save_config", side_effect=lambda cfg: saved.append(AppConfig(**cfg.__dict__))),
            patch("ui_qt.main_window.list_live_audio_inputs", return_value=[]),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        window = MainWindow()
        window.show()
        QApplication.processEvents()
        self.addCleanup(window.close)
        return window, saved

    def test_default_layout_has_all_panels_visible(self) -> None:
        win, _ = self._build_window()
        names = {dock.objectName() for dock in win.docks}
        self.assertEqual(names, {"dock_setup", "dock_progress", "dock_hardware", "dock_queue"})
        self.assertTrue(win.setup_dock.isVisible())
        self.assertTrue(win.progress_dock.isVisible())
        self.assertTrue(win.hardware_dock.isVisible())

    def test_lock_layout_disables_moving_and_floating(self) -> None:
        win, _ = self._build_window()
        win.lock_layout_action.setChecked(True)
        for dock in win.docks:
            self.assertEqual(dock.features(), QDockWidget.NoDockWidgetFeatures)
        win.lock_layout_action.setChecked(False)
        for dock in win.docks:
            self.assertTrue(dock.features() & QDockWidget.DockWidgetMovable)

    def test_layout_round_trips_through_config(self) -> None:
        win, saved = self._build_window()
        win.hardware_dock.hide()
        win._save_dock_layout()
        state = saved[-1].dock_layout
        self.assertTrue(state)

        restored, _ = self._build_window(AppConfig(dock_layout=state))
        QApplication.processEvents()
        self.assertFalse(restored.hardware_dock.isVisible())
        self.assertTrue(restored.setup_dock.isVisible())

    def test_incompatible_saved_layout_falls_back_to_default(self) -> None:
        win, _ = self._build_window(AppConfig(dock_layout="bm90IGEgbGF5b3V0"))
        QApplication.processEvents()
        self.assertTrue(win.setup_dock.isVisible())
        self.assertTrue(win.hardware_dock.isVisible())

    def test_reset_layout_shows_closed_panels(self) -> None:
        win, _ = self._build_window()
        win.hardware_dock.hide()
        win._reset_dock_layout()
        QApplication.processEvents()
        self.assertTrue(win.hardware_dock.isVisible())

    def test_input_segment_follows_input_mode_combo(self) -> None:
        win, _ = self._build_window()
        win.input_mode_combo.setCurrentIndex(1)
        self.assertTrue(win.input_segment_buttons[1].isChecked())
        win.input_segment_buttons[0].click()
        self.assertEqual(win.input_mode_combo.currentIndex(), 0)

    def test_hardware_samples_reach_panel_and_idle_clears_value(self) -> None:
        win, _ = self._build_window()
        win.hw_sample.emit({"cpu": 40.0, "ram": 50.0})
        QApplication.processEvents()
        self.assertEqual(win.hw_panel.row("cpu").sample_count, 1)
        self.assertFalse(win.hw_panel.row("gpu").isVisible())
        win.hw_sample.emit({"cpu": 10.0, "ram": 20.0, "gpu": 55.0, "vram_used": 3.0, "vram_total": 12.0})
        QApplication.processEvents()
        self.assertTrue(win.hw_panel.row("vram").isVisibleTo(win.hw_panel))
        win.hw_sample.emit({})
        QApplication.processEvents()
        self.assertEqual(win.hw_panel.stage_label.text(), "Idle")

    def test_advanced_options_collapsed_by_default_and_remembered(self) -> None:
        win, saved = self._build_window()
        self.assertFalse(win.advanced_body.isVisibleTo(win.advanced_options_card))
        win.advanced_toggle.setChecked(True)
        self.assertTrue(saved[-1].setup_advanced_expanded)


if __name__ == "__main__":
    unittest.main()
