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

    def test_colour_theme_menu_switches_theme_and_persists_choice(self) -> None:
        from ui_qt import theme

        win, saved = self._build_window()
        self.addCleanup(theme.apply_theme, self._app, "system")
        labels = [action.text() for action in win.colour_theme_group.actions()]
        self.assertEqual(labels, ["Iron-gall", "Verdigris", "Ochre", "Graphite"])
        before = theme.active_palette().page
        ochre = next(a for a in win.colour_theme_group.actions() if a.text() == "Ochre")
        ochre.trigger()
        self.assertEqual(saved[-1].theme_id, "ochre")
        self.assertEqual(theme.active_theme().id, "ochre")
        self.assertNotEqual(theme.active_palette().page, before)
        self.assertIn(theme.active_palette().page, self._app.styleSheet())

    def test_theme_editor_save_persists_and_cancel_restores(self) -> None:
        from unittest.mock import MagicMock

        from PySide6.QtWidgets import QDialog

        from ui_qt import theme

        win, saved = self._build_window()
        self.addCleanup(theme.apply_theme, self._app, "system")

        class FakeDialog:
            previewApplied = MagicMock()
            selected_theme_id = "graphite"
            custom_themes: list[dict] = []

            def __init__(self, *args: object) -> None:
                pass

            def exec(self) -> int:
                theme.apply_theme(self_app, "light", "graphite", [])  # unsaved preview
                return self.result_code

        self_app = self._app
        FakeDialog.result_code = QDialog.Rejected
        with patch("ui_qt.main_window.ThemeEditorDialog", FakeDialog):
            win.open_theme_editor()
        self.assertEqual(theme.active_theme().id, "iron-gall")  # preview undone
        self.assertFalse(saved)

        FakeDialog.result_code = QDialog.Accepted
        with patch("ui_qt.main_window.ThemeEditorDialog", FakeDialog):
            win.open_theme_editor()
        self.assertEqual(saved[-1].theme_id, "graphite")
        self.assertEqual(theme.active_theme().id, "graphite")

    def test_recording_accent_follows_live_capture_state(self) -> None:
        win, _ = self._build_window()
        win._update_live_mode_ui()
        self.assertEqual(win.live_timer_label.property("state"), "idle")
        win._live_capture_active = True
        self.addCleanup(setattr, win, "_live_capture_active", False)
        win._update_live_mode_ui()
        self.assertEqual(win.live_timer_label.property("state"), "recording")
        self.assertEqual(win.stop_live_btn.property("role"), "primary")
        win._live_paused = True
        self.addCleanup(setattr, win, "_live_paused", False)
        win._update_live_mode_ui()
        self.assertEqual(win.live_timer_label.property("state"), "idle")

    def test_forced_theme_sets_native_color_scheme(self) -> None:
        from PySide6.QtCore import Qt

        from ui_qt import theme

        hints = self._app.styleHints()
        self.addCleanup(theme.apply_theme, self._app, "system")
        self.assertEqual(theme.apply_theme(self._app, "dark"), "dark")
        self.assertEqual(theme.apply_theme(self._app, "light"), "light")
        # Headless platforms ignore colour-scheme overrides; real ones apply them asynchronously.
        if not hasattr(hints, "setColorScheme") or self._app.platformName() in {"offscreen", "minimal"}:
            return
        theme.apply_theme(self._app, "dark")
        QApplication.processEvents()
        self.assertEqual(hints.colorScheme(), Qt.ColorScheme.Dark)

    def test_advanced_options_collapsed_by_default_and_remembered(self) -> None:
        win, saved = self._build_window()
        self.assertFalse(win.advanced_body.isVisibleTo(win.advanced_options_card))
        win.advanced_toggle.setChecked(True)
        self.assertTrue(saved[-1].setup_advanced_expanded)


if __name__ == "__main__":
    unittest.main()
