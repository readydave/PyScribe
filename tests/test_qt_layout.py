from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PySide6.QtWidgets import QApplication

from services.config_service import AppConfig
from services.model_service import RuntimeInfo
from services.ui_themes import DEFAULT_THEME_ID
from ui_qt.components import tokens
from ui_qt import theme
from ui_qt.job_stages import Stage
from ui_qt.main_window import TRANSCRIPT_MIN_WIDTH, MainWindow
from qt_close import close_and_drain


class LayoutTests(unittest.TestCase):
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
        window.main_stack.setCurrentIndex(1)  # Transcribe page (the app opens on Projects)
        window.show()
        QApplication.processEvents()
        self.addCleanup(close_and_drain, window)
        return window, saved

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
        self.assertEqual(labels, ["Graphite Studio", "Iron-gall", "Verdigris", "Ochre", "Graphite"])
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
        self.assertEqual(theme.active_theme().id, DEFAULT_THEME_ID)  # preview undone
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

    def _window_1100(self) -> tuple[MainWindow, list[AppConfig]]:
        win, saved = self._build_window()
        win.resize(1100, 700)
        QApplication.processEvents()
        return win, saved

    # --- the redesigned layout -----------------------------------------------------------------------

    def test_there_are_no_docks_and_old_dock_settings_are_ignored(self) -> None:
        win, _ = self._build_window(AppConfig(dock_layout="bm90IGEgbGF5b3V0", dock_locked=True))
        for name in ("dock_host", "docks", "setup_dock", "progress_dock", "hardware_dock", "queue_dock"):
            self.assertFalse(hasattr(win, name), name)
        self.assertTrue(win.transcription_scroll.isVisible())
        self.assertTrue(win.main_surface.isVisible())

    def test_transcribe_page_keeps_setup_and_transcript_readable_at_1100(self) -> None:
        win, _ = self._window_1100()
        self.assertGreaterEqual(win.transcription_scroll.width(), 340)
        self.assertGreaterEqual(win.main_surface.width(), TRANSCRIPT_MIN_WIDTH)
        self.assertEqual(win.main_surface.minimumWidth(), TRANSCRIPT_MIN_WIDTH)

    def test_status_line_is_visible_without_opening_anything(self) -> None:
        win, _ = self._build_window()
        win.status_label.setText("Selected: meeting.wav")
        self.assertTrue(win.status_label.isVisible())
        self.assertFalse(win.details_drawer.isVisible())
        self.assertFalse(win.performance_drawer.isVisible())

    def test_splitter_handle_is_styled(self) -> None:
        for mode in ("light", "dark"):
            qss = theme.build_qss(mode)
            self.assertIn("QSplitter::handle:horizontal", qss)
            self.assertNotIn("QDockWidget", qss)

    # --- job status ------------------------------------------------------------------------------------

    def _running(self, win: MainWindow) -> None:
        win.worker_thread = SimpleNamespace(isRunning=lambda: True, quit=lambda: None, wait=lambda *_: True)
        self.addCleanup(setattr, win, "worker_thread", None)

    def test_status_bar_is_hidden_when_idle(self) -> None:
        win, _ = self._build_window()
        self.assertFalse(win.job_status.isVisible())
        self.assertFalse(win.cancel_btn.isEnabled())

    def test_status_bar_shows_stage_progress_and_elapsed_while_a_job_runs(self) -> None:
        win, _ = self._build_window()
        self._running(win)
        win.job_tracker.reset({Stage.TRANSCRIBE, Stage.SPEAKERS})
        win.job_tracker.start(Stage.TRANSCRIBE)
        win.job_tracker.progress(Stage.TRANSCRIBE, 37)
        win._sync_timeline()
        self.assertTrue(win.job_status.isVisible())
        self.assertEqual(win.job_status.stage_label.text(), "Transcribing")
        self.assertEqual(win.job_status.progress.value(), 37)
        self.assertRegex(win.job_status.elapsed_label.text(), r"^\d+:\d{2}$")
        win.job_tracker.complete(Stage.TRANSCRIBE, 1.0)
        win.job_tracker.start(Stage.SPEAKERS)
        win._sync_timeline()
        self.assertEqual(win.job_status.stage_label.text(), "Identifying speakers")
        win.worker_thread = None
        win._update_live_mode_ui()
        self.assertFalse(win.job_status.isVisible())

    def test_status_bar_before_the_first_stage_starts_and_for_live_capture(self) -> None:
        win, _ = self._build_window()
        self._running(win)
        win.job_tracker.reset({Stage.TRANSCRIBE})
        win._sync_timeline()
        self.assertEqual(win.job_status.stage_label.text(), "Starting…")
        win.worker_thread = None
        win._live_capture_active = True
        self.addCleanup(setattr, win, "_live_capture_active", False)
        win._update_live_mode_ui()
        self.assertEqual(win.job_status.stage_label.text(), "Recording")
        win._live_paused = True
        self.addCleanup(setattr, win, "_live_paused", False)
        win._update_live_mode_ui()
        self.assertEqual(win.job_status.stage_label.text(), "Recording (paused)")

    def test_cancel_lives_on_the_status_bar_and_force_stop_in_details(self) -> None:
        win, _ = self._build_window()
        self.assertIs(win.cancel_btn, win.job_status.cancel_button)
        parent = win.force_stop_btn.parent()
        while parent is not None and parent is not win.details_drawer:
            parent = parent.parent()
        self.assertIs(parent, win.details_drawer)

    def test_cancel_button_calls_cancel_transcription(self) -> None:
        win, _ = self._build_window()
        with patch.object(MainWindow, "cancel_transcription") as cancel:
            win.cancel_btn.setEnabled(True)
            win.cancel_btn.click()
        cancel.assert_called_once()

    # --- drawers ---------------------------------------------------------------------------------------

    def test_details_and_performance_drawers_exclude_each_other(self) -> None:
        win, _ = self._build_window()
        win.job_status.details_button.click()
        self.assertTrue(win.details_drawer.is_open())
        self.assertTrue(win.job_details_action.isChecked())
        win.performance_button.click()
        self.assertTrue(win.performance_drawer.is_open())
        self.assertFalse(win.details_drawer.is_open())
        self.assertFalse(win.job_details_action.isChecked())
        self.assertTrue(win.performance_action.isChecked())
        win.performance_action.trigger()
        self.assertFalse(win.performance_drawer.is_open())

    def test_drawers_hold_the_timeline_log_and_hardware_panel(self) -> None:
        win, _ = self._build_window()
        win._set_drawer(win.details_drawer, True)
        QApplication.processEvents()
        self.assertTrue(win.job_timeline.isVisibleTo(win.details_drawer))
        win._append_terminal_log("hello from the pipeline")
        self.assertIn("hello from the pipeline", win.terminal_log.toPlainText())
        win._set_drawer(win.performance_drawer, True)
        QApplication.processEvents()
        self.assertTrue(win.hw_panel.isVisibleTo(win.performance_drawer))

    # --- batch queue -----------------------------------------------------------------------------------

    def test_batch_queue_is_collapsed_until_files_are_added(self) -> None:
        import tempfile
        from pathlib import Path

        win, _ = self._build_window()
        self.assertFalse(win.queue_card.isVisibleTo(win.transcription_scroll))
        self.assertEqual(win.queue_toggle.text(), "Batch queue")
        with tempfile.TemporaryDirectory() as tmp:
            media = Path(tmp) / "a.wav"
            media.write_bytes(b"RIFF")
            win._handle_incoming_paths([str(media)])
        self.assertTrue(win.queue_toggle.isChecked())
        self.assertTrue(win.queue_card.isVisibleTo(win.transcription_scroll))
        self.assertEqual(win.queue_toggle.text(), "Batch queue (1)")

    # --- settings --------------------------------------------------------------------------------------

    def test_theme_segment_and_menu_stay_in_sync(self) -> None:
        win, saved = self._build_window()
        self.addCleanup(theme.apply_theme, self._app, "system")
        win.theme_segment._buttons["dark"].click()
        self.assertEqual(win.theme_mode, "dark")
        self.assertEqual(saved[-1].theme_mode, "dark")
        self.assertTrue(win.theme_actions["dark"].isChecked())
        win.theme_actions["light"].trigger()
        self.assertEqual(win.theme_segment.current(), "light")

    def test_colour_combo_lists_themes_and_switches(self) -> None:
        win, saved = self._build_window()
        self.addCleanup(theme.apply_theme, self._app, "system")
        names = [win.colour_theme_combo.itemText(i) for i in range(win.colour_theme_combo.count())]
        self.assertEqual(names, ["Graphite Studio", "Iron-gall", "Verdigris", "Ochre", "Graphite"])
        self.assertEqual(win.colour_theme_combo.currentData(), DEFAULT_THEME_ID)
        win.colour_theme_combo.activated.emit(3)
        self.assertEqual(saved[-1].theme_id, "ochre")
        self.assertEqual(theme.active_theme().id, "ochre")

    def test_transcript_text_size_setting_updates_config_styles_and_studio(self) -> None:
        win, saved = self._build_window()
        self.addCleanup(theme.apply_theme, self._app, "system")
        self.addCleanup(tokens.set_transcript_px, tokens.TRANSCRIPT_PX_DEFAULT)
        with patch.object(win.studio, "refresh_text_size") as refresh:
            win.transcript_size_spin.setValue(20)
        self.assertEqual(saved[-1].transcript_font_px, 20)
        self.assertIn("font-size: 20px", self._app.styleSheet())
        refresh.assert_called_once()

    def test_reduced_motion_setting_applies_immediately(self) -> None:
        win, saved = self._build_window()
        self.addCleanup(tokens.configure_motion, "auto")
        win.reduced_motion_combo.setCurrentIndex(win.reduced_motion_combo.findData("on"))
        self.assertEqual(saved[-1].reduced_motion, "on")
        self.assertTrue(tokens.reduced_motion())
        win.reduced_motion_combo.setCurrentIndex(win.reduced_motion_combo.findData("off"))
        self.assertFalse(tokens.reduced_motion())

    def test_projects_folder_can_be_changed_and_reset(self) -> None:
        import os
        import tempfile
        from pathlib import Path

        from services.project_store import DEFAULT_PROJECTS_ROOT

        win, saved = self._build_window()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"PYSCRIBE_PROJECTS_DIR": ""}):
            win._set_projects_dir(tmp)
            self.assertEqual(win.project_store.root, Path(tmp))
            self.assertEqual(win.studio_controller._store.root, Path(tmp))
            self.assertEqual(saved[-1].projects_dir, tmp)
            self.assertEqual(win.projects_dir_input.text(), tmp)
            win._set_projects_dir(None)
            self.assertEqual(win.project_store.root, DEFAULT_PROJECTS_ROOT)
            self.assertIsNone(saved[-1].projects_dir)

    def test_projects_folder_is_locked_by_the_environment_variable(self) -> None:
        win, _ = self._build_window()  # conftest sets PYSCRIBE_PROJECTS_DIR for every test
        self.assertFalse(win.projects_dir_browse_btn.isEnabled())
        self.assertFalse(win.projects_dir_reset_btn.isEnabled())
        self.assertIn("PYSCRIBE_PROJECTS_DIR", win.projects_dir_input.toolTip())

    def test_projects_folder_cannot_change_during_a_job(self) -> None:
        win, saved = self._build_window()
        self._running(win)
        before = win.project_store
        with patch("ui_qt.main_window.QMessageBox.information") as info:
            win._set_projects_dir("/tmp/somewhere")
        info.assert_called_once()
        self.assertIs(win.project_store, before)

    def test_diagnostics_buttons_open_the_right_things(self) -> None:
        win, _ = self._build_window()
        win.nav_rail.set_current("settings")
        buttons = {b.text(): b for b in win.findChildren(type(win.projects_dir_browse_btn)) if b.text()}
        buttons["Performance panel"].click()
        self.assertTrue(win.performance_drawer.is_open())
        with patch.object(MainWindow, "open_benchmark_dialog") as bench:
            buttons["Run benchmark…"].click()
        bench.assert_called_once()

    # --- templates -------------------------------------------------------------------------------------

    def test_use_template_opens_the_dialog_with_that_template(self) -> None:
        win, _ = self._build_window()
        seen: dict[str, object] = {}

        class FakeDialog:
            def __init__(self, **kwargs: object) -> None:
                seen.update(kwargs)

            def exec(self) -> int:
                return 0

        win.transcript_text = "[S1] hello"
        with patch("ui_qt.main_window.LLMPostprocessDialog", FakeDialog), patch.object(win, "_ensure_llm_profiles", return_value=True):
            win._use_template("action-items")
        self.assertEqual(seen["initial_template_id"], "action-items")
        self.assertFalse(seen["prefer_loaded_transcript"])
        seen.clear()
        win.transcript_text = ""
        with patch("ui_qt.main_window.LLMPostprocessDialog", FakeDialog), patch.object(win, "_ensure_llm_profiles", return_value=True):
            win._use_template("decision-log")
        self.assertTrue(seen["prefer_loaded_transcript"])

    def test_studio_messages_reach_the_status_bar(self) -> None:
        win, _ = self._build_window()
        win._on_studio_status("Exported notes.srt")
        self.assertEqual(win.status_label.text(), "Exported notes.srt")
        self.assertEqual(win.statusBar().currentMessage(), "Exported notes.srt")


if __name__ == "__main__":
    unittest.main()
