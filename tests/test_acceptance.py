"""Acceptance checks from the UX/UI redesign specification, run against the real window (offscreen)."""

from __future__ import annotations

import math
import os
import struct
import tempfile
import time
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget

from services import export_service
from services.config_service import AppConfig
from services.model_service import RuntimeInfo
from services.project_store import sanitize_segments
from services.ui_themes import PRESETS, contrast, mix, resolve
from ui_qt import theme
from ui_qt.components import tokens
from ui_qt.main_window import MainWindow
from ui_qt.views.studio import controller as controller_module
from ui_qt.views.studio.segment_model import NameRole
from qt_close import close_and_drain

SEGMENTS = sanitize_segments(
    [
        {"start": 0.0, "end": 6.0, "speaker": "S1", "text": "Thanks everyone for joining the quarterly review."},
        {"start": 6.0, "end": 14.0, "speaker": "S2", "text": "Revenue grew twelve percent, driven by enterprise."},
        {"start": 14.0, "end": 22.0, "speaker": "S3", "text": "What does churn look like for the enterprise segment?"},
        {"start": 22.0, "end": 30.0, "speaker": "S2", "text": "Churn is flat quarter over quarter."},
    ]
)


def write_wav(path: Path, seconds: float = 30.0) -> None:
    rate = 8000
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"".join(struct.pack("<h", int(7000 * math.sin(i / 10.0))) for i in range(int(rate * seconds))))


def pump(app: QApplication, condition=lambda: False, timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return condition()


class AcceptanceCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(theme.apply_theme, self._app, "system")
        self.addCleanup(tokens.configure_motion, "auto")
        self.addCleanup(tokens.set_transcript_px, tokens.TRANSCRIPT_PX_DEFAULT)

    def _window(self, config: AppConfig | None = None, size: tuple[int, int] = (1366, 768)) -> MainWindow:
        runtime = RuntimeInfo(device="cpu", compute_type="int8", gpu_name="N/A", vram_gb=0.0, cpu_count=8)
        patches = [
            patch("ui_qt.main_window.detect_runtime", return_value=runtime),
            patch("ui_qt.main_window.load_config", return_value=config or AppConfig()),
            patch("ui_qt.main_window.save_config"),
            patch("ui_qt.main_window.list_live_audio_inputs", return_value=[]),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        window = MainWindow()
        window.resize(*size)
        window.show()
        QApplication.processEvents()
        self.addCleanup(close_and_drain, window)
        return window

    def _project(self, win: MainWindow, media: str = "", **changes: object):
        project = win.project_store.create(title="Quarterly Review", media_path=media)
        values: dict[str, object] = dict(status="completed", segments=SEGMENTS, duration_s=30.0, transcript_text="x")
        values.update(changes)
        win.project_store.update(project.id, **values)
        win._refresh_projects()
        return project


class FirstScreenTests(AcceptanceCase):
    def test_a_new_user_can_import_or_record_without_seeing_advanced_configuration(self) -> None:
        win = self._window()
        self.assertIs(win.main_stack.currentWidget(), win.projects_home)
        home = win.projects_home
        for action in (home.drop_card, home.live_card, home.import_card):
            self.assertTrue(action.isVisibleTo(win))
            self.assertTrue(action.focusPolicy() & Qt.TabFocus)
        for advanced in (win.advanced_toggle, win.model_combo, win.diar_checkbox, win.hotwords_input, win.visual_checkbox,
                         win.live_compute_combo, win.hw_panel, win.job_timeline):
            self.assertFalse(advanced.isVisibleTo(win), type(advanced).__name__)

    def test_choosing_a_file_leads_to_setup_with_advanced_options_still_tucked_away(self) -> None:
        win = self._window()
        media = Path(self._tmp.name) / "call.wav"
        write_wav(media, 2)
        win.projects_home.file_chosen.emit(str(media))
        self.assertEqual(win.nav_rail.current(), "transcribe")
        self.assertEqual(win.media_path, str(media))
        self.assertTrue(win.model_combo.isVisibleTo(win))
        self.assertFalse(win.advanced_body.isVisibleTo(win))
        self.assertFalse(win.details_drawer.isVisible())

    def test_live_capture_starts_from_the_first_screen_with_tuning_hidden(self) -> None:
        win = self._window()
        win.projects_home.live_requested.emit()
        self.assertTrue(win._is_live_mode())
        self.assertTrue(win.live_device_combo.isVisibleTo(win))
        self.assertFalse(win.live_compute_combo.isVisibleTo(win))


class StudioWorkflowTests(AcceptanceCase):
    def test_search_play_seek_identify_speaker_inspect_ai_and_export_from_one_workspace(self) -> None:
        win = self._window()
        media = Path(self._tmp.name) / "review.wav"
        write_wav(media, 30)
        project = self._project(
            win, str(media), speaker_names={"S2": "Raj"},
            ai_outputs={"meeting-summary": {"text": "Revenue up 12%.", "created": "", "profile": ""}},
        )
        QTest.mouseClick(win.projects_home._cards[project.id], Qt.LeftButton)
        studio = win.studio
        self.assertIs(win.main_stack.currentWidget(), studio)
        self.assertTrue(pump(self._app, lambda: studio.transport.waveform.duration_ms() >= 29000))

        studio.search.setText("churn")  # search
        self.assertEqual(studio.match_label.text(), "1 / 2")
        row = studio.view.current_match_row
        self.assertEqual(studio.view.segment_model.index(row).data(Qt.DisplayRole), SEGMENTS[2]["text"])

        studio.view.setCurrentIndex(studio.view.segment_model.index(row))  # seek by keyboard, from the text
        QTest.keyClick(studio.view, Qt.Key_Return)
        self.assertTrue(pump(self._app, lambda: abs(studio.transport.position_ms() - 14000) < 400))
        self.assertEqual(studio.view.active_row, row)
        self.assertEqual(studio.view.segment_model.index(studio.view.active_row).data(NameRole), "S3")

        studio.transport.seek_ms(25000)  # seek from the transport/waveform; the active speaker follows
        self.assertTrue(pump(self._app, lambda: studio.view.active_row == 3))
        self.assertEqual(studio.view.segment_model.index(3).data(NameRole), "Raj")  # identified speaker

        studio.inspector.tabs.set_current("summary")  # inspect AI output
        self.assertIn("Revenue up 12%.", studio.inspector.summary.browser.toPlainText())

        target = Path(self._tmp.name) / "review.srt"  # export
        with patch.object(controller_module.QFileDialog, "getSaveFileName", return_value=(str(target), "")):
            studio.export_requested.emit("srt")
        self.assertIn("Raj: Churn is flat", target.read_text(encoding="utf-8"))

    def test_the_inspector_collapses_to_widen_the_transcript(self) -> None:
        win = self._window()
        project = self._project(win)
        win._open_project(project.id)
        QApplication.processEvents()
        before = win.studio.body.width()
        win.studio.inspector.tabs.set_collapsed(True)
        QApplication.processEvents()
        self.assertGreater(win.studio.body.width(), before + 200)
        self.assertEqual(win.studio.inspector.tabs.current(), "summary")  # only one inspector view at a time
        self.assertEqual(sum(1 for b in win.studio.inspector.tabs._buttons.values() if b.isChecked()), 1)


class WhileProcessingTests(AcceptanceCase):
    def test_stage_elapsed_time_and_cancel_are_visible_straight_away(self) -> None:
        from types import SimpleNamespace

        from ui_qt.job_stages import Stage

        win = self._window()
        win.nav_rail.set_current("transcribe")
        win.worker_thread = SimpleNamespace(isRunning=lambda: True, quit=lambda: None, wait=lambda *_: True)
        self.addCleanup(setattr, win, "worker_thread", None)
        win.job_tracker.reset({Stage.TRANSCRIBE})
        win.job_tracker.start(Stage.TRANSCRIBE)
        win._sync_timeline()  # the same call every progress event makes; nothing else is needed for it to appear
        bar = win.job_status
        self.assertTrue(bar.isVisibleTo(win))
        self.assertTrue(bar.stage_label.text())
        self.assertTrue(bar.cancel_button.isVisibleTo(win))
        bar._started_at -= 65
        bar._tick()
        self.assertEqual(bar.elapsed_label.text(), "01:05")
        for other_page in ("projects", "settings"):  # and it stays visible wherever the user goes
            win.nav_rail.set_current(other_page)
            self.assertTrue(bar.isVisibleTo(win), other_page)

    def test_telemetry_and_logs_do_not_take_screen_space_until_asked_for(self) -> None:
        win = self._window()
        win.nav_rail.set_current("transcribe")
        for hidden in (win.hw_panel, win.job_timeline, win.terminal_log, win.force_stop_btn):
            self.assertFalse(hidden.isVisibleTo(win))
        win._set_drawer(win.performance_drawer, True)
        QApplication.processEvents()
        self.assertTrue(win.hw_panel.isVisibleTo(win))
        self.assertTrue(win.transcript_text == "" and win.text_area.isVisibleTo(win))  # the core workflow stays put


class LayoutTests(AcceptanceCase):
    def _inside(self, win: MainWindow, widget: QWidget) -> bool:
        origin = widget.mapTo(win, QPoint(0, 0))
        rect = QRect(origin, widget.size())
        return rect.left() >= 0 and rect.right() <= win.width()

    def _pages(self, win: MainWindow) -> dict[str, list[QWidget]]:
        studio = win.studio
        return {
            "projects": [win.projects_home.drop_card, win.projects_home.live_card, win.projects_home.import_card, win.projects_home.footer],
            "transcribe": [win.transcribe_btn, win.model_combo, win.status_label, win.text_area, win.open_btn],
            "live": [win.live_source_combo, win.live_device_combo, win.live_title_input, win.live_state_chip],
            "templates": [win.templates_view.new_button, win.templates_view.list, win.templates_view.use_button],
            "settings": [win.theme_segment, win.transcript_size_spin, win.projects_dir_input, win.reduced_motion_combo],
            "studio": [studio.search, studio.export_button, studio.view, studio.transport.play_button, studio.inspector.tabs],
        }

    def _check(self, size: tuple[int, int], *, setup_may_scroll: bool = False) -> None:
        win = self._window(size=size)
        self._project(win)
        for key, widgets in self._pages(win).items():
            if key == "studio":
                win._open_project(win.project_store.list_summaries()[0].id)
            else:
                win.nav_rail.set_current(key)
            win.resize(*size)
            for _ in range(5):
                QApplication.processEvents()
            for widget in widgets:
                self.assertTrue(widget.isVisibleTo(win), f"{key}: {type(widget).__name__} hidden at {size}")
                self.assertTrue(self._inside(win, widget), f"{key}: {type(widget).__name__} clipped horizontally at {size}")
        win.nav_rail.set_current("transcribe")
        QApplication.processEvents()
        if not setup_may_scroll:
            self.assertFalse(win.transcription_scroll.horizontalScrollBar().isVisible(), f"setup column scrolls sideways at {size}")

    def test_every_page_fits_at_1366x768(self) -> None:
        self._check((1366, 768))

    def test_every_page_fits_at_1100x700(self) -> None:
        self._check((1100, 700))

    def test_every_page_fits_at_the_minimum_window_size(self) -> None:
        # The floor is narrower than rail + setup + transcript need, so the setup column may scroll sideways
        # (never clipping a control); collapsing the left rail gives it room back.
        self._check((840, 560), setup_may_scroll=True)

    def test_large_transcript_text_still_lays_out(self) -> None:
        win = self._window()
        win._save_config(transcript_font_px=28)
        win._apply_theme()
        project = self._project(win)
        win._open_project(project.id)
        QApplication.processEvents()
        view = win.studio.view
        small = view.visualRect(view.segment_model.index(0)).height()
        tokens.set_transcript_px(16)
        view.refresh_style()
        QApplication.processEvents()
        self.assertGreater(small, view.visualRect(view.segment_model.index(0)).height())  # 28 px rows are taller than 16 px rows


class KeyboardTests(AcceptanceCase):
    def _reachable(self, start: QWidget) -> list[QWidget]:
        found, widget = [start], start.nextInFocusChain()
        while widget is not start:
            if widget.isVisibleTo(widget.window()) and widget.isEnabled() and (widget.focusPolicy() & Qt.TabFocus):
                found.append(widget)
            widget = widget.nextInFocusChain()
        return found

    def test_there_are_no_empty_tab_stops(self) -> None:
        win = self._window()
        project = self._project(win)
        win._open_project(project.id)
        from PySide6.QtWidgets import QScrollArea

        stops = self._reachable(win.nav_rail.button("projects"))
        self.assertEqual([w for w in stops if isinstance(w, QScrollArea)], [])

    def test_every_control_on_home_is_reachable_with_the_keyboard(self) -> None:
        win = self._window()
        project = self._project(win)
        reachable = self._reachable(win.nav_rail.button("projects"))
        for expected in [*win.nav_rail.buttons(), win.projects_home.drop_card, win.projects_home.live_card,
                         win.projects_home.import_card, win.projects_home._cards[project.id], win.projects_home._cards[project.id].menu_button]:
            self.assertIn(expected, reachable, type(expected).__name__)

    def test_every_studio_control_is_reachable_with_the_keyboard(self) -> None:
        win = self._window()
        media = Path(self._tmp.name) / "r.wav"
        write_wav(media, 5)
        project = self._project(win, str(media))
        win._open_project(project.id)
        QApplication.processEvents()
        studio = win.studio
        reachable = self._reachable(studio.back_button)
        for expected in (studio.search, studio.export_button, studio.more_button, studio.view, studio.inspector.tabs.collapse_button,
                         studio.transport.play_button, studio.transport.back_button, studio.transport.forward_button,
                         studio.transport.waveform, studio.transport.speed_combo, *studio.inspector.tabs._buttons.values()):
            self.assertIn(expected, reachable, type(expected).__name__)

    def test_focus_and_selection_are_visibly_styled(self) -> None:
        theme.apply_theme(self._app, "dark")
        qss = self._app.styleSheet()
        for selector in ("QPushButton[variant]:focus", "QPushButton[nav]:focus", "QPushButton[tab]:focus", 'QFrame[studioCard="true"][interactive="true"]:focus',
                         "QLineEdit:focus", "QComboBox:focus", "QCheckBox:focus", "QPushButton[nav]:checked", "QPushButton[tab]:checked",
                         '[selected="true"]', "QListWidget::item:selected", "QPushButton[segment]:checked"):
            self.assertIn(selector, qss)


class MotionTests(AcceptanceCase):
    def test_reduced_motion_removes_animation_everywhere(self) -> None:
        win = self._window(AppConfig(reduced_motion="on"))
        self.assertTrue(tokens.reduced_motion())
        win._set_drawer(win.details_drawer, True)
        self.assertEqual(win.details_drawer.maximumWidth(), win.details_drawer._width)  # opened in one step
        self.assertEqual(win.details_drawer._animation.state(), win.details_drawer._animation.State.Stopped)
        win.job_status.show_job("Starting…", None)
        self.assertEqual((win.job_status.progress.minimum(), win.job_status.progress.maximum()), (0, 100))  # no busy bar

    def test_motion_is_used_when_allowed(self) -> None:
        win = self._window(AppConfig(reduced_motion="off"))
        self.assertFalse(tokens.reduced_motion())
        win.job_status.show_job("Starting…", None)
        self.assertEqual(win.job_status.progress.maximum(), 0)  # busy indicator
        win._set_drawer(win.details_drawer, True)
        self.assertGreater(win.details_drawer._animation.duration(), 0)
        win.details_drawer._animation.stop()


class ContrastTests(unittest.TestCase):
    def test_active_line_highlight_keeps_text_readable_in_every_theme(self) -> None:
        for preset in PRESETS:
            for mode in ("light", "dark"):
                palette = resolve(preset.id, None, mode).palette
                active_bg = mix(palette.page, palette.accent, 46 / 255)  # the translucent fill the transcript paints
                self.assertGreaterEqual(contrast(palette.ink, active_bg), 4.5, f"{preset.id} {mode}")
                self.assertGreaterEqual(contrast(palette.accent_text, palette.accent), 4.5, f"{preset.id} {mode} search match")

    def test_speaker_names_keep_contrast_on_the_highlight(self) -> None:
        for preset in PRESETS:
            for mode in ("light", "dark"):
                resolved = resolve(preset.id, None, mode)
                active_bg = mix(resolved.palette.page, resolved.palette.accent, 46 / 255)
                for color in resolved.speaker_colors:
                    self.assertGreaterEqual(contrast(color, active_bg), 3.0, f"{preset.id} {mode} {color}")  # bold 14 px name


class LargeTranscriptTests(AcceptanceCase):
    def test_a_three_hour_project_opens_searches_and_follows_playback_quickly(self) -> None:
        win = self._window()
        text = "We reviewed the roadmap, the hiring plan and the budget for next quarter in detail. "
        segments = sanitize_segments(
            [{"start": i * 3.6, "end": i * 3.6 + 3.5, "speaker": f"S{i % 4 + 1}", "text": text * 2} for i in range(3000)]
        )
        project = self._project(win, segments=segments, duration_s=10800.0)
        started = time.monotonic()
        win._open_project(project.id)
        QApplication.processEvents()
        self.assertLess(time.monotonic() - started, 1.5, "open")
        view = win.studio.view
        started = time.monotonic()
        win.studio.search.setText("budget")
        self.assertEqual(view.match_count(), 3000)
        self.assertLess(time.monotonic() - started, 1.0, "search")
        started = time.monotonic()
        for second in range(0, 2000, 4):
            view.set_active_time(float(second), follow=False)
        self.assertLess(time.monotonic() - started, 0.5, "follow playback")
        view.scrollToBottom()
        QApplication.processEvents()
        self.assertFalse(view.grab().isNull())


class ExportConsistencyTests(unittest.TestCase):
    def test_srt_and_text_use_the_same_speaker_names(self) -> None:
        from services.project_store import Project

        project = Project(id="20260101-000000-abcdef12", title="t", segments=SEGMENTS, speaker_names={"S1": "Sarah"})
        self.assertIn("Sarah:", export_service.export_srt(project))
        self.assertIn("Sarah:", export_service.export_txt(project))


if __name__ == "__main__":
    unittest.main()
