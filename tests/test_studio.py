"""Transcript Studio: model, transcript list, waveform, transport, inspector, view and controller."""

from __future__ import annotations

import math
import os
import struct
import tempfile
import time
import unittest
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

from PySide6.QtCore import QModelIndex, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from services.config_service import AppConfig
from services.project_store import Project, ProjectStore, sanitize_segments
from ui_qt.views.studio import controller as controller_module
from ui_qt.views.studio.controller import StudioController
from ui_qt.views.studio.inspector import OutputPage, SpeakersPage, SummaryPage, VisualsPage, speaker_stats
from ui_qt.views.studio.segment_model import NameRole, SegmentModel, StartRole
from ui_qt.views.studio.studio_view import StudioView, subtitle_for
from ui_qt.views.studio.transcript_view import SegmentEditor, TranscriptView, format_clock
from ui_qt.views.studio.transport_bar import TransportBar
from ui_qt.views.studio.waveform_widget import WaveformWidget

RAW = [
    {"start": 0.0, "end": 4.0, "speaker": "S1", "text": "Welcome everyone to the quarterly review."},
    {"start": 4.0, "end": 9.0, "speaker": "S2", "text": "Thanks. Revenue grew twelve percent."},
    {"start": 9.0, "end": 14.0, "speaker": "S1", "text": "Great. What about the review of churn?"},
    {"start": 14.0, "end": 16.0, "speaker": "S?", "text": "Hmm."},
    {"start": 16.0, "end": 30.0, "speaker": "S2", "text": "Churn is flat, reviewing enterprise next."},
]
SEGMENTS = sanitize_segments(RAW)


def write_wav(path: Path, seconds: float = 6.0) -> None:
    rate = 8000
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(i / 9.0))) for i in range(int(rate * seconds))))


def pump(app: QApplication, condition=lambda: False, timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return condition()


class QtCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls._app = QApplication.instance() or QApplication([])


class SegmentModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = SegmentModel()
        self.model.set_segments(list(reversed(SEGMENTS)), {"S1": "Sarah"})

    def test_sorted_by_start_and_names(self) -> None:
        self.assertEqual([self.model.index(r).data(StartRole) for r in range(5)], [0.0, 4.0, 9.0, 14.0, 16.0])
        self.assertEqual(self.model.index(0).data(NameRole), "Sarah")
        self.assertEqual(self.model.index(1).data(NameRole), "S2")
        self.assertEqual(self.model.index(3).data(NameRole), "")  # unknown speaker has no name

    def test_row_for_time(self) -> None:
        self.assertEqual(self.model.row_for_time(-1), -1 + 0)  # before the first segment
        self.assertEqual(self.model.row_for_time(0.0), 0)
        self.assertEqual(self.model.row_for_time(3.99), 0)
        self.assertEqual(self.model.row_for_time(4.0), 1)
        self.assertEqual(self.model.row_for_time(500), 4)
        self.assertEqual(SegmentModel().row_for_time(3), -1)

    def test_set_data_normalises_and_signals(self) -> None:
        edits: list[tuple[int, str]] = []
        self.model.text_edited.connect(lambda r, t: edits.append((r, t)))
        index = self.model.index(1)
        self.assertTrue(self.model.setData(index, "  Revenue   grew\n13 percent. "))
        self.assertEqual(edits, [(1, "Revenue grew 13 percent.")])
        self.assertFalse(self.model.setData(index, "Revenue grew 13 percent."))  # unchanged
        self.assertFalse(self.model.setData(index, "   "))  # never blank a segment
        self.assertFalse(self.model.setData(QModelIndex(), "x"))
        self.assertEqual(len(edits), 1)

    def test_speaker_names_update(self) -> None:
        self.model.set_speaker_names({"S2": "Raj"})
        self.assertEqual(self.model.index(1).data(NameRole), "Raj")
        self.assertEqual(self.model.index(0).data(NameRole), "S1")


class TranscriptViewTests(QtCase):
    def _view(self) -> TranscriptView:
        view = TranscriptView()
        view.resize(800, 400)
        view.show()
        view.load(SEGMENTS, {"S1": "Sarah"})
        QApplication.processEvents()
        self.addCleanup(view.close)
        return view

    def test_clock_format(self) -> None:
        self.assertEqual(format_clock(52), "00:52")
        self.assertEqual(format_clock(3725), "1:02:05")

    def test_active_row_follows_time(self) -> None:
        view = self._view()
        view.set_active_time(10.0)
        self.assertEqual(view.active_row, 2)
        view.set_active_time(10.5)
        self.assertEqual(view.active_row, 2)
        view.set_active_time(20.0, follow=False)
        self.assertEqual(view.active_row, 4)

    def test_click_and_enter_seek_to_segment_start(self) -> None:
        view = self._view()
        seen: list[float] = []
        view.seek_requested.connect(seen.append)
        index = view.segment_model.index(1)
        QTest.mouseClick(view.viewport(), Qt.LeftButton, pos=view.visualRect(index).center())
        self.assertEqual(seen, [4.0])
        view.setCurrentIndex(view.segment_model.index(2))
        QTest.keyClick(view, Qt.Key_Return)
        self.assertEqual(seen, [4.0, 9.0])

    def test_space_toggles_playback(self) -> None:
        view = self._view()
        hits: list[int] = []
        view.toggle_play_requested.connect(lambda: hits.append(1))
        view.setFocus()
        QTest.keyClick(view, Qt.Key_Space)
        self.assertEqual(hits, [1])

    def test_search_counts_wraps_and_is_case_insensitive(self) -> None:
        view = self._view()
        seen: list[tuple[int, int]] = []
        view.matches_changed.connect(lambda c, t: seen.append((c, t)))
        view.set_query("REVIEW")  # rows 0, 2 and 4 ("reviewing")
        self.assertEqual(view.match_count(), 3)
        self.assertEqual(seen[-1], (1, 3))
        self.assertEqual(view.current_match_row, 0)
        view.next_match()
        view.next_match()
        self.assertEqual(view.current_match_row, 4)
        view.next_match()
        self.assertEqual(view.current_match_row, 0)  # wraps
        view.previous_match()
        self.assertEqual(view.current_match_row, 4)
        view.set_query("zzz")
        self.assertEqual((view.match_count(), view.current_match_row, seen[-1]), (0, -1, (0, 0)))
        view.set_query("")
        self.assertEqual(seen[-1], (0, 0))

    def test_inline_edit_commits_on_enter_and_cancels_on_escape(self) -> None:
        view = self._view()
        edits: list[tuple[int, str]] = []
        view.segment_model.text_edited.connect(lambda r, t: edits.append((r, t)))
        index = view.segment_model.index(3)
        view.edit(index)
        editor = view.findChild(SegmentEditor)
        self.assertIsNotNone(editor)
        editor.setPlainText("Hm, good point.")
        QTest.keyClick(editor, Qt.Key_Return)
        QApplication.processEvents()
        self.assertEqual(edits, [(3, "Hm, good point.")])
        view.edit(index)
        editor = view.findChild(SegmentEditor)
        editor.setPlainText("never saved")
        QTest.keyClick(editor, Qt.Key_Escape)
        QApplication.processEvents()
        self.assertEqual(view.segment_model.index(3).data(Qt.DisplayRole), "Hm, good point.")
        self.assertEqual(len(edits), 1)

    def test_renders_without_error_when_painting(self) -> None:
        view = self._view()
        view.set_query("churn")
        view.set_active_time(10)
        pixmap = view.grab()
        self.assertFalse(pixmap.isNull())

    def test_three_hour_transcript_loads_quickly(self) -> None:
        segments = [{"start": i * 3.5, "end": i * 3.5 + 3, "speaker": f"S{i % 3 + 1}", "text": "A reasonably long sentence about the quarterly plan and its risks. " * 2} for i in range(3000)]
        view = TranscriptView()
        view.resize(800, 500)
        view.show()
        started = time.monotonic()
        view.load(sanitize_segments(segments), {})
        QApplication.processEvents()
        view.scrollToBottom()
        QApplication.processEvents()
        self.assertLess(time.monotonic() - started, 1.5)
        view.close()


class WaveformWidgetTests(QtCase):
    def test_click_seeks_proportionally_and_keys_step(self) -> None:
        widget = WaveformWidget()
        widget.resize(400, 44)
        widget.show()
        seen: list[int] = []
        widget.seek_requested.connect(seen.append)
        QTest.mouseClick(widget, Qt.LeftButton, pos=widget.rect().center())
        self.assertEqual(seen, [])  # no duration yet
        widget.set_duration_ms(60_000)
        widget.set_position_ms(10_000)
        QTest.mouseClick(widget, Qt.LeftButton, pos=widget.rect().center())
        self.assertAlmostEqual(seen[-1], 30_000, delta=400)
        QTest.keyClick(widget, Qt.Key_Right)
        QTest.keyClick(widget, Qt.Key_Left)
        self.assertEqual(seen[-2:], [15_000, 5_000])
        widget.set_position_ms(58_000)
        QTest.keyClick(widget, Qt.Key_Right)
        self.assertEqual(seen[-1], 60_000)  # clamped
        self.assertFalse(widget.grab().isNull())

    def test_fraction(self) -> None:
        widget = WaveformWidget()
        widget.set_duration_ms(1000)
        widget.set_position_ms(250)
        self.assertEqual(widget.fraction(), 0.25)
        widget.set_position_ms(5000)
        self.assertEqual(widget.fraction(), 1.0)


class TransportBarTests(QtCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.wav = Path(self._tmp.name) / "tone.wav"
        write_wav(self.wav)

    def _bar(self) -> TransportBar:
        bar = TransportBar()
        bar.show()
        self.addCleanup(bar.shutdown)
        return bar

    def test_without_media_it_is_disabled_with_a_reason(self) -> None:
        bar = self._bar()
        self.assertFalse(bar.is_available())
        self.assertFalse(bar.play_button.isEnabled())
        self.assertTrue(bar.notice.isVisibleTo(bar))
        self.assertFalse(bar.load("/does/not/exist.wav"))
        self.assertIn("no longer available", bar.notice.text())
        bar.toggle()  # must not raise
        bar.seek_ms(1000)
        self.assertEqual(bar.position_ms(), 0)

    def test_loads_seeks_and_reports_time(self) -> None:
        bar = self._bar()
        positions: list[int] = []
        bar.position_changed.connect(positions.append)
        self.assertTrue(bar.load(str(self.wav), 6000))
        self.assertTrue(bar.is_available())
        self.assertFalse(bar.notice.isVisibleTo(bar))
        self.assertTrue(pump(self._app, lambda: bar.waveform.duration_ms() >= 5900))
        bar.seek_ms(2500)
        self.assertTrue(pump(self._app, lambda: abs(bar.position_ms() - 2500) < 200))
        self.assertIn("00:02 / 00:06", bar.time_label.text())
        self.assertIn(2500, positions)
        bar.seek_ms(999_999)  # clamped to the end
        self.assertLessEqual(bar.waveform.position_ms(), 6100)

    def test_a_seek_made_while_the_file_is_still_opening_is_applied(self) -> None:
        bar = self._bar()
        bar.load(str(self.wav), 6000)
        bar.seek_ms(3000)  # immediately, before the player has loaded the file
        self.assertTrue(pump(self._app, lambda: abs(bar.position_ms() - 3000) < 250))

    def test_skip_buttons_and_speed(self) -> None:
        bar = self._bar()
        bar.load(str(self.wav), 6000)
        pump(self._app, lambda: bar.waveform.duration_ms() >= 5900)
        bar.seek_ms(1000)
        bar.forward_button.click()
        pump(self._app, lambda: bar.position_ms() >= 5000)
        self.assertGreaterEqual(bar.position_ms(), 5000)
        bar.back_button.click()
        pump(self._app, lambda: bar.position_ms() <= 1000)
        self.assertLessEqual(bar.position_ms(), 1000)
        bar.speed_combo.setCurrentIndex(3)
        self.assertEqual(bar._player.playbackRate(), 1.5)

    def test_unload_clears_state(self) -> None:
        bar = self._bar()
        bar.load(str(self.wav), 6000)
        bar.unload("gone")
        self.assertFalse(bar.is_available())
        self.assertEqual(bar.notice.text(), "gone")


class InspectorTests(QtCase):
    def test_speaker_stats_counts_turns_and_time(self) -> None:
        stats = speaker_stats(SEGMENTS)
        self.assertEqual(stats["S1"], (2, 9.0))
        self.assertEqual(stats["S2"], (2, 19.0))
        self.assertNotIn("S?", stats)
        merged = sanitize_segments([{"start": 0, "end": 2, "speaker": "S1", "text": "a"}, {"start": 2, "end": 5, "speaker": "S1", "text": "b"}])
        self.assertEqual(speaker_stats(merged)["S1"], (1, 5.0))

    def test_speakers_page_emits_only_real_renames(self) -> None:
        page = SpeakersPage()
        page.set_project(Project(id="20260101-000000-abcdef12", title="t", segments=SEGMENTS, speaker_names={"S1": "Sarah"}))
        self.assertEqual(sorted(page.edits), ["S1", "S2"])
        self.assertEqual(page.edits["S1"].text(), "Sarah")
        self.assertFalse(page.empty.isVisibleTo(page) if page.isVisible() else not page.empty.isHidden())
        seen: list[tuple[str, str]] = []
        page.renamed.connect(lambda label, name: seen.append((label, name)))
        page.edits["S1"].editingFinished.emit()  # unchanged
        page.edits["S2"].setText("  Raj ")
        page.edits["S2"].editingFinished.emit()
        page.edits["S2"].editingFinished.emit()  # repeated focus-out must not re-save
        page.edits["S1"].setText("")
        page.edits["S1"].editingFinished.emit()  # clearing a name is a rename to ""
        self.assertEqual(seen, [("S2", "Raj"), ("S1", "")])

    def test_speakers_page_empty_state(self) -> None:
        page = SpeakersPage()
        page.set_project(Project(id="20260101-000000-abcdef12", title="t", segments=sanitize_segments([{"text": "x", "start": 0, "end": 1}])))
        self.assertEqual(page.edits, {})
        self.assertFalse(page.empty.isHidden())

    def test_output_pages_show_placeholder_then_markdown(self) -> None:
        page = OutputPage("action-items", "Generate action items…", "Nothing yet.")
        self.assertEqual(page.browser.toPlainText(), "Nothing yet.")
        self.assertEqual(page.generate_button.text(), "Generate action items…")
        requested: list[str] = []
        page.generate_requested.connect(requested.append)
        page.generate_button.click()
        self.assertEqual(requested, ["action-items"])
        page.set_outputs({"action-items": {"text": "- [ ] Send the deck", "created": "", "profile": ""}})
        self.assertIn("Send the deck", page.browser.toPlainText())
        self.assertEqual(page.generate_button.text(), "Regenerate action items…")

    def test_summary_page_lists_other_outputs(self) -> None:
        page = SummaryPage("meeting-summary", "Generate summary…", "None.")
        outputs = {
            "meeting-summary": {"text": "# Summary", "created": "", "profile": ""},
            "action-items": {"text": "- a", "created": "", "profile": ""},  # belongs to its own tab
            "decision-log": {"text": "Decided X", "created": "", "profile": ""},
        }
        page.set_outputs(outputs)
        self.assertEqual([page.chooser.itemData(i) for i in range(page.chooser.count())], ["meeting-summary", "decision-log"])
        self.assertTrue(not page.chooser.isHidden())
        page.chooser.setCurrentIndex(1)
        self.assertIn("Decided X", page.browser.toPlainText())

    def test_visuals_page_placeholder(self) -> None:
        page = VisualsPage()
        page.set_report("")
        self.assertIn("No visual analysis", page.text.toPlainText())
        page.set_report("Slide 1: Q3 revenue")
        self.assertEqual(page.text.toPlainText(), "Slide 1: Q3 revenue")


def make_project(store: ProjectStore, media: str = "", **changes: object) -> Project:
    project = store.create(title="Quarterly Review", media_path=media)
    values: dict[str, object] = dict(status="completed", segments=SEGMENTS, duration_s=30.0, transcript_text="x")
    values.update(changes)
    return store.update(project.id, **values)  # type: ignore[return-value]


class StudioViewTests(QtCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = ProjectStore(Path(self._tmp.name) / "projects")

    def _studio(self) -> StudioView:
        studio = StudioView()
        studio.resize(1200, 700)
        studio.show()
        self.addCleanup(studio.shutdown)
        self.addCleanup(studio.close)
        return studio

    def test_subtitle(self) -> None:
        project = make_project(self.store)
        text = subtitle_for(project)
        self.assertIn("00:30", text)
        self.assertIn("2 speakers", text)

    def test_load_project_fills_every_part(self) -> None:
        studio = self._studio()
        studio.load_project(make_project(self.store, speaker_names={"S1": "Sarah"}), [0.2, 0.8, 0.5])
        self.assertEqual(studio.title_label.text(), "Quarterly Review")
        self.assertEqual(studio.view.segment_model.rowCount(), 5)
        self.assertEqual(studio.body.currentIndex(), 0)
        self.assertTrue(studio.export_button.isEnabled())
        self.assertTrue(studio.transport.waveform.has_peaks())
        self.assertEqual(studio.inspector.speakers.edits["S1"].text(), "Sarah")

    def test_project_without_transcript_points_to_visuals(self) -> None:
        studio = self._studio()
        project = make_project(self.store, segments=[], transcript_text="", visual_report="Slide 1: revenue")
        studio.load_project(project)
        self.assertEqual(studio.body.currentIndex(), 1)
        self.assertFalse(studio.export_button.isEnabled())
        self.assertEqual(studio.inspector.tabs.current(), "visuals")

    def test_search_label_and_shortcut(self) -> None:
        studio = self._studio()
        studio.load_project(make_project(self.store))
        self.assertEqual(studio.match_label.text(), "")
        self.assertFalse(studio.next_button.isEnabled())
        studio.search.setText("review")
        self.assertEqual(studio.match_label.text(), "1 / 3")
        self.assertTrue(studio.next_button.isEnabled())
        studio.next_button.click()
        self.assertEqual(studio.match_label.text(), "2 / 3")
        studio.search.setText("nothing like this")
        self.assertEqual(studio.match_label.text(), "No matches")
        studio.view.setFocus()
        studio.focus_search()
        QApplication.processEvents()
        self.assertTrue(studio.search.hasFocus())
        studio._clear_search()
        self.assertEqual(studio.search.text(), "")

    def test_click_on_text_seeks_playback_and_highlights(self) -> None:
        studio = self._studio()
        media = Path(self._tmp.name) / "tone.wav"
        write_wav(media, 30.0)
        studio.load_project(make_project(self.store, media=str(media)))
        pump(self._app, lambda: studio.transport.waveform.duration_ms() >= 29000)
        index = studio.view.segment_model.index(2)
        QTest.mouseClick(studio.view.viewport(), Qt.LeftButton, pos=studio.view.visualRect(index).center())
        self.assertTrue(pump(self._app, lambda: abs(studio.transport.position_ms() - 9000) < 300))
        self.assertEqual(studio.view.active_row, 2)
        studio.transport.waveform.seek_requested.emit(20_000)  # seeking from the waveform moves the text too
        self.assertTrue(pump(self._app, lambda: studio.view.active_row == 4))

    def test_missing_media_still_lets_text_seek_highlight(self) -> None:
        studio = self._studio()
        studio.load_project(make_project(self.store, media="/gone.wav"))
        self.assertFalse(studio.transport.is_available())
        studio._on_seek_from_text(14.0)
        self.assertEqual(studio.view.active_row, 3)

    def test_export_menu_and_inspector_collapse(self) -> None:
        studio = self._studio()
        studio.load_project(make_project(self.store))
        keys: list[str] = []
        studio.export_requested.connect(keys.append)
        for action in studio.export_menu.actions():
            action.trigger()
        self.assertEqual(keys, ["txt", "md", "srt", "vtt", "json"])
        before = studio.splitter.sizes()[1]
        studio.inspector.tabs.set_collapsed(True)
        QApplication.processEvents()
        self.assertLess(studio.splitter.sizes()[1], before)
        studio.inspector.tabs.set_collapsed(False)

    def test_generate_and_rename_signals_pass_through(self) -> None:
        studio = self._studio()
        studio.load_project(make_project(self.store))
        gen: list[str] = []
        names: list[tuple[str, str]] = []
        studio.generate_requested.connect(gen.append)
        studio.speaker_renamed.connect(lambda a, b: names.append((a, b)))
        studio.inspector.summary.generate_button.click()
        studio.inspector.actions.generate_button.click()
        studio.inspector.speakers.edits["S2"].setText("Raj")
        studio.inspector.speakers.edits["S2"].editingFinished.emit()
        self.assertEqual(gen, ["meeting-summary", "action-items"])
        self.assertEqual(names, [("S2", "Raj")])


class FakeDialog:
    created: list["FakeDialog"] = []
    output = "## Summary\n- decided things"
    template = "meeting-summary"

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        FakeDialog.created.append(self)

    def exec(self) -> int:
        return 1

    def output_text(self) -> str:
        return FakeDialog.output

    def output_template_id(self) -> str:
        return FakeDialog.template

    def output_profile_name(self) -> str:
        return "local-llm"


class StudioControllerTests(QtCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = ProjectStore(Path(self._tmp.name) / "projects")
        self.view = StudioView()
        self.view.resize(1000, 600)
        self.view.show()
        self.window = QApplication.activeWindow() or self.view
        self.saved_dirs: list[str] = []
        self.profiles_ok = True
        self.controller = StudioController(
            self.view, self.store, self.view,
            get_config=lambda: AppConfig(), ensure_llm_profiles=lambda: self.profiles_ok, is_busy=lambda: False,
            get_save_dir=lambda: self._tmp.name, set_save_dir=self.saved_dirs.append, save_config=lambda: None,
        )
        self.changed: list[int] = []
        self.statuses: list[str] = []
        self.controller.project_changed.connect(lambda: self.changed.append(1))
        self.controller.status.connect(self.statuses.append)
        FakeDialog.created.clear()
        self.addCleanup(self.view.shutdown)
        self.addCleanup(self.view.close)

    def test_open_missing_project_is_refused(self) -> None:
        self.assertFalse(self.controller.open_project("20260101-000000-abcdef12"))

    def test_reidentify_action_needs_saved_audio(self) -> None:
        self.controller.open_project(make_project(self.store, media="/nonexistent/a.wav").id)
        self.assertFalse(self.view.rediarize_action.isEnabled())
        media = Path(self._tmp.name) / "tone.wav"
        write_wav(media)
        self.controller.open_project(make_project(self.store, media=str(media)).id)
        self.assertTrue(self.view.rediarize_action.isEnabled())

    def test_reidentify_relabels_in_place_and_resets_names(self) -> None:
        media = Path(self._tmp.name) / "tone.wav"
        write_wav(media)
        project = make_project(self.store, media=str(media), speaker_names={"S1": "Sarah"})
        self.controller.open_project(project.id)
        relabelled = [dict(seg, speaker="S1") for seg in SEGMENTS]
        with (
            patch("ui_qt.views.studio.controller.QInputDialog.getInt", return_value=(1, True)) as ask,
            patch("ui_qt.views.studio.controller.QMessageBox.question", return_value=QMessageBox.Yes),
            patch("services.transcription_service.rediarize_segments", return_value=relabelled) as run,
        ):
            self.controller.rediarize()
            self.assertTrue(pump(self._app, lambda: not self.controller.is_rediarizing))
        self.assertEqual(ask.call_args.args[3], 2, "default is the current speaker count")
        self.assertEqual(run.call_args.kwargs["num_speakers"], 1)
        saved = self.store.load(project.id)
        assert saved is not None
        self.assertEqual(saved.speaker_count, 1)
        self.assertEqual(saved.speaker_names, {})
        self.assertIn("Found 1 speaker.", self.statuses)
        self.assertTrue(self.changed)

    def test_reidentify_failure_leaves_project_unchanged(self) -> None:
        media = Path(self._tmp.name) / "tone.wav"
        write_wav(media)
        project = make_project(self.store, media=str(media))
        self.controller.open_project(project.id)
        with (
            patch("ui_qt.views.studio.controller.QInputDialog.getInt", return_value=(3, True)),
            patch("ui_qt.views.studio.controller.QMessageBox.warning") as warn,
            patch("services.transcription_service.rediarize_segments", side_effect=RuntimeError("no speakers")),
        ):
            self.controller.rediarize()
            self.assertTrue(pump(self._app, lambda: warn.called))
        self.assertFalse(self.controller.is_rediarizing)
        self.assertTrue(self.view.rediarize_action.isEnabled())
        self.assertEqual(self.store.load(project.id).segments, SEGMENTS)  # type: ignore[union-attr]

    def test_open_computes_and_caches_the_waveform_once(self) -> None:
        media = Path(self._tmp.name) / "tone.wav"
        write_wav(media)
        project = make_project(self.store, media=str(media))
        self.assertTrue(self.controller.open_project(project.id))
        self.assertTrue(pump(self._app, lambda: self.view.transport.waveform.has_peaks()))
        self.assertGreater(len(self.store.load_peaks(project.id)), 100)
        self.assertEqual(len(self.store.list_summaries()[0].preview_peaks), 64)
        self.assertTrue(self.changed)
        with patch.object(controller_module, "compute_peaks", side_effect=AssertionError("must use the cache")):
            self.assertTrue(self.controller.open_project(project.id))
        self.assertTrue(self.view.transport.waveform.has_peaks())

    def test_switching_project_drops_a_late_waveform(self) -> None:
        media = Path(self._tmp.name) / "tone.wav"
        write_wav(media)
        first = make_project(self.store, media=str(media))
        second = make_project(self.store)
        self.controller.open_project(first.id)
        self.controller.open_project(second.id)
        pump(self._app, timeout=0.5)
        self.assertFalse(self.view.transport.waveform.has_peaks())

    def test_edits_are_saved_after_a_pause_and_on_flush(self) -> None:
        project = make_project(self.store)
        self.controller.open_project(project.id)
        self.view.view.segment_model.setData(self.view.view.segment_model.index(1), "Revenue grew thirteen percent.")
        self.assertEqual(self.store.load(project.id).segments[1]["text"], "Thanks. Revenue grew twelve percent.")  # not yet
        self.assertTrue(pump(self._app, lambda: self.store.load(project.id).segments[1]["text"].endswith("thirteen percent."), 3))
        self.assertIn("thirteen", self.store.load(project.id).transcript_text)
        self.view.view.segment_model.setData(self.view.view.segment_model.index(0), "Welcome all.")
        self.controller.flush()  # immediately, e.g. when leaving the Studio
        self.assertEqual(self.store.load(project.id).segments[0]["text"], "Welcome all.")

    def test_flush_without_changes_writes_nothing(self) -> None:
        project = make_project(self.store)
        self.controller.open_project(project.id)
        self.changed.clear()
        self.controller.flush()
        self.assertEqual(self.changed, [])

    def test_speaker_rename_updates_view_and_store(self) -> None:
        project = make_project(self.store)
        self.controller.open_project(project.id)
        self.view.speaker_renamed.emit("S1", "Sarah")
        self.assertEqual(self.store.load(project.id).speaker_names, {"S1": "Sarah"})
        self.assertEqual(self.view.view.segment_model.index(0).data(NameRole), "Sarah")
        self.assertIn("2 speakers", self.view.subtitle_label.text())

    def test_rename_project(self) -> None:
        project = make_project(self.store)
        self.controller.open_project(project.id)
        with patch.object(controller_module.QInputDialog, "getText", return_value=("Board Review", True)):
            self.controller.rename_project()
        self.assertEqual(self.store.load(project.id).title, "Board Review")
        self.assertEqual(self.view.title_label.text(), "Board Review")
        with patch.object(controller_module.QInputDialog, "getText", return_value=("   ", True)):
            self.controller.rename_project()
        self.assertEqual(self.store.load(project.id).title, "Board Review")

    def test_export_writes_file_with_extension_and_remembers_folder(self) -> None:
        project = make_project(self.store, speaker_names={"S1": "Sarah"})
        self.controller.open_project(project.id)
        target = Path(self._tmp.name) / "out"
        with patch.object(controller_module.QFileDialog, "getSaveFileName", return_value=(str(target), "")):
            self.controller.export("srt")
        written = Path(self._tmp.name) / "out.srt"
        self.assertIn("Sarah: Welcome everyone", written.read_text(encoding="utf-8"))
        self.assertEqual(self.saved_dirs, [self._tmp.name])
        self.assertEqual(self.statuses[-1], "Exported out.srt")
        with patch.object(controller_module.QFileDialog, "getSaveFileName", return_value=("", "")):
            self.controller.export("txt")  # cancelled dialog
        self.assertEqual(len(self.saved_dirs), 1)

    def test_export_failure_is_reported(self) -> None:
        project = make_project(self.store)
        self.controller.open_project(project.id)
        bad = Path(self._tmp.name) / "no_such_dir" / "x.txt"
        with patch.object(controller_module.QFileDialog, "getSaveFileName", return_value=(str(bad), "")), patch.object(
            controller_module.QMessageBox, "warning"
        ) as warn:
            self.controller.export("txt")
        warn.assert_called_once()
        self.assertEqual(self.saved_dirs, [])

    def test_export_flushes_pending_edits_first(self) -> None:
        project = make_project(self.store)
        self.controller.open_project(project.id)
        self.view.view.segment_model.setData(self.view.view.segment_model.index(0), "Edited opening line.")
        target = Path(self._tmp.name) / "edited.txt"
        with patch.object(controller_module.QFileDialog, "getSaveFileName", return_value=(str(target), "")):
            self.controller.export("txt")
        self.assertIn("Edited opening line.", target.read_text(encoding="utf-8"))

    def test_copy_transcript(self) -> None:
        project = make_project(self.store)
        self.controller.open_project(project.id)
        self.controller.copy_transcript()
        self.assertIn("[00:00] S1: Welcome everyone", QGuiApplication.clipboard().text())

    def test_reveal_opens_only_the_project_folder(self) -> None:
        project = make_project(self.store)
        self.controller.open_project(project.id)
        with patch.object(controller_module, "open_folder") as opener:
            self.controller.reveal()
        opener.assert_called_once_with(str(self.store.project_dir(project.id)))

    def test_generate_stores_output_with_the_project(self) -> None:
        project = make_project(self.store, speaker_names={"S1": "Sarah"})
        self.controller.open_project(project.id)
        with patch.object(controller_module, "LLMPostprocessDialog", FakeDialog):
            self.controller.generate("meeting-summary")
        dialog = FakeDialog.created[0]
        self.assertEqual(dialog.kwargs["initial_template_id"], "meeting-summary")
        self.assertIn("Sarah: Welcome everyone", str(dialog.kwargs["current_transcript_text"]))
        self.assertNotIn("[00:", str(dialog.kwargs["current_transcript_text"]))  # no timestamps in the prompt
        stored = self.store.load(project.id).ai_outputs["meeting-summary"]
        self.assertEqual((stored["text"], stored["profile"]), (FakeDialog.output, "local-llm"))
        self.assertIn("decided things", self.view.inspector.summary.browser.toPlainText())

    def test_generate_keeps_the_template_actually_used(self) -> None:
        project = make_project(self.store)
        self.controller.open_project(project.id)
        FakeDialog.template = "decision-log"
        self.addCleanup(setattr, FakeDialog, "template", "meeting-summary")
        with patch.object(controller_module, "LLMPostprocessDialog", FakeDialog):
            self.controller.generate("meeting-summary")
        self.assertEqual(list(self.store.load(project.id).ai_outputs), ["decision-log"])

    def test_generate_without_profiles_or_text_does_not_open_the_dialog(self) -> None:
        project = make_project(self.store)
        self.controller.open_project(project.id)
        self.profiles_ok = False
        with patch.object(controller_module, "LLMPostprocessDialog", FakeDialog):
            self.controller.generate("meeting-summary")
        self.assertEqual(FakeDialog.created, [])
        empty = make_project(self.store, segments=[], transcript_text="")
        self.controller.open_project(empty.id)
        self.profiles_ok = True
        with patch.object(controller_module, "LLMPostprocessDialog", FakeDialog), patch.object(
            controller_module.QMessageBox, "information"
        ) as info:
            self.controller.generate("meeting-summary")
        info.assert_called_once()
        self.assertEqual(FakeDialog.created, [])

    def test_empty_dialog_output_stores_nothing(self) -> None:
        project = make_project(self.store)
        self.controller.open_project(project.id)
        FakeDialog.output = ""
        self.addCleanup(setattr, FakeDialog, "output", "## Summary\n- decided things")
        with patch.object(controller_module, "LLMPostprocessDialog", FakeDialog):
            self.controller.generate("meeting-summary")
        self.assertEqual(self.store.load(project.id).ai_outputs, {})

    def test_forget_clears_the_open_project(self) -> None:
        project = make_project(self.store)
        self.controller.open_project(project.id)
        self.controller.forget("20260101-000000-abcdef12")
        self.assertEqual(self.view.project_id, project.id)
        self.controller.forget(project.id)
        self.assertIsNone(self.view.project_id)
        self.assertEqual(self.view.view.segment_model.rowCount(), 0)


if __name__ == "__main__":
    unittest.main()
