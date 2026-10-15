"""Projects Home, the navigation rail and the project records created for jobs."""

from __future__ import annotations

import datetime
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from services.config_service import AppConfig
from services.model_service import RuntimeInfo
from services.project_store import ProjectStore, ProjectSummary
from ui_qt.main_window import MainWindow, TranscriptionWorker, _slim_segments
from ui_qt.views import projects_home
from ui_qt.views.projects_home import ProjectsHome
from qt_close import close_and_drain

SEGMENTS = [
    {"start": 0.0, "end": 4.0, "speaker": "S1", "text": "Hello.", "words": [1]},
    {"start": 4.0, "end": 9.5, "speaker": "S2", "text": "Hi there."},
]


def summary(index: int, **overrides: object) -> ProjectSummary:
    data = dict(
        id=f"20260101-0000{index:02d}-abcdef{index:02d}", title=f"Project {index}", kind="file", status="completed",
        created="2026-01-01T10:00:00", updated=f"2026-01-01T10:{index:02d}:00", duration_s=3137.0, speaker_count=4, model="small",
    )
    data.update(overrides)
    return ProjectSummary(**data)  # type: ignore[arg-type]


class HelperTests(unittest.TestCase):
    def test_greeting_by_time_of_day_and_name(self) -> None:
        at = lambda hour: datetime.datetime(2026, 1, 1, hour)  # noqa: E731
        self.assertEqual(projects_home.greeting(at(9), "dave"), "Good morning, Dave")
        self.assertEqual(projects_home.greeting(at(14), "dave"), "Good afternoon, Dave")
        self.assertEqual(projects_home.greeting(at(21), "dave"), "Good evening, Dave")
        self.assertEqual(projects_home.greeting(at(14), "root"), "Good afternoon")
        self.assertEqual(projects_home.greeting(at(14), ""), "Good afternoon")

    def test_duration_and_facts(self) -> None:
        self.assertEqual(projects_home.format_duration(3137), "52:17")
        self.assertEqual(projects_home.format_duration(3984), "1:06:24")
        self.assertEqual(projects_home.format_duration(-5), "0:00")
        self.assertEqual(projects_home.describe(summary(1)), "52:17  •  4 speakers")
        self.assertEqual(projects_home.describe(summary(1, speaker_count=1)), "52:17  •  1 speaker")
        self.assertEqual(projects_home.describe(summary(1, duration_s=0, speaker_count=0)), "")

    def test_activity_wording(self) -> None:
        now = datetime.datetime(2026, 6, 12, 15, 0)
        self.assertEqual(projects_home.format_activity("2026-06-12T14:14:00", now), "Today 2:14 PM")
        self.assertEqual(projects_home.format_activity("2026-06-11T11:03:00", now), "Yesterday 11:03 AM")
        self.assertEqual(projects_home.format_activity("2026-06-01T09:05:00", now), "Jun 1, 9:05 AM")
        self.assertEqual(projects_home.format_activity("garbage", now), "")


class QtCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls._app = QApplication.instance() or QApplication([])


class ProjectsHomeTests(QtCase):
    def _home(self, summaries: list[ProjectSummary]) -> ProjectsHome:
        home = ProjectsHome()
        home.resize(1000, 700)
        home.show()
        home.set_projects(summaries)
        QApplication.processEvents()
        self.addCleanup(home.close)
        return home

    def test_empty_state_shows_primary_action(self) -> None:
        home = self._home([])
        self.assertTrue(home.empty_state.isVisible())
        self.assertFalse(home.grid_host.isVisible())
        hits: list[int] = []
        home.browse_requested.connect(lambda: hits.append(1))
        home.empty_state.action_button.click()
        self.assertEqual(hits, [1])

    def test_first_screen_has_no_advanced_settings(self) -> None:
        home = self._home([])
        for text in ("Model", "Diarization", "OCR", "Compute", "Hotwords"):
            self.assertFalse(any(label.text().startswith(text) for label in home.findChildren(type(home.footer))), text)
        self.assertEqual(home.footer.text(), "🛡  All processing is local by default.")

    def test_recent_limit_and_view_all(self) -> None:
        home = self._home([summary(i) for i in range(1, 9)])
        self.assertEqual(len(home._cards), projects_home.RECENT_LIMIT)
        self.assertTrue(home.view_all_button.isVisible())
        home.view_all_button.click()
        self.assertEqual(len(home._cards), 8)
        self.assertEqual(home.recent_title.text(), "All projects")
        home.view_all_button.click()
        self.assertEqual(len(home._cards), projects_home.RECENT_LIMIT)

    def test_few_projects_hide_view_all(self) -> None:
        home = self._home([summary(1), summary(2)])
        self.assertFalse(home.view_all_button.isVisible())
        self.assertTrue(home.grid_host.isVisible())
        self.assertFalse(home.empty_state.isVisible())

    def test_card_shows_title_facts_and_status(self) -> None:
        home = self._home([summary(1, status="transcribing")])
        card = home._cards[summary(1).id]
        self.assertEqual(card.title_label.text(), "Project 1")
        self.assertEqual(card.facts_label.text(), "52:17  •  4 speakers")
        self.assertEqual(card.status_chip.text(), "Transcribing…")
        home.set_progress(summary(1).id, 37)
        self.assertEqual(card.status_chip.text(), "Transcribing… 37%")
        home.set_progress("missing", 5)  # unknown ids are ignored

    def test_card_click_and_key_open_the_project(self) -> None:
        home = self._home([summary(1)])
        opened: list[str] = []
        home.open_project_requested.connect(opened.append)
        card = home._cards[summary(1).id]
        QTest.mouseClick(card, Qt.LeftButton)
        QTest.keyClick(card, Qt.Key_Return)
        self.assertEqual(opened, [summary(1).id] * 2)

    def test_secondary_actions_emit(self) -> None:
        home = self._home([])
        seen: list[str] = []
        home.live_requested.connect(lambda: seen.append("live"))
        home.import_folder_requested.connect(lambda: seen.append("folder"))
        home.browse_requested.connect(lambda: seen.append("browse"))
        QTest.mouseClick(home.live_card, Qt.LeftButton)
        QTest.mouseClick(home.import_card, Qt.LeftButton)
        QTest.mouseClick(home.drop_card, Qt.LeftButton)
        self.assertEqual(seen, ["live", "folder", "browse"])


class SegmentPlumbingTests(QtCase):
    def test_slim_segments_drops_words_and_junk(self) -> None:
        slim = _slim_segments([*SEGMENTS, "junk", None])
        self.assertEqual(slim[0], {"start": 0.0, "end": 4.0, "speaker": "S1", "text": "Hello."})
        self.assertEqual(len(slim), 2)
        self.assertEqual(_slim_segments(None), [])  # type: ignore[arg-type]

    def test_worker_keeps_segments_before_emitting_finished(self) -> None:
        runtime = RuntimeInfo(device="cpu", compute_type="int8", gpu_name="N/A", vram_gb=0.0, cpu_count=8)
        worker = TranscriptionWorker(
            "a.wav", "small", run_mode="full", use_diarization=True, diar_backend="accurate", max_speakers=None,
            use_visual_analysis=False, visual_profile="balanced", visual_ocr_backend="auto", visual_scope="slides_only",
            visual_sample_seconds=1.0, language=None, runtime_override=runtime,
        )
        seen: list[tuple[bool, list]] = []
        worker.finished.connect(lambda cancelled, *_: seen.append((cancelled, list(worker.last_segments))))
        worker._emit_finished({"cancelled": False, "transcript": "t", "segments": _slim_segments(SEGMENTS)}, "latest")
        self.assertEqual(seen[0][0], False)
        self.assertEqual(len(seen[0][1]), 2)  # already set when the slot runs
        worker._emit_finished({"cancelled": True}, "latest")  # no segments key
        self.assertEqual(worker.last_segments, [])


class MainWindowProjectTests(QtCase):
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
        self.addCleanup(close_and_drain, window)
        return window

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _media(self, name: str = "meeting.wav") -> str:
        path = Path(self._tmp.name) / name
        path.write_bytes(b"RIFF")
        return str(path)

    def test_opens_on_projects_and_nav_switches_pages(self) -> None:
        win = self._build_window()
        self.assertEqual(win.nav_rail.current(), "projects")
        self.assertIs(win.main_stack.currentWidget(), win.projects_home)
        for key, index in (("transcribe", 1), ("templates", 2), ("settings", 3), ("projects", 0)):
            win.nav_rail.set_current(key)
            self.assertEqual(win.main_stack.currentIndex(), index, key)

    def test_live_and_transcribe_nav_switch_input_mode(self) -> None:
        win = self._build_window()
        win.nav_rail.set_current("live")
        self.assertTrue(win._is_live_mode())
        win.nav_rail.set_current("transcribe")
        self.assertFalse(win._is_live_mode())

    def test_nav_does_not_change_mode_while_live_capture_runs(self) -> None:
        win = self._build_window()
        win.nav_rail.set_current("live")
        win._live_capture_active = True
        win.nav_rail.set_current("transcribe")
        self.assertTrue(win._is_live_mode())
        win._live_capture_active = False

    def test_choosing_a_file_on_home_goes_to_transcribe(self) -> None:
        win = self._build_window()
        path = self._media()
        win.projects_home.file_chosen.emit(path)
        self.assertEqual(win.media_path, path)
        self.assertEqual(win.nav_rail.current(), "transcribe")

    def test_unsupported_file_stays_on_home(self) -> None:
        win = self._build_window()
        bad = Path(self._tmp.name) / "notes.txt"
        bad.write_text("x")
        with patch("ui_qt.main_window.QMessageBox.warning"):
            win.projects_home.file_chosen.emit(str(bad))
        self.assertIsNone(win.media_path)
        self.assertEqual(win.nav_rail.current(), "projects")

    def test_home_actions_are_blocked_while_a_job_runs(self) -> None:
        win = self._build_window()
        win._live_capture_active = True
        with patch("ui_qt.main_window.QMessageBox.information") as info:
            win.projects_home.file_chosen.emit(self._media())
            win.projects_home.live_requested.emit()
        self.assertEqual(info.call_count, 2)
        self.assertIsNone(win.media_path)
        win._live_capture_active = False

    def test_job_creates_and_completes_a_project(self) -> None:
        win = self._build_window()
        path = self._media("Quarterly Review.wav")
        win._begin_project(path, "small")
        store: ProjectStore = win.project_store
        (record,) = store.list_summaries()
        self.assertEqual((record.title, record.status, record.kind), ("Quarterly Review", "transcribing", "file"))
        self.assertEqual(len(win.projects_home._cards), 1)
        win.worker = SimpleNamespace(last_segments=SEGMENTS)
        win._finish_project("completed", transcript="[S1] Hello.\n[S2] Hi there.")
        project = store.load(record.id)
        assert project is not None
        self.assertEqual(project.status, "completed")
        self.assertEqual(project.speaker_count, 2)
        self.assertEqual(project.duration_s, 9.5)
        self.assertNotIn("words", project.segments[0])
        self.assertIsNone(win._active_project_id)
        win.worker = None

    def test_failed_job_without_output_leaves_no_project(self) -> None:
        win = self._build_window()
        win._begin_project(self._media(), "small")
        win._finish_project("failed")
        self.assertEqual(win.project_store.list_summaries(), [])

    def test_cancelled_job_keeps_partial_text(self) -> None:
        win = self._build_window()
        win._begin_project(self._media(), "small")
        win._finish_project("cancelled", transcript="partial text")
        (record,) = win.project_store.list_summaries()
        self.assertEqual(record.status, "cancelled")

    def test_kind_follows_batch_and_live(self) -> None:
        win = self._build_window()
        win._batch_active = True
        win._begin_project(self._media("a.wav"), "small")
        win._batch_active = False
        win._live_finalizing = True
        win.live_title_input.setText("Standup")
        win._begin_project(self._media("b.wav"), "small")
        win._live_finalizing = False
        kinds = {s.title: s.kind for s in win.project_store.list_summaries()}
        self.assertEqual(kinds, {"a": "batch", "Standup": "live"})

    def test_progress_updates_the_card(self) -> None:
        win = self._build_window()
        win._begin_project(self._media(), "small")
        win._on_transcription_progress(42)
        (card,) = win.projects_home._cards.values()
        self.assertEqual(card.status_chip.text(), "Transcribing… 42%")

    def test_open_rename_and_delete_project(self) -> None:
        win = self._build_window()
        project = win.project_store.create(title="Vendor Call", kind="file")
        win.project_store.update(project.id, status="completed", transcript_text="[S1] Hello", segments=SEGMENTS)
        win._refresh_projects()
        win._open_project(project.id)
        self.assertEqual(win.text_area.toPlainText(), "[S1] Hello")
        self.assertEqual(win.nav_rail.current(), "transcribe")
        self.assertTrue(win.copy_btn.isEnabled())
        with patch("ui_qt.main_window.QInputDialog.getText", return_value=("Vendor Sync", True)):
            win._rename_project(project.id)
        self.assertEqual(win.project_store.load(project.id).title, "Vendor Sync")
        with patch("ui_qt.main_window.QMessageBox.question", return_value=QMessageBox.No):
            win._delete_project(project.id)
        self.assertIsNotNone(win.project_store.load(project.id))
        with patch("ui_qt.main_window.QMessageBox.question", return_value=QMessageBox.Yes):
            win._delete_project(project.id)
        self.assertIsNone(win.project_store.load(project.id))
        self.assertEqual(win.projects_home._summaries, [])

    def test_reveal_uses_the_project_folder_only(self) -> None:
        win = self._build_window()
        project = win.project_store.create(title="x")
        with patch("ui_qt.main_window.open_folder") as opener:
            win._reveal_project(project.id)
            win._reveal_project("../../etc")
        opener.assert_called_once_with(str(win.project_store.project_dir(project.id)))

    def test_sidebar_collapse_still_works_with_the_rail(self) -> None:
        win = self._build_window()
        win._toggle_sidebar_collapsed()
        self.assertTrue(win.nav_rail.is_collapsed())
        self.assertEqual(win.nav_rail.width(), 52)
        win._toggle_sidebar_collapsed()
        self.assertFalse(win.nav_rail.is_collapsed())
        self.assertTrue(win.nav_rail.button("projects").isVisible())


if __name__ == "__main__":
    unittest.main()
