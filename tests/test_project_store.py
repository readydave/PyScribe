from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services import project_store
from services.project_store import Project, ProjectStore, is_valid_project_id, sanitize_segments

SEGMENTS = [
    {"start": 0.0, "end": 4.2, "speaker": "S1", "text": "Hello everyone.", "words": [{"w": "x"}]},
    {"start": 4.2, "end": 9.0, "speaker": "S2", "text": "Thanks for joining."},
    {"start": 9.0, "end": 12.5, "speaker": "S1", "text": "Let's start."},
    {"start": 12.5, "end": 13.0, "speaker": "S?", "text": "Um."},
]


class ProjectStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = ProjectStore(Path(self._tmp.name) / "projects")

    def test_create_save_load_round_trip(self) -> None:
        project = self.store.create(title="Quarterly Review", kind="file", media_path="/m/a.mp4", model="small")
        self.assertTrue(is_valid_project_id(project.id))
        self.assertEqual(project.status, "transcribing")
        project.status = "completed"
        project.segments = sanitize_segments(SEGMENTS)
        project.duration_s = 13.0
        self.store.save(project)
        loaded = self.store.load(project.id)
        assert loaded is not None
        self.assertEqual(loaded.title, "Quarterly Review")
        self.assertEqual(loaded.status, "completed")
        self.assertEqual(loaded.speaker_count, 2)  # S? is not a speaker
        self.assertNotIn("words", loaded.segments[0])

    def test_files_and_dirs_are_private(self) -> None:
        project = self.store.create(title="x")
        folder = self.store.project_dir(project.id)
        self.assertEqual(stat.S_IMODE(folder.stat().st_mode), 0o700)
        for name in ("project.json", "summary.json"):
            self.assertEqual(stat.S_IMODE((folder / name).stat().st_mode), 0o600)
        self.assertEqual([p.name for p in folder.iterdir() if p.name.endswith(".tmp")], [])

    def test_invalid_ids_never_leave_the_store(self) -> None:
        for bad in ("../etc", "..", "", "20260101-000000-zzzzzzzz", "/abs", "a/b", "20260101-000000-abcdef12/.."):
            with self.assertRaises(ValueError):
                self.store.project_dir(bad)
            self.assertIsNone(self.store.load(bad))
            self.assertFalse(self.store.delete(bad) if is_valid_project_id(bad) else False)

    def test_delete_removes_only_the_project(self) -> None:
        media = Path(self._tmp.name) / "keep.wav"
        media.write_bytes(b"x")
        project = self.store.create(title="x", media_path=str(media))
        self.assertTrue(self.store.delete(project.id))
        self.assertIsNone(self.store.load(project.id))
        self.assertTrue(media.exists())
        self.assertFalse(self.store.delete(project.id))

    def test_delete_refuses_a_symlinked_project_folder(self) -> None:
        target = Path(self._tmp.name) / "elsewhere"
        target.mkdir()
        (target / "precious.txt").write_text("keep")
        self.store.root.mkdir(parents=True)
        link = self.store.root / "20260101-000000-abcdef12"
        link.symlink_to(target, target_is_directory=True)
        self.assertFalse(self.store.delete(link.name))
        self.assertTrue((target / "precious.txt").exists())
        self.assertEqual(self.store.list_summaries(), [])

    def test_list_is_newest_first_and_skips_junk(self) -> None:
        a = self.store.create(title="first")
        b = self.store.create(title="second")
        self.store.update(a.id, title="first renamed")  # bumps updated
        (self.store.root / "not-a-project").mkdir()
        broken = self.store.root / "20260101-000000-deadbeef"
        broken.mkdir()
        (broken / "project.json").write_text("{not json", encoding="utf-8")
        summaries = self.store.list_summaries()
        self.assertEqual({s.id for s in summaries}, {a.id, b.id})
        self.assertEqual(summaries[0].title, "first renamed")

    def test_list_reads_summary_file_not_the_transcript(self) -> None:
        project = self.store.create(title="big")
        project.segments = sanitize_segments(SEGMENTS)
        project.duration_s = 13
        self.store.save(project)
        folder = self.store.project_dir(project.id)
        (folder / "project.json").write_text("{corrupt", encoding="utf-8")
        summary = self.store.list_summaries()[0]
        self.assertEqual((summary.title, summary.speaker_count, summary.duration_s), ("big", 2, 13.0))

    def test_summary_rebuilt_when_missing(self) -> None:
        project = self.store.create(title="rebuild")
        (self.store.project_dir(project.id) / "summary.json").unlink()
        self.assertEqual(self.store.list_summaries()[0].title, "rebuild")

    def test_update_segments_rebuilds_text_and_speakers(self) -> None:
        project = self.store.create(title="x")
        updated = self.store.update_segments(project.id, SEGMENTS[:2])
        assert updated is not None
        self.assertEqual(updated.transcript_text, "[S1] Hello everyone.\n[S2] Thanks for joining.")
        self.assertEqual(updated.speaker_count, 2)

    def test_rename_and_speaker_names(self) -> None:
        project = self.store.create(title="x")
        self.assertEqual(self.store.rename(project.id, "  New title  ").title, "New title")
        with self.assertRaises(ValueError):
            self.store.rename(project.id, "   ")
        named = self.store.rename_speaker(project.id, "S1", "Sarah")
        assert named is not None
        self.assertEqual(named.speaker_names, {"S1": "Sarah"})
        self.assertEqual(self.store.rename_speaker(project.id, "S1", "").speaker_names, {})
        self.assertIsNone(self.store.rename_speaker("20260101-000000-abcdef12", "S1", "x"))

    def test_ai_output_is_stored(self) -> None:
        project = self.store.create(title="x")
        saved = self.store.set_ai_output(project.id, "meeting-summary", "# Summary", profile="local")
        assert saved is not None
        self.assertEqual(saved.ai_outputs["meeting-summary"]["text"], "# Summary")
        self.assertEqual(saved.ai_outputs["meeting-summary"]["profile"], "local")

    def test_update_rejects_unknown_and_protected_fields(self) -> None:
        project = self.store.create(title="x")
        for field in ("id", "created", "speaker_count", "nope"):
            with self.assertRaises(ValueError):
                self.store.update(project.id, **{field: 1})

    def test_loading_sanitises_hostile_json(self) -> None:
        project = self.store.create(title="x")
        path = self.store.project_dir(project.id) / "project.json"
        data = json.loads(path.read_text())
        data.update(
            title="t" * 500, kind="evil", status="weird", duration_s="NaN", segments=[{"text": "ok", "start": "a"}, 5, {"text": ""}],
            speaker_names={"S1": "n" * 200, "": "x"}, preview_peaks=[5, -1, "z"], ai_outputs={"k": "not a dict"},
        )
        path.write_text(json.dumps(data), encoding="utf-8")
        loaded = self.store.load(project.id)
        assert loaded is not None
        self.assertEqual(len(loaded.title), project_store.MAX_TITLE)
        self.assertEqual((loaded.kind, loaded.status, loaded.duration_s), ("file", "completed", 0.0))
        self.assertEqual(loaded.segments, [{"start": 0.0, "end": 0.0, "speaker": None, "text": "ok"}])
        self.assertEqual(loaded.speaker_names, {"S1": "n" * project_store.MAX_SPEAKER_NAME})
        self.assertEqual(loaded.preview_peaks, [1.0, 0.0, 0.0])
        self.assertEqual(loaded.ai_outputs, {})

    def test_mismatched_id_in_file_is_rejected(self) -> None:
        a, b = self.store.create(title="a"), self.store.create(title="b")
        (self.store.project_dir(a.id) / "summary.json").write_text(
            (self.store.project_dir(b.id) / "summary.json").read_text(), encoding="utf-8"
        )
        titles = {s.id: s.title for s in self.store.list_summaries()}
        self.assertEqual(titles[a.id], "a")  # rebuilt from its own project.json

    def test_segment_caps_and_ordering(self) -> None:
        cleaned = sanitize_segments([{"text": "a", "start": 5, "end": 2}])
        self.assertEqual(cleaned[0]["end"], 5.0)  # end never precedes start
        self.assertEqual(sanitize_segments("nope"), [])


class ProjectsRootTests(unittest.TestCase):
    def test_env_then_config_then_default(self) -> None:
        with patch.dict(os.environ, {project_store.ENV_PROJECTS_DIR: "/tmp/from-env"}):
            self.assertEqual(project_store.resolve_projects_root("/tmp/cfg"), Path("/tmp/from-env"))
        with patch.dict(os.environ, {project_store.ENV_PROJECTS_DIR: ""}):
            self.assertEqual(project_store.resolve_projects_root("/tmp/cfg"), Path("/tmp/cfg"))
            self.assertEqual(project_store.resolve_projects_root("relative/dir"), project_store.DEFAULT_PROJECTS_ROOT)
            self.assertEqual(project_store.resolve_projects_root(None), project_store.DEFAULT_PROJECTS_ROOT)


class SetSpeakerSegmentsTests(unittest.TestCase):
    def test_replaces_segments_resets_names_and_rebuilds_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ProjectStore(Path(tmp) / "projects")
            project = store.create(title="Call")
            store.update(project.id, segments=sanitize_segments(SEGMENTS), speaker_names={"S1": "Sarah"})
            relabelled = [
                {"start": 0.0, "end": 4.2, "speaker": "S1", "text": "Hello everyone."},
                {"start": 4.2, "end": 9.0, "speaker": "S3", "text": "Thanks for joining."},
            ]
            updated = store.set_speaker_segments(project.id, relabelled)
            assert updated is not None
            self.assertEqual(updated.speaker_names, {})
            self.assertEqual(updated.speaker_count, 2)
            self.assertEqual(updated.transcript_text, "[S1] Hello everyone.\n[S3] Thanks for joining.")

    def test_unknown_project_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ProjectStore(Path(tmp) / "projects")
            self.assertIsNone(store.set_speaker_segments("20260101-000000-abcdef12", []))


if __name__ == "__main__":
    unittest.main()
