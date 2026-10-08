from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from services.mcp_service import (
    MAX_QUEUED_JOBS,
    JobHooks,
    JobManager,
    McpToolError,
    RunnerOutput,
    TranscriptStore,
    allowed_roots,
    clean_names_terms,
    page_text,
    validate_media_path,
)


class _TempCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve()


class MediaPathTests(_TempCase):
    def _file(self, name: str, content: bytes = b"x") -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def test_accepts_media_inside_a_root(self) -> None:
        audio = self._file("talks/meeting.MP3")
        self.assertEqual(validate_media_path(str(audio), [self.root]), audio)

    def test_rejects_bad_input(self) -> None:
        for bad in (None, 5, "", "   ", "a\x00b"):
            with self.assertRaises(McpToolError):
                validate_media_path(bad, [self.root])

    def test_rejects_missing_directories_and_wrong_types(self) -> None:
        with self.assertRaises(McpToolError):
            validate_media_path(str(self.root / "nope.mp3"), [self.root])
        with self.assertRaises(McpToolError):
            validate_media_path(str(self.root), [self.root])
        for name in ("notes.txt", "id_rsa", "script.py", "a.mp3.exe", "archive.zip"):
            with self.assertRaises(McpToolError, msg=name):
                validate_media_path(str(self._file(name)), [self.root])

    def test_rejects_files_outside_roots_including_traversal_and_symlinks(self) -> None:
        inside = self.root / "inside"
        outside = self.root / "outside"
        inside.mkdir()
        secret = self._file("outside/secret.mp3")
        with self.assertRaises(McpToolError):
            validate_media_path(str(secret), [inside])
        with self.assertRaises(McpToolError):
            validate_media_path(str(inside / ".." / "outside" / "secret.mp3"), [inside])
        link = inside / "link.mp3"
        try:
            link.symlink_to(secret)
        except OSError:
            self.skipTest("symlinks unavailable")
        with self.assertRaises(McpToolError):
            validate_media_path(str(link), [inside])
        self.assertTrue(outside.is_dir())

    def test_no_roots_is_an_error(self) -> None:
        with self.assertRaises(McpToolError):
            validate_media_path(str(self._file("a.mp3")), [])

    def test_roots_come_from_the_environment_or_home(self) -> None:
        env = {"PYSCRIBE_MCP_ROOTS": os.pathsep.join([str(self.root), str(self.root / "missing")])}
        self.assertEqual(allowed_roots(env), [self.root])
        with patch("pathlib.Path.home", return_value=self.root):
            self.assertEqual(allowed_roots({}), [self.root])


class TranscriptStoreTests(_TempCase):
    def _store(self, live: Path | None = None) -> TranscriptStore:
        return TranscriptStore(self.root / "store", live_root=live)

    def test_save_list_and_read_round_trip(self) -> None:
        store = self._store()
        tid = store.save(source_name="Team Sync (v2).mp4", model="small", text="[S1] hi", plain_text="hi",
                         duration_seconds=12.34)
        summaries = store.list_summaries()
        self.assertEqual([s.id for s in summaries], [tid])
        self.assertTrue(summaries[0].has_speakers)
        self.assertEqual(store.read(tid)[0], "[S1] hi")
        self.assertEqual(store.read(tid, speaker_labels=False)[0], "hi")
        self.assertEqual(store.read(tid)[1]["source_file"], "Team Sync (v2).mp4")
        if os.name != "nt":
            self.assertEqual((store.directory / f"{tid}.json").stat().st_mode & 0o777, 0o600)
            self.assertEqual(store.directory.stat().st_mode & 0o777, 0o700)

    def test_ids_cannot_escape_the_store(self) -> None:
        store = self._store()
        (self.root / "secret.json").write_text(json.dumps({"id": "secret", "text": "TOP SECRET"}))
        for bad in ("../secret", "..\\secret", "/etc/passwd", "secret.json", "", "a", "live-../x", "LIVE-x"):
            with self.assertRaises(McpToolError, msg=bad):
                store.read(bad)

    def test_corrupt_files_are_ignored(self) -> None:
        store = self._store()
        store.directory.mkdir(parents=True)
        (store.directory / "broken.json").write_text("{not json")
        (store.directory / "wrongid.json").write_text(json.dumps({"id": "../../x", "text": "y"}))
        self.assertEqual(store.list_summaries(), [])

    def _live_session(self, name: str, status: str = "completed", title: str = "Standup") -> Path:
        live = self.root / "live"
        folder = live / name
        folder.mkdir(parents=True)
        final = folder / "final.txt"
        final.write_text("live words", encoding="utf-8")
        (folder / "session.json").write_text(json.dumps(
            {"status": status, "session_title": title, "started_at": "2026-10-08T10:00:00Z",
             "selected_model": "small", "final_transcript_path": str(final)}))
        return live

    def test_live_sessions_are_listed_and_readable(self) -> None:
        live = self._live_session("2026-10-08_1000")
        store = self._store(live)
        listed = store.list_summaries(source="live")
        self.assertEqual([s.id for s in listed], ["live-2026-10-08_1000"])
        text, meta = store.read("live-2026-10-08_1000")
        self.assertEqual(text, "live words")
        self.assertEqual(meta["source"], "live")
        self.assertEqual(store.list_summaries(source="saved"), [])

    def test_unfinished_and_unknown_live_sessions_are_not_readable(self) -> None:
        live = self._live_session("running", status="recording")
        store = self._store(live)
        self.assertEqual(store.list_summaries(source="live"), [])
        for bad in ("live-running", "live-missing", "live-../live/running"):
            with self.assertRaises(McpToolError, msg=bad):
                store.read(bad)

    def test_live_session_pointing_outside_its_folder_is_ignored(self) -> None:
        live = self._live_session("evil")
        secret = self.root / "secret.txt"
        secret.write_text("password")
        meta_path = live / "evil" / "session.json"
        meta = json.loads(meta_path.read_text())
        meta["final_transcript_path"] = str(secret)
        meta_path.write_text(json.dumps(meta))
        store = self._store(live)
        self.assertEqual(store.list_summaries(source="live"), [])
        with self.assertRaises(McpToolError):
            store.read("live-evil")

    def test_limit_and_ordering(self) -> None:
        store = self._store()
        ids = []
        for n in range(3):
            ids.append(store.save(source_name=f"f{n}.mp3", model="m", text="t", plain_text="t", duration_seconds=1))
            time.sleep(0.02)
        newest_first = [s.id for s in store.list_summaries()]
        self.assertEqual(newest_first, list(reversed(ids)))
        self.assertEqual(len(store.list_summaries(limit=2)), 2)


class PagingTests(unittest.TestCase):
    def test_pages_cover_the_text_exactly(self) -> None:
        text = "abcdefghij" * 500
        offset, collected = 0, ""
        while offset is not None:
            page = page_text(text, offset, 1000)
            collected += page["text"]
            self.assertEqual(page["total_chars"], len(text))
            offset = page["next_offset"]
        self.assertEqual(collected, text)

    def test_limits_are_clamped(self) -> None:
        self.assertEqual(page_text("x" * 5000, -5, 1)["returned_chars"], 1000)
        self.assertEqual(page_text("x" * 500_000, 0, 10**9)["returned_chars"], 100_000)
        past_end = page_text("short", 99, 1000)
        self.assertEqual((past_end["text"], past_end["next_offset"]), ("", None))

    def test_names_and_terms_are_flattened_and_limited(self) -> None:
        self.assertEqual(clean_names_terms("  Okafor,\n  Kubernetes  "), "Okafor, Kubernetes")
        self.assertEqual(len(clean_names_terms("x" * 900)), 500)


class JobManagerTests(_TempCase):
    def _manager(self, runner) -> JobManager:
        return JobManager(runner, TranscriptStore(self.root / "store"))

    def test_job_runs_and_saves_a_transcript(self) -> None:
        def runner(params, hooks: JobHooks) -> RunnerOutput:
            hooks.stage("transcribing", "working")
            hooks.progress(55)
            return RunnerOutput(text="[S1] hello", plain_text="hello", model="small", duration_seconds=3.0)

        manager = self._manager(runner)
        job = manager.submit({"file_name": "call.mp3"})
        job = manager.wait(job.id, 10)
        self.assertEqual(job.status, "completed")
        self.assertEqual(job.percent, 100.0)
        self.assertTrue(job.transcript_id)
        self.assertEqual(manager._store.read(job.transcript_id)[0], "[S1] hello")

    def test_jobs_run_one_at_a_time_in_order(self) -> None:
        order: list[str] = []
        running = {"now": 0, "max": 0}
        gate = threading.Event()

        def runner(params, hooks) -> RunnerOutput:
            running["now"] += 1
            running["max"] = max(running["max"], running["now"])
            order.append(params["file_name"])
            gate.wait(2)
            running["now"] -= 1
            return RunnerOutput("text", "text", "m", 1.0)

        manager = self._manager(runner)
        jobs = [manager.submit({"file_name": f"{n}.mp3"}) for n in range(3)]
        self.assertEqual(manager.queue_position(jobs[2]), 3 if jobs[0].status == "queued" else 2)
        gate.set()
        for job in jobs:
            manager.wait(job.id, 10)
        self.assertEqual(order, ["0.mp3", "1.mp3", "2.mp3"])
        self.assertEqual(running["max"], 1)

    def test_failure_is_reported_and_the_worker_survives(self) -> None:
        def runner(params, hooks) -> RunnerOutput:
            if params["file_name"] == "bad.mp3":
                raise RuntimeError("boom")
            if params["file_name"] == "model.mp3":
                raise McpToolError("The model is not downloaded yet.")
            return RunnerOutput("ok text", "ok text", "m", 1.0)

        manager = self._manager(runner)
        bad = manager.wait(manager.submit({"file_name": "bad.mp3"}).id, 10)
        self.assertEqual(bad.status, "failed")
        self.assertIn("RuntimeError", bad.error or "")
        model = manager.wait(manager.submit({"file_name": "model.mp3"}).id, 10)
        self.assertEqual(model.error, "The model is not downloaded yet.")
        good = manager.wait(manager.submit({"file_name": "good.mp3"}).id, 10)
        self.assertEqual(good.status, "completed")

    def test_empty_transcript_is_a_failure(self) -> None:
        manager = self._manager(lambda p, h: RunnerOutput("  ", "", "m", 1.0))
        job = manager.wait(manager.submit({"file_name": "silence.mp3"}).id, 10)
        self.assertEqual(job.status, "failed")
        self.assertIn("No speech", job.error or "")

    def test_cancel_queued_and_running_jobs(self) -> None:
        started = threading.Event()

        def runner(params, hooks: JobHooks) -> RunnerOutput:
            started.set()
            for _ in range(100):
                if hooks.cancelled:
                    break
                time.sleep(0.02)
            return RunnerOutput("partial", "partial", "m", 1.0)

        manager = self._manager(runner)
        first = manager.submit({"file_name": "a.mp3"})
        second = manager.submit({"file_name": "b.mp3"})
        self.assertTrue(started.wait(5))
        self.assertEqual(manager.cancel(second.id).status, "cancelled")
        manager.cancel(first.id)
        done = manager.wait(first.id, 10)
        self.assertEqual(done.status, "cancelled")
        self.assertIsNone(done.transcript_id)  # a cancelled job saves nothing

    def test_queue_limit_and_unknown_ids(self) -> None:
        release = threading.Event()
        manager = self._manager(lambda p, h: (release.wait(5), RunnerOutput("t", "t", "m", 1.0))[1])
        jobs = []
        try:
            for n in range(MAX_QUEUED_JOBS + 1):  # one runs, the rest wait
                jobs.append(manager.submit({"file_name": f"{n}.mp3"}))
                time.sleep(0.01)
            with self.assertRaises(McpToolError):
                manager.submit({"file_name": "one-too-many.mp3"})
        finally:
            release.set()
            for job in jobs:
                manager.wait(job.id, 20)  # let the worker finish before the temp folder is removed
        for bad in ("nope", "", "../x"):
            with self.assertRaises(McpToolError):
                manager.get(bad)

    def test_wait_times_out_without_finishing(self) -> None:
        release = threading.Event()
        manager = self._manager(lambda p, h: (release.wait(5), RunnerOutput("t", "t", "m", 1.0))[1])
        job = manager.submit({"file_name": "slow.mp3"})
        started = time.monotonic()
        waited = manager.wait(job.id, 0.3)
        self.assertLess(time.monotonic() - started, 2)
        self.assertFalse(waited.done)
        release.set()
        self.assertTrue(manager.wait(job.id, 10).done)


if __name__ == "__main__":
    unittest.main()
