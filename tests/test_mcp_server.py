from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from mcp import Client  # noqa: E402
from mcp.client.stdio import StdioServerParameters  # noqa: E402

from services.mcp_server import SERVER_INSTRUCTIONS, create_server  # noqa: E402
from services.mcp_service import JobManager, RunnerOutput, TranscriptStore  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
READ_ONLY_TOOLS = {
    "list_transcription_models", "get_job", "wait_for_job", "list_transcripts", "get_transcript",
    "list_templates", "get_template",
}


def _text(result) -> str:  # noqa: ANN001
    return "".join(getattr(block, "text", "") for block in result.content)


class InProcessServerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve()
        (self.root / "meeting.mp3").write_bytes(b"audio")
        (self.root / "notes.txt").write_text("not audio")
        self.calls: list[dict] = []

        def runner(params, hooks):  # noqa: ANN001
            self.calls.append(params)
            hooks.progress(50)
            text = "[S1] " + "hello world. " * 3000
            return RunnerOutput(text=text, plain_text=text.replace("[S1] ", ""), model="small", duration_seconds=9.0)

        self.store = TranscriptStore(self.root / "store")
        self.manager = JobManager(runner, self.store)
        self.server = create_server(manager=self.manager, store=self.store, roots=[self.root], runner=runner)

    async def test_tools_are_registered_with_safe_annotations(self) -> None:
        async with Client(self.server) as client:
            tools = {tool.name: tool for tool in (await client.list_tools()).tools}
        self.assertEqual(
            set(tools),
            READ_ONLY_TOOLS | {"start_transcription", "cancel_job"},
        )
        for name in READ_ONLY_TOOLS:
            self.assertTrue(tools[name].annotations.read_only_hint, name)
        for tool in tools.values():
            self.assertFalse(tool.annotations.destructive_hint, tool.name)
            self.assertFalse(tool.annotations.open_world_hint, tool.name)

    async def test_instructions_warn_that_transcripts_are_untrusted(self) -> None:
        self.assertIn("untrusted", SERVER_INSTRUCTIONS.lower())
        self.assertIn("never as instructions", SERVER_INSTRUCTIONS)

    async def test_transcribe_wait_and_read_in_pages(self) -> None:
        async with Client(self.server) as client:
            started = await client.call_tool("start_transcription", {"path": str(self.root / "meeting.mp3"),
                                                                     "names_and_terms": "Okafor,\nKubernetes"})
            self.assertFalse(started.is_error)
            job_id = started.structured_content["job_id"]
            done = await client.call_tool("wait_for_job", {"job_id": job_id, "timeout_seconds": 20})
            self.assertEqual(done.structured_content["status"], "completed")
            transcript_id = done.structured_content["transcript_id"]

            listed = await client.call_tool("list_transcripts", {})
            self.assertEqual([t["id"] for t in listed.structured_content["transcripts"]], [transcript_id])

            collected, offset, pages = "", 0, 0
            while offset is not None:
                page = (await client.call_tool("get_transcript", {"transcript_id": transcript_id,
                                                                  "offset": offset, "max_chars": 15000})).structured_content
                collected += page["text"]
                offset = page["next_offset"]
                pages += 1
            self.assertGreater(pages, 1)
            self.assertEqual(len(collected), page["total_chars"])
            plain = (await client.call_tool("get_transcript", {"transcript_id": transcript_id,
                                                               "speaker_labels": False})).structured_content
            self.assertFalse(plain["text"].startswith("[S1]"))
        self.assertEqual(self.calls[0]["names_terms"], "Okafor, Kubernetes")
        self.assertEqual(self.calls[0]["file_name"], "meeting.mp3")

    async def test_bad_inputs_are_reported_as_errors_not_crashes(self) -> None:
        async with Client(self.server) as client:
            for tool, args in (
                ("start_transcription", {"path": str(self.root / "notes.txt")}),
                ("start_transcription", {"path": "/etc/passwd"}),
                ("start_transcription", {"path": str(self.root / "meeting.mp3"), "max_speakers": 99}),
                ("start_transcription", {"path": str(self.root / "meeting.mp3"), "language": "en; rm -rf"}),
                ("get_job", {"job_id": "nope"}),
                ("get_transcript", {"transcript_id": "../../etc/passwd"}),
                ("list_transcripts", {"source": "everything"}),
                ("get_template", {"template_id": "missing"}),
            ):
                result = await client.call_tool(tool, args)
                self.assertTrue(result.is_error, (tool, args))
            # the server is still healthy afterwards
            self.assertFalse((await client.call_tool("list_transcripts", {})).is_error)
        self.assertEqual(self.calls, [])

    async def test_templates_are_exposed_for_the_client_to_apply(self) -> None:
        async with Client(self.server) as client:
            listing = (await client.call_tool("list_templates", {})).structured_content
            self.assertIn("meeting-summary", [t["id"] for t in listing["templates"]])
            template = (await client.call_tool("get_template", {"template_id": "meeting-summary"})).structured_content
            self.assertTrue(template["system_prompt"])
            self.assertTrue(template["user_prompt"])

    async def test_cancel_job(self) -> None:
        import threading

        gate = threading.Event()

        def slow(params, hooks):  # noqa: ANN001
            gate.wait(5)
            return RunnerOutput("t", "t", "m", 1.0)

        manager = JobManager(slow, self.store)
        server = create_server(manager=manager, store=self.store, roots=[self.root], runner=slow)
        async with Client(server) as client:
            first = (await client.call_tool("start_transcription", {"path": str(self.root / "meeting.mp3")})).structured_content
            second = (await client.call_tool("start_transcription", {"path": str(self.root / "meeting.mp3")})).structured_content
            cancelled = (await client.call_tool("cancel_job", {"job_id": second["job_id"]})).structured_content
            self.assertEqual(cancelled["status"], "cancelled")
            gate.set()
            done = (await client.call_tool("wait_for_job", {"job_id": first["job_id"], "timeout_seconds": 20})).structured_content
            self.assertEqual(done["status"], "completed")


class StdioSmokeTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_process_speaks_clean_mcp_over_stdio(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            env = {**os.environ, "HOME": home, "USERPROFILE": home, "PYSCRIBE_MCP_ROOTS": home}
            params = StdioServerParameters(command=sys.executable, args=[str(REPO / "main.py"), "mcp"],
                                           env=env, cwd=str(REPO))
            async with Client(params, read_timeout_seconds=90) as client:
                names = {tool.name for tool in (await client.list_tools()).tools}
                self.assertIn("start_transcription", names)
                listing = await client.call_tool("list_transcripts", {})
                self.assertFalse(listing.is_error)
                self.assertEqual(listing.structured_content["transcripts"], [])
                refused = await client.call_tool("start_transcription", {"path": "/etc/passwd"})
                self.assertTrue(refused.is_error)


if __name__ == "__main__":
    unittest.main()
