"""Tests for the Listener stage strip renderer and the transcribe() generator's yields."""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("gradio")

import gradio as gr  # noqa: E402

import app  # noqa: E402
from services.job_stages import ACTIVE, DISABLED, DONE, FAILED, PENDING, STAGE_ORDER, Stage, StageInfo  # noqa: E402


def _stages(**states: str) -> dict[Stage, StageInfo]:
    out = {stage: StageInfo() for stage in STAGE_ORDER}
    out[Stage.SAVE] = StageInfo(state=DISABLED)
    for name, state in states.items():
        out[Stage(name)] = StageInfo(state=state, percent=40 if state == ACTIVE else 0)
    return out


@dataclass
class _Upd:
    stages: dict = field(default_factory=lambda: _stages())
    status: str = ""
    transcript: str = ""
    overall_pct: int = 0
    result: object = None
    error: BaseException | None = None
    cancelled: bool = False
    finished: bool = False


class RenderStageStripTests(unittest.TestCase):
    def test_labels_states_and_hidden_save(self) -> None:
        html_out = app.render_stage_strip(_stages(load=DONE, transcribe=ACTIVE, speakers=DISABLED))
        for label in ("Load model", "Transcribe", "Speakers", "Visuals"):
            self.assertIn(label, html_out)
        self.assertNotIn("Save", html_out)
        self.assertIn("pyscribe-stage-done", html_out)
        self.assertIn("pyscribe-stage-active", html_out)
        self.assertIn("Transcribe 40%", html_out)
        self.assertIn("pyscribe-stage-disabled", html_out)

    def test_failed_state_and_note_are_escaped(self) -> None:
        out = app.render_stage_strip(_stages(transcribe=FAILED), note="<script>alert(1)</script>")
        self.assertIn("pyscribe-stage-failed", out)
        self.assertNotIn("<script>", out)
        self.assertIn("&lt;script&gt;", out)

    def test_empty_stages_render_nothing(self) -> None:
        self.assertEqual(app.render_stage_strip(None), "")

    def test_css_uses_theme_variables_for_both_modes(self) -> None:
        css = app.build_css()
        self.assertIn(".pyscribe-stage-active", css)
        self.assertIn("body.dark", css)
        self.assertIn("--pyscribe-stage-done", css)


def _run(fake_updates, *, raise_before: Exception | None = None):
    def fake_job(**kwargs):
        if raise_before is not None:
            raise raise_before
        yield from fake_updates

    patches = [
        patch.object(app, "_ensure_listener_runtime"),
        patch.object(app, "run_media_job", fake_job),
        patch.object(app.pyscribe_services, "normalize_model_name", lambda m: m),
        patch.object(
            app.pyscribe_services,
            "resolve_transcription_model",
            lambda m: SimpleNamespace(display_name=m, supports_diarization=True),
        ),
    ]
    for p in patches:
        p.start()
    try:
        return list(
            app.transcribe("a.wav", "small", "full", False, "off", "", False, "balanced", "auto", 1.0, "", False, progress=lambda *a, **k: None)
        )
    finally:
        for p in reversed(patches):
            p.stop()


class TranscribeGeneratorTests(unittest.TestCase):
    def test_success_yields_strip_and_final_message(self) -> None:
        result = SimpleNamespace(transcript=" hello ", cancelled=False)
        updates = [
            _Upd(stages=_stages(load=ACTIVE), status="Loading", overall_pct=5),
            _Upd(stages=_stages(load=ACTIVE), status="Loading", overall_pct=6),  # heartbeat, same content
            _Upd(stages=_stages(load=DONE, transcribe=ACTIVE), status="Transcribing", transcript="hel", overall_pct=50),
            _Upd(stages=_stages(load=DONE, transcribe=DONE), status="Done", transcript="hello", overall_pct=100, result=result, finished=True),
        ]
        out = _run(updates)
        self.assertTrue(all(len(item) == 6 for item in out))
        self.assertEqual(len(out), 4)  # start + 2 changed updates + final (heartbeat skipped)
        self.assertIn("Transcribing", out[2][0])
        self.assertEqual(out[2][1], "hel")
        self.assertIn("Transcription complete!", out[-1][0])
        self.assertEqual(out[-1][1], "hello")
        self.assertEqual(out[-1][4], "Transcription complete!")
        self.assertIn("pyscribe-stage-done", out[-1][5])

    def test_cancel_yields_cancelled_message(self) -> None:
        result = SimpleNamespace(transcript="part", cancelled=True)
        out = _run([_Upd(status="Cancelled", cancelled=True, finished=True, result=result)])
        self.assertEqual(out[-1][0], "Status: Transcription cancelled.")
        self.assertEqual(out[-1][4], "Cancelled.")
        self.assertEqual(out[-1][1], "part")

    def test_error_renders_failed_strip_then_raises_gradio_error(self) -> None:
        updates = [_Upd(stages=_stages(transcribe=FAILED), error=RuntimeError("boom <b>"), finished=True)]
        gen_out: list = []

        def fake_job(**kwargs):
            yield from updates

        with patch.object(app, "_ensure_listener_runtime"), patch.object(app, "run_media_job", fake_job), patch.object(
            app.pyscribe_services, "normalize_model_name", lambda m: m
        ), patch.object(
            app.pyscribe_services, "resolve_transcription_model", lambda m: SimpleNamespace(display_name=m, supports_diarization=True)
        ):
            gen = app.transcribe("a.wav", "small", "full", False, "off", "", False, "balanced", "auto", 1.0, "", False, progress=lambda *a, **k: None)
            with self.assertRaises(gr.Error) as ctx:
                for item in gen:
                    gen_out.append(item)
        self.assertIn("boom", str(ctx.exception))
        self.assertIn("pyscribe-stage-failed", gen_out[-1][5])
        self.assertNotIn("<b>", gen_out[-1][5])
        self.assertFalse(app._transcription_active.is_set())

    def test_exception_before_first_update_becomes_gradio_error(self) -> None:
        with self.assertRaises(gr.Error):
            _run([], raise_before=TypeError("bad kwarg"))
        self.assertFalse(app._transcription_active.is_set())


if __name__ == "__main__":
    unittest.main()
