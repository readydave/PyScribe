"""Tests for listener-side LLM post-processing helper functions."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import app
from services.llm_connection_service import LLMConnectionProfile
from services.llm_postprocess_service import LLMPostprocessResult
from services.prompt_template_service import PromptTemplate


def _profile(*, scope: str = "local", allow_concurrent: bool = False) -> LLMConnectionProfile:
    return LLMConnectionProfile(
        name="listener-profile",
        provider="ollama",
        scope=scope,
        base_url="http://127.0.0.1:11434",
        api_key=None,
        default_model="llama3",
        timeout_seconds=8.0,
        verify_tls=True,
        allowed_cidrs=("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"),
        enabled=True,
        allow_concurrent_with_local_transcription=allow_concurrent,
    )


def _template() -> PromptTemplate:
    return PromptTemplate(
        id="meeting-summary",
        name="Meeting Summary",
        version=1,
        description="",
        tags=(),
        output_format="markdown",
        enabled=True,
        built_in=True,
        system_prompt="system",
        user_prompt_scaffold="user scaffold",
        source_path="inline",
    )


class ListenerLLMPostprocessHelperTests(unittest.TestCase):
    def test_source_mode_visibility_current_transcript(self) -> None:
        file_update, paste_update = app._update_postprocess_source_fields("Current transcript")
        self.assertFalse(bool(file_update.get("visible")))
        self.assertFalse(bool(paste_update.get("visible")))

    def test_source_mode_visibility_upload_mode(self) -> None:
        file_update, paste_update = app._update_postprocess_source_fields("Upload/paste transcript")
        self.assertTrue(bool(file_update.get("visible")))
        self.assertTrue(bool(paste_update.get("visible")))

    def test_connection_test_requires_profile(self) -> None:
        with patch("app._find_enabled_llm_profile", return_value=None):
            status, _model_update = app.test_listener_llm_connection("", "")
        self.assertIn("no enabled profile", status.lower())

    def test_postprocess_blocks_local_profile_when_transcription_active(self) -> None:
        app._transcription_active.set()
        try:
            with patch("app._find_enabled_llm_profile", return_value=_profile(scope="local", allow_concurrent=True)):
                status, output = app.run_listener_llm_postprocess(
                    current_transcript="hello",
                    source_mode="Current transcript",
                    uploaded_transcript_file=None,
                    pasted_transcript="",
                    uploaded_ocr_file=None,
                    notes_text="",
                    profile_name="listener-profile",
                    template_id="meeting-summary",
                    selected_model="",
                )
        finally:
            app._transcription_active.clear()

        self.assertIn("blocked", status.lower())
        self.assertEqual(output, "")

    def test_postprocess_success_path(self) -> None:
        profile = _profile(scope="lan", allow_concurrent=True)
        expected = LLMPostprocessResult(
            status="pass",
            provider="ollama",
            model="llama3",
            output_text="Summary text",
            error_code=None,
            error_detail=None,
        )
        with patch("app._find_enabled_llm_profile", return_value=profile):
            with patch("app.pyscribe_services.get_prompt_template", return_value=_template()):
                with patch("app.pyscribe_services.run_llm_postprocess", return_value=expected):
                    status, output = app.run_listener_llm_postprocess(
                        current_transcript="hello",
                        source_mode="Current transcript",
                        uploaded_transcript_file=None,
                        pasted_transcript="",
                        uploaded_ocr_file=None,
                        notes_text="",
                        profile_name="listener-profile",
                        template_id="meeting-summary",
                        selected_model="",
                    )
        self.assertIn("complete", status.lower())
        self.assertEqual(output, "Summary text")


if __name__ == "__main__":
    unittest.main()
