"""Tests for LLM post-processing execution service."""

from __future__ import annotations

from io import BytesIO
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from services.llm_connection_service import LLMConnectionProfile
from services.llm_postprocess_service import LLMPostprocessRequest, run_llm_postprocess
from services.prompt_template_service import PromptTemplate


class _MockHTTPResponse:
    def __init__(self, payload: object) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _MockHTTPResponse:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:  # noqa: ANN001
        return False


class _RawHTTPResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _RawHTTPResponse:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:  # noqa: ANN001
        return False


def _profile(
    *,
    provider: str = "ollama",
    default_model: str | None = "llama3",
    base_url: str = "http://127.0.0.1:11434",
    api_key: str | None = None,
) -> LLMConnectionProfile:
    return LLMConnectionProfile(
        name="test-profile",
        provider=provider,
        scope="local",
        base_url=base_url,
        api_key=api_key,
        default_model=default_model,
        timeout_seconds=8.0,
        verify_tls=True,
        allowed_cidrs=("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"),
        enabled=True,
        allow_concurrent_with_local_transcription=False,
    )


def _template() -> PromptTemplate:
    return PromptTemplate(
        id="meeting-summary",
        name="Meeting Summary",
        version=1,
        description="",
        tags=("meetings",),
        output_format="markdown",
        enabled=True,
        built_in=True,
        system_prompt="You are a concise summarizer.",
        user_prompt_scaffold="Summarize the meeting.",
        source_path="inline",
    )


class LLMPostprocessServiceTests(unittest.TestCase):
    def test_missing_transcript_returns_failure(self) -> None:
        result = run_llm_postprocess(
            _profile(),
            _template(),
            LLMPostprocessRequest(transcript_text="", ocr_text="", notes_text="", selected_model=None),
        )
        self.assertEqual(result.status, "fail")
        self.assertEqual(result.error_code, "missing_transcript")

    def test_missing_model_returns_failure(self) -> None:
        result = run_llm_postprocess(
            _profile(default_model=None),
            _template(),
            LLMPostprocessRequest(transcript_text="hello", ocr_text="", notes_text="", selected_model=None),
        )
        self.assertEqual(result.status, "fail")
        self.assertEqual(result.error_code, "missing_model")

    def test_ollama_success(self) -> None:
        captured_prompt = {"value": ""}

        def _mock_urlopen(request, timeout=8):  # noqa: ANN001, ARG001
            self.assertTrue(request.full_url.endswith("/api/generate"))
            payload = json.loads(request.data.decode("utf-8"))
            captured_prompt["value"] = str(payload.get("prompt", ""))
            return _MockHTTPResponse({"response": "Summary output", "done": True})

        with patch("services.llm_postprocess_service.urlrequest.urlopen", side_effect=_mock_urlopen):
            result = run_llm_postprocess(
                _profile(),
                _template(),
                LLMPostprocessRequest(
                    transcript_text="Speaker A: status update",
                    ocr_text="Slide: Q1 metrics",
                    notes_text="Keep it to 5 bullets.",
                    selected_model=None,
                ),
            )

        self.assertEqual(result.status, "pass")
        self.assertEqual(result.output_text, "Summary output")
        self.assertIn("OCR Report:", captured_prompt["value"])
        self.assertIn("Additional Notes:", captured_prompt["value"])

    def test_openai_compatible_success(self) -> None:
        def _mock_urlopen(request, timeout=8):  # noqa: ANN001, ARG001
            self.assertTrue(request.full_url.endswith("/v1/chat/completions"))
            return _MockHTTPResponse({"choices": [{"message": {"content": "Action items..."}}]})

        with patch("services.llm_postprocess_service.urlrequest.urlopen", side_effect=_mock_urlopen):
            result = run_llm_postprocess(
                _profile(
                    provider="openai_compatible",
                    base_url="http://127.0.0.1:1234",
                    default_model="qwen2.5",
                    api_key="token",
                ),
                _template(),
                LLMPostprocessRequest(
                    transcript_text="Decision: proceed with pilot.",
                    ocr_text="",
                    notes_text="",
                    selected_model=None,
                ),
            )

        self.assertEqual(result.status, "pass")
        self.assertIn("Action items", result.output_text)

    def test_auth_failure_maps_code(self) -> None:
        def _raise_401(request, timeout=8):  # noqa: ANN001, ARG001
            raise HTTPError(request.full_url, 401, "Unauthorized", hdrs=None, fp=BytesIO(b"{}"))

        with patch("services.llm_postprocess_service.urlrequest.urlopen", side_effect=_raise_401):
            result = run_llm_postprocess(
                _profile(),
                _template(),
                LLMPostprocessRequest(
                    transcript_text="hello",
                    ocr_text="",
                    notes_text="",
                    selected_model=None,
                ),
            )

        self.assertEqual(result.status, "fail")
        self.assertEqual(result.error_code, "auth_failed")

    def test_invalid_json_maps_api_mismatch(self) -> None:
        def _invalid_json(request, timeout=8):  # noqa: ANN001, ARG001
            return _RawHTTPResponse(b"not-json")

        with patch("services.llm_postprocess_service.urlrequest.urlopen", side_effect=_invalid_json):
            result = run_llm_postprocess(
                _profile(),
                _template(),
                LLMPostprocessRequest(
                    transcript_text="hello",
                    ocr_text="",
                    notes_text="",
                    selected_model=None,
                ),
            )

        self.assertEqual(result.status, "fail")
        self.assertEqual(result.error_code, "api_mismatch")


if __name__ == "__main__":
    unittest.main()
