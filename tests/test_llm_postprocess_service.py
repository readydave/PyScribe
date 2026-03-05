"""Tests for LLM post-processing execution service."""

from __future__ import annotations

from io import BytesIO
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from pathlib import Path
import tempfile

from services.llm_connection_service import LLMConnectionProfile
from services.llm_postprocess_service import (
    LLMPostprocessRequest,
    LLMRunControl,
    build_llm_payload_preview,
    prepare_llm_postprocess_payload,
    run_llm_postprocess,
)
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


class _StreamingHTTPResponse:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines

    def __iter__(self):  # noqa: ANN204
        return iter(self._lines)

    def __enter__(self) -> _StreamingHTTPResponse:
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

    def test_runtime_scope_policy_is_enforced(self) -> None:
        lan_profile = LLMConnectionProfile(
            name="lan-openai",
            provider="openai_compatible",
            scope="lan",
            base_url="http://127.0.0.1:1234",
            api_key=None,
            default_model="qwen2.5",
            timeout_seconds=8.0,
            verify_tls=True,
            allowed_cidrs=("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"),
            enabled=True,
            allow_concurrent_with_local_transcription=False,
        )
        result = run_llm_postprocess(
            lan_profile,
            _template(),
            LLMPostprocessRequest(transcript_text="hello", ocr_text="", notes_text="", selected_model=None),
        )
        self.assertEqual(result.status, "fail")
        self.assertEqual(result.error_code, "policy_loopback_with_lan")

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
                    extra_context_text="Prior sprint summary pasted by user.",
                    selected_model=None,
                ),
            )

        self.assertEqual(result.status, "pass")
        self.assertEqual(result.output_text, "Summary output")
        self.assertIn("OCR Report:", captured_prompt["value"])
        self.assertIn("Additional Notes:", captured_prompt["value"])
        self.assertIn("Pasted Context:", captured_prompt["value"])

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

    def test_openai_streaming_success_emits_chunks(self) -> None:
        streamed: list[str] = []

        def _mock_urlopen(request, timeout=8):  # noqa: ANN001, ARG001
            self.assertTrue(request.full_url.endswith("/v1/chat/completions"))
            payload = json.loads(request.data.decode("utf-8"))
            self.assertTrue(bool(payload.get("stream")))
            return _StreamingHTTPResponse(
                [
                    b'data: {"choices":[{"delta":{"content":"Action "}}]}\n',
                    b'data: {"choices":[{"delta":{"content":"items"}}]}\n',
                    b"data: [DONE]\n",
                ]
            )

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
                on_output_chunk=streamed.append,
            )

        self.assertEqual(result.status, "pass")
        self.assertEqual(result.output_text, "Action items")
        self.assertEqual("".join(streamed), "Action items")

    def test_cancelled_before_send_returns_cancelled(self) -> None:
        run_control = LLMRunControl()
        run_control.request_cancel()

        result = run_llm_postprocess(
            _profile(),
            _template(),
            LLMPostprocessRequest(
                transcript_text="hello",
                ocr_text="",
                notes_text="",
                selected_model=None,
            ),
            run_control=run_control,
        )

        self.assertEqual(result.status, "fail")
        self.assertEqual(result.error_code, "cancelled")

    def test_openai_streaming_cancel_midstream_returns_partial(self) -> None:
        streamed: list[str] = []
        run_control = LLMRunControl()

        def _on_output_chunk(chunk: str) -> None:
            streamed.append(chunk)
            run_control.request_cancel()

        def _mock_urlopen(request, timeout=8):  # noqa: ANN001, ARG001
            self.assertTrue(request.full_url.endswith("/v1/chat/completions"))
            return _StreamingHTTPResponse(
                [
                    b'data: {"choices":[{"delta":{"content":"Action "}}]}\n',
                    b'data: {"choices":[{"delta":{"content":"items"}}]}\n',
                    b"data: [DONE]\n",
                ]
            )

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
                on_output_chunk=_on_output_chunk,
                run_control=run_control,
            )

        self.assertEqual(result.status, "fail")
        self.assertEqual(result.error_code, "cancelled")
        self.assertEqual(result.output_text, "Action")
        self.assertEqual("".join(streamed), "Action ")

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

    def test_timeout_retries_with_extended_timeout(self) -> None:
        calls = {"count": 0}

        def _mock_urlopen(request, timeout=8):  # noqa: ANN001
            calls["count"] += 1
            if calls["count"] == 1:
                raise URLError("timed out")
            self.assertGreaterEqual(float(timeout), 30.0)
            return _MockHTTPResponse({"response": "Summary after retry", "done": True})

        with patch("services.llm_postprocess_service.urlrequest.urlopen", side_effect=_mock_urlopen):
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

        self.assertEqual(result.status, "pass")
        self.assertIn("Summary after retry", result.output_text)
        self.assertGreaterEqual(calls["count"], 2)

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

    def test_payload_preview_builder(self) -> None:
        payload = build_llm_payload_preview(
            template=_template(),
            request=LLMPostprocessRequest(
                transcript_text="Line one",
                ocr_text="Slide note",
                notes_text="Use bullets",
                extra_context_text="Company policy text",
            ),
        )
        self.assertIn("Transcript:", payload)
        self.assertIn("OCR Report:", payload)
        self.assertIn("Additional Notes:", payload)
        self.assertIn("Pasted Context:", payload)

    def test_prepare_payload_blocks_non_multimodal_without_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "shot.png"
            image_path.write_bytes(b"fake")
            request = LLMPostprocessRequest(
                transcript_text="hello",
                selected_model="llama3",
                image_paths=(str(image_path),),
                include_images=True,
                ocr_fallback_for_images=False,
            )
            prepared = prepare_llm_postprocess_payload(_profile(), _template(), request)
        self.assertEqual(prepared.status, "fail")
        self.assertEqual(prepared.error_code, "model_not_multimodal")

    def test_prepare_payload_uses_ocr_fallback_for_text_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "shot.png"
            image_path.write_bytes(b"fake")
            request = LLMPostprocessRequest(
                transcript_text="hello",
                selected_model="llama3",
                image_paths=(str(image_path),),
                include_images=True,
                image_ocr_backend="auto",
                ocr_fallback_for_images=True,
            )
            with patch("services.llm_postprocess_service.extract_text_from_images", return_value=("UI text", "pytesseract", None)):
                prepared = prepare_llm_postprocess_payload(_profile(), _template(), request)
        self.assertEqual(prepared.status, "pass")
        self.assertEqual(prepared.image_paths_for_payload, ())
        self.assertIn("Image OCR Context:", prepared.payload_text)

    def test_openai_multimodal_payload_includes_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "shot.png"
            image_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")

            def _mock_urlopen(request, timeout=8):  # noqa: ANN001, ARG001
                payload = json.loads(request.data.decode("utf-8"))
                content = payload["messages"][1]["content"]
                self.assertIsInstance(content, list)
                image_parts = [part for part in content if isinstance(part, dict) and part.get("type") == "image_url"]
                self.assertGreaterEqual(len(image_parts), 1)
                return _MockHTTPResponse({"choices": [{"message": {"content": "Vision summary"}}]})

            with patch("services.llm_postprocess_service.urlrequest.urlopen", side_effect=_mock_urlopen):
                result = run_llm_postprocess(
                    _profile(
                        provider="openai_compatible",
                        base_url="http://127.0.0.1:1234",
                        default_model="qwen2.5-vl",
                    ),
                    _template(),
                    LLMPostprocessRequest(
                        transcript_text="hello",
                        image_paths=(str(image_path),),
                        include_images=True,
                    ),
                )
        self.assertEqual(result.status, "pass")


if __name__ == "__main__":
    unittest.main()
