"""Tests for LLM connection profile parsing and diagnostics."""

from __future__ import annotations

from io import BytesIO
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from services.llm_connection_service import (
    get_failure_suggestions,
    load_llm_profiles,
    test_connection,
)


class _MockHTTPResponse:
    def __init__(self, payload: object) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _MockHTTPResponse:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:  # noqa: ANN001
        return False


class LLMConnectionServiceTests(unittest.TestCase):
    def test_load_profiles_filters_invalid_provider(self) -> None:
        profiles = load_llm_profiles(
            [
                {"name": "good", "provider": "ollama", "scope": "local", "base_url": "http://127.0.0.1:11434"},
                {"name": "bad", "provider": "unknown", "scope": "local", "base_url": "http://127.0.0.1:1"},
            ]
        )
        self.assertEqual(len(profiles), 1)
        self.assertEqual(profiles[0].provider, "ollama")

    def test_local_scope_rejects_non_local_endpoint(self) -> None:
        profile = load_llm_profiles(
            [
                {
                    "name": "bad-local",
                    "provider": "ollama",
                    "scope": "local",
                    "base_url": "http://192.168.1.20:11434",
                }
            ]
        )[0]
        result = test_connection(profile)
        self.assertEqual(result.status, "fail")
        self.assertEqual(result.failure_code, "policy_blocked_non_local")

    def test_ollama_connection_success(self) -> None:
        profile = load_llm_profiles(
            [
                {
                    "name": "local-ollama",
                    "provider": "ollama",
                    "scope": "local",
                    "base_url": "http://127.0.0.1:11434",
                    "default_model": "llama3",
                }
            ]
        )[0]

        def _mock_urlopen(request, timeout=8):  # noqa: ANN001, ARG001
            url = request.full_url
            if url.endswith("/api/tags"):
                return _MockHTTPResponse({"models": [{"name": "llama3"}]})
            if url.endswith("/api/ps"):
                return _MockHTTPResponse({"models": [{"name": "llama3"}]})
            if url.endswith("/api/generate"):
                return _MockHTTPResponse({"response": "OK", "done": True})
            raise AssertionError(f"Unexpected URL: {url}")

        with patch("services.llm_connection_service.urlrequest.urlopen", side_effect=_mock_urlopen):
            result = test_connection(profile)

        self.assertEqual(result.status, "pass")
        self.assertEqual(result.selected_model, "llama3")
        self.assertIn("llama3", result.detected_models)

    def test_openai_auth_failure_maps_to_auth_failed(self) -> None:
        profile = load_llm_profiles(
            [
                {
                    "name": "remote-openai",
                    "provider": "openai_compatible",
                    "scope": "local",
                    "base_url": "http://127.0.0.1:1234",
                }
            ]
        )[0]

        def _raise_401(request, timeout=8):  # noqa: ANN001, ARG001
            raise HTTPError(request.full_url, 401, "Unauthorized", hdrs=None, fp=BytesIO(b"{}"))

        with patch("services.llm_connection_service.urlrequest.urlopen", side_effect=_raise_401):
            result = test_connection(profile)

        self.assertEqual(result.status, "fail")
        self.assertEqual(result.failure_code, "auth_failed")

    def test_openai_no_models(self) -> None:
        profile = load_llm_profiles(
            [
                {
                    "name": "remote-openai",
                    "provider": "openai_compatible",
                    "scope": "local",
                    "base_url": "http://127.0.0.1:1234",
                }
            ]
        )[0]

        def _mock_urlopen(request, timeout=8):  # noqa: ANN001, ARG001
            if request.full_url.endswith("/v1/models"):
                return _MockHTTPResponse({"data": []})
            raise AssertionError(f"Unexpected URL: {request.full_url}")

        with patch("services.llm_connection_service.urlrequest.urlopen", side_effect=_mock_urlopen):
            result = test_connection(profile)

        self.assertEqual(result.status, "fail")
        self.assertEqual(result.failure_code, "no_models_available")

    def test_failure_suggestions(self) -> None:
        suggestions = get_failure_suggestions("timeout")
        self.assertGreaterEqual(len(suggestions), 1)


if __name__ == "__main__":
    unittest.main()
