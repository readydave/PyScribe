"""Tests for LLM connection profile parsing and diagnostics."""

from __future__ import annotations

from io import BytesIO
import json
import socket
import types
import unittest
import sys
from unittest.mock import patch
from urllib.error import HTTPError

from services.llm_connection_service import (
    discover_local_networks,
    get_failure_suggestions,
    load_llm_profiles,
    scan_lan_for_llm_instances,
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

    def test_lm_studio_provider_supported(self) -> None:
        profiles = load_llm_profiles(
            [
                {
                    "name": "lm",
                    "provider": "lm_studio",
                    "scope": "local",
                    "base_url": "http://127.0.0.1:1234",
                }
            ]
        )
        self.assertEqual(len(profiles), 1)
        self.assertEqual(profiles[0].provider, "lm_studio")

    def test_profile_api_key_can_resolve_from_env_reference(self) -> None:
        with patch.dict("os.environ", {"PYSCRIBE_TEST_API_KEY": "env-secret"}, clear=False):
            profile = load_llm_profiles(
                [
                    {
                        "name": "env-key",
                        "provider": "openai_compatible",
                        "scope": "local",
                        "base_url": "http://127.0.0.1:1234",
                        "api_key": "env:PYSCRIBE_TEST_API_KEY",
                    }
                ]
            )[0]
        self.assertEqual(profile.api_key, "env-secret")

    def test_openai_compatible_base_url_normalizes_terminal_v1(self) -> None:
        profile = load_llm_profiles(
            [
                {
                    "name": "openai-v1",
                    "provider": "openai_compatible",
                    "scope": "local",
                    "base_url": "http://127.0.0.1:1234/v1",
                }
            ]
        )[0]
        self.assertEqual(profile.base_url, "http://127.0.0.1:1234")

    def test_lan_scope_rejects_loopback_endpoint(self) -> None:
        profile = load_llm_profiles(
            [
                {
                    "name": "lan-loopback",
                    "provider": "openai_compatible",
                    "scope": "lan",
                    "base_url": "http://127.0.0.1:1234",
                }
            ]
        )[0]
        result = test_connection(profile)
        self.assertEqual(result.status, "fail")
        self.assertEqual(result.failure_code, "policy_loopback_with_lan")

    def test_discover_local_networks_with_mocked_psutil(self) -> None:
        fake_addr = types.SimpleNamespace(
            family=socket.AF_INET,
            address="192.168.1.77",
            netmask="255.255.255.0",
        )
        fake_psutil = types.SimpleNamespace(net_if_addrs=lambda: {"Ethernet": [fake_addr]})
        with patch.dict(sys.modules, {"psutil": fake_psutil}):
            networks = discover_local_networks()
        self.assertGreaterEqual(len(networks), 1)
        self.assertIn("192.168.1.0/24", {item.network_cidr for item in networks})

    def test_scan_lan_discovers_ollama(self) -> None:
        def _mock_get(url: str, *, timeout_seconds: float, headers=None):  # noqa: ANN001, ARG001
            if url.endswith("127.0.0.1:11434/api/tags"):
                return {"models": [{"name": "llama3"}]}
            raise RuntimeError("not found")

        with patch("services.llm_connection_service._http_json_get", side_effect=_mock_get):
            results = scan_lan_for_llm_instances(
                ["127.0.0.0/30"],
                include_ollama=True,
                include_openai_compatible=False,
                include_lm_studio=False,
                timeout_seconds=0.05,
                max_hosts_per_network=4,
            )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].provider, "ollama")
        self.assertEqual(results[0].host, "127.0.0.1")

    def test_failure_suggestions(self) -> None:
        suggestions = get_failure_suggestions("timeout")
        self.assertGreaterEqual(len(suggestions), 1)


if __name__ == "__main__":
    unittest.main()
