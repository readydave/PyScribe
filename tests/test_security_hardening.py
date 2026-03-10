"""Tests for security hardening changes."""

from __future__ import annotations

import ipaddress
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from services.hf_auth_service import clear_session_hf_token, get_hf_token, save_hf_token
from services.llm_connection_service import evaluate_profile_scope_policy, load_llm_profiles
from services.logging_service import _find_writable_log_path


class LLMTlsPolicyTests(unittest.TestCase):
    def test_https_lan_profile_requires_tls_verification(self) -> None:
        profile = load_llm_profiles(
            [
                {
                    "name": "lan-openai",
                    "provider": "openai_compatible",
                    "scope": "lan",
                    "base_url": "https://192.168.1.20:1234",
                    "verify_tls": False,
                }
            ]
        )[0]

        with patch(
            "services.llm_connection_service._resolve_host_ips",
            return_value=[ipaddress.ip_address("192.168.1.20")],
        ):
            ok, code, detail = evaluate_profile_scope_policy(profile)

        self.assertFalse(ok)
        self.assertEqual(code, "policy_tls_verification_required")
        self.assertIn("localhost/loopback", detail or "")

    def test_https_local_loopback_can_disable_tls_verification(self) -> None:
        profile = load_llm_profiles(
            [
                {
                    "name": "local-openai",
                    "provider": "openai_compatible",
                    "scope": "local",
                    "base_url": "https://127.0.0.1:1234",
                    "verify_tls": False,
                }
            ]
        )[0]

        ok, code, detail = evaluate_profile_scope_policy(profile)

        self.assertTrue(ok)
        self.assertIsNone(code)
        self.assertIsNone(detail)


class HFTokenHardeningTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_session_hf_token()

    def test_save_hf_token_session_only_does_not_persist_or_set_env(self) -> None:
        with patch("services.hf_auth_service.HfFolder.save_token") as mock_save, patch.dict(
            os.environ,
            {},
            clear=True,
        ):
            save_hf_token("hf_secret", persist=False)
            token = get_hf_token()
            self.assertNotIn("HF_TOKEN", os.environ)

        mock_save.assert_not_called()
        self.assertEqual(token, "hf_secret")

    def test_save_hf_token_can_persist_when_requested(self) -> None:
        with patch("services.hf_auth_service.HfFolder.save_token") as mock_save:
            save_hf_token("hf_secret", persist=True)

        mock_save.assert_called_once_with("hf_secret")
        self.assertEqual(get_hf_token(), "hf_secret")


class LoggingHardeningTests(unittest.TestCase):
    def test_default_log_path_prefers_user_private_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "services.logging_service.Path.home",
            return_value=Path(temp_dir),
        ), patch.dict(os.environ, {}, clear=True):
            log_path = _find_writable_log_path()

        self.assertEqual(log_path, Path(temp_dir) / ".pyscribe" / "logs" / "pyscribe.log")


if __name__ == "__main__":
    unittest.main()
