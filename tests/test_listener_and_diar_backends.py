"""Unit tests for listener security helpers and diarization backend compatibility."""

from __future__ import annotations

import importlib
import os
import sys
import types
import unittest
from unittest.mock import patch

from services import listener_security_service


def _import_diar_backends_with_stub() -> types.ModuleType:
    stub = types.ModuleType("diarization")
    stub.run_diarization = lambda *args, **kwargs: []
    with patch.dict(sys.modules, {"diarization": stub}):
        sys.modules.pop("diar_backends", None)
        return importlib.import_module("diar_backends")


class ListenerSecurityServiceTests(unittest.TestCase):
    def test_reject_legacy_auth_pass_flag(self) -> None:
        with self.assertRaises(SystemExit):
            listener_security_service.reject_legacy_auth_pass_flag(["main.py", "--auth-pass=secret"])

    def test_resolve_listener_auth_from_env(self) -> None:
        with patch.dict(
            os.environ,
            {"PYSCRIBE_AUTH_USER": "alice", "PYSCRIBE_AUTH_PASS": "secret"},
            clear=False,
        ):
            user, password = listener_security_service.resolve_listener_auth(None)
        self.assertEqual(user, "alice")
        self.assertEqual(password, "secret")

    def test_validate_listener_security_requires_auth_for_nonlocal(self) -> None:
        with self.assertRaises(SystemExit):
            listener_security_service.validate_listener_security(
                "0.0.0.0",
                auth_user=None,
                auth_pass=None,
                allow_nonlocal_host=True,
            )

    def test_validate_listener_security_requires_auth_for_share(self) -> None:
        with self.assertRaises(SystemExit):
            listener_security_service.validate_listener_security(
                "127.0.0.1",
                auth_user=None,
                auth_pass=None,
                allow_nonlocal_host=False,
                share=True,
            )


class DiarBackendsCompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.diar_backends = _import_diar_backends_with_stub()

    def tearDown(self) -> None:
        sys.modules.pop("diar_backends", None)

    def test_backend_availability_filters_pyannote(self) -> None:
        def _fake_find_spec(name: str, *args: object, **kwargs: object) -> object | None:
            del args, kwargs
            if name == "pyannote.audio":
                return object()
            return None

        with patch("importlib.util.find_spec", side_effect=_fake_find_spec):
            status = self.diar_backends.backend_availability()

        self.assertTrue(status["accurate"].available)
        self.assertTrue(status["fast"].available)
        self.assertNotIn("sortformer", status)


if __name__ == "__main__":
    unittest.main()
