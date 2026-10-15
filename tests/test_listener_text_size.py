"""Tests for the Listener text-size control (client-side CSS/JS embedded in app.py)."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import pytest

pytest.importorskip("gradio")

import app  # noqa: E402


class ListenerTextSizeTests(unittest.TestCase):
    def test_head_script_defines_control_and_storage_key(self) -> None:
        head = app.CUSTOM_HEAD
        self.assertIn("pyscribe-text-size", head)
        self.assertIn("pyscribe.listener.textScalePct", head)
        self.assertIn("--pyscribe-text-scale", head)

    def test_head_script_guards_storage_access(self) -> None:
        # localStorage can throw (private mode, blocked site data); both calls must be wrapped.
        self.assertGreaterEqual(app.CUSTOM_HEAD.count("catch (e)"), 2)

    def test_css_scales_gradio_text_variables(self) -> None:
        css = app.CUSTOM_CSS
        self.assertIn("#pyscribe-text-size", css)
        for var in ("--text-sm", "--text-md", "--text-lg"):
            self.assertIn(f"{var}: calc(", css)

    def test_interface_still_builds(self) -> None:
        # The ready flag skips runtime detection, which can re-exec the process.
        with (
            patch.object(app, "_LISTENER_RUNTIME_READY", True),
            patch.object(app, "ALL_MODELS", ["small"]),
            patch.object(app, "RECOMMENDED_MODEL", "small"),
            patch.object(app, "AVAILABLE_DIAR_BACKENDS", ["accurate"]),
        ):
            self.assertIsNotNone(app.create_interface())


if __name__ == "__main__":
    unittest.main()
