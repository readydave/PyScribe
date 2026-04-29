"""Tests for the interactive launcher menu."""

from __future__ import annotations

import unittest
import os
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("PYSCRIBE_REEXEC_DONE", "1")
os.environ.setdefault("PYSCRIBE_LOG_DIR", "/tmp/pyscribe-test-logs")
os.environ.setdefault("PYSCRIBE_CACHE_DIR", "/tmp/pyscribe-test-cache")

import main


class LauncherPromptTests(unittest.TestCase):
    def test_prompt_launch_mode_defaults_to_qt_on_timeout(self) -> None:
        with patch("main._input_with_timeout", return_value=None), patch("builtins.print"):
            self.assertEqual(main.prompt_launch_mode(), "1")

    def test_prompt_launch_mode_accepts_listener_choice(self) -> None:
        with patch("main._input_with_timeout", return_value="2\n"), patch("builtins.print"):
            self.assertEqual(main.prompt_launch_mode(), "2")

    def test_prompt_launch_mode_reprompts_after_invalid_choice(self) -> None:
        with patch("main._input_with_timeout", side_effect=["bad\n", "1\n"]), patch("builtins.print"):
            self.assertEqual(main.prompt_launch_mode(), "1")


if __name__ == "__main__":
    unittest.main()
