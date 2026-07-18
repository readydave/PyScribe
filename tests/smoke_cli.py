"""Basic smoke checks that require no third-party installs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_cmd(*args: str) -> None:
    proc = subprocess.run(args, cwd=ROOT, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(f"Command failed: {' '.join(args)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")


def test_main_help() -> None:
    run_cmd(sys.executable, "main.py", "--help")


def test_mode_helps() -> None:
    run_cmd(sys.executable, "main.py", "serve", "--help")
    run_cmd(sys.executable, "main.py", "qt", "--help")


def test_py_compile_core_files() -> None:
    run_cmd(
        sys.executable,
        "-m",
        "compileall",
        "-q",
        "main.py",
        "app.py",
        "utils.py",
        "models.py",
        "diarization.py",
        "diar_backends.py",
        "services",
        "ui_qt",
    )
