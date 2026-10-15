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
        "py_compile",
        "main.py",
        "app.py",
        "services/config_service.py",
        "services/logging_service.py",
        "services/model_service.py",
        "services/multimodal_service.py",
        "services/transcription_service.py",
        "ui_qt/main_window.py",
        "ui_qt/benchmark_dialog.py",
    )
