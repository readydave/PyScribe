"""Cross-platform OS helpers used by frontends."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess


def open_folder(path: str) -> None:
    """
    Opens a folder in the platform file manager.
    Raises RuntimeError if the platform command fails.
    """
    folder = os.path.abspath(path)
    system = platform.system().lower()

    try:
        if system == "windows":
            os.startfile(folder)  # type: ignore[attr-defined]
            return
        if system == "darwin":
            _run_open_cmd(["open", folder])
            return

        # Linux: try common handlers in order.
        for cmd in ("xdg-open", "gio", "gnome-open", "kde-open"):
            if shutil.which(cmd):
                if cmd == "gio":
                    _run_open_cmd([cmd, "open", folder])
                else:
                    _run_open_cmd([cmd, folder])
                return
        raise RuntimeError("No folder opener found (tried xdg-open/gio/gnome-open/kde-open).")
    except Exception as exc:
        raise RuntimeError(f"Could not open folder '{folder}': {exc}") from exc


def _run_open_cmd(args: list[str]) -> None:
    env = os.environ.copy()
    # Avoid Conda/Pinokio library overrides breaking system openers (/bin/sh, xdg-open, etc.).
    for key in (
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "PYTHONHOME",
        "PYTHONPATH",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
    ):
        env.pop(key, None)
    result = subprocess.run(args, capture_output=True, text=True, check=False, env=env)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(stderr or f"Command failed: {' '.join(args)}")
