"""Cross-platform OS helpers used by frontends."""

from __future__ import annotations

import os
import platform
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
            subprocess.Popen(["open", folder], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return

        subprocess.Popen(["xdg-open", folder], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:
        raise RuntimeError(f"Could not open folder '{folder}': {exc}") from exc
