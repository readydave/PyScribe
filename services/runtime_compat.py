"""Runtime compatibility helpers."""

from __future__ import annotations

import platform
import re
import sys


def ensure_platform_sys_version_compat() -> None:
    """
    Patches platform._sys_version for conda-forge style sys.version strings.
    This avoids ValueError crashes in dependencies that call platform.python_implementation().
    """
    # Apply only once per process.
    if getattr(platform, "_pyscribe_sysver_patch", False):
        return

    try:
        platform.python_implementation()
        return
    except ValueError:
        pass

    original = platform._sys_version  # type: ignore[attr-defined]

    def _safe_sys_version(sys_version=None):
        try:
            return original(sys_version)
        except ValueError:
            text = sys_version or sys.version
            match = re.search(
                r"(?P<ver>\d+\.\d+\.\d+).*?\((?P<buildno>[^,]+),\s*(?P<builddate>[^)]+)\)\s*\[(?P<compiler>[^\]]+)\]",
                text,
            )
            if match:
                return (
                    "CPython",
                    match.group("ver"),
                    "",
                    "",
                    match.group("buildno").strip(),
                    match.group("builddate").strip(),
                    match.group("compiler").strip(),
                )
            return ("CPython", platform.python_version(), "", "", "", "", "")

    platform._sys_version = _safe_sys_version  # type: ignore[attr-defined]
    platform._pyscribe_sysver_patch = True  # type: ignore[attr-defined]

