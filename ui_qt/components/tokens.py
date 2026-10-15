"""Spacing, radius, type and motion tokens for the Qt component library.

Colours are not defined here; they come from the active theme palette (``ui_qt.theme``).
"""

from __future__ import annotations

import os

SPACE: dict[str, int] = {"xs": 4, "sm": 8, "md": 12, "lg": 16, "xl": 24, "xxl": 32}
RADIUS: dict[str, int] = {"sm": 6, "md": 8, "lg": 12, "pill": 999}

# Supporting UI sits at 12-14 px; the transcript is larger and user-adjustable (see config ``transcript_font_px``).
TYPE_PX: dict[str, int] = {"caption": 12, "body": 13, "label": 14, "heading": 17, "title": 22}
TRANSCRIPT_PX_DEFAULT = 16
TRANSCRIPT_PX_MIN = 12
TRANSCRIPT_PX_MAX = 28

MOTION_MS: dict[str, int] = {"fast": 120, "base": 180, "slow": 260}
REDUCED_MOTION_CHOICES = ("auto", "on", "off")

_reduced_motion = False
_transcript_px = TRANSCRIPT_PX_DEFAULT


def _env_prefers_reduced_motion() -> bool:
    return os.environ.get("PYSCRIBE_REDUCED_MOTION", "").strip().lower() in {"1", "true", "yes", "on"}


def configure_motion(setting: str = "auto") -> bool:
    """Apply the ``reduced_motion`` setting (``auto``, ``on`` or ``off``) and return the effective flag.

    Qt exposes no OS reduced-motion hint, so ``auto`` follows ``PYSCRIBE_REDUCED_MOTION`` and is otherwise off.
    """
    global _reduced_motion
    choice = str(setting or "auto").strip().lower()
    if choice == "on":
        _reduced_motion = True
    elif choice == "off":
        _reduced_motion = False
    else:
        _reduced_motion = _env_prefers_reduced_motion()
    return _reduced_motion


def reduced_motion() -> bool:
    return _reduced_motion


def motion_ms(name: str = "base") -> int:
    """Animation duration in milliseconds; 0 when reduced motion is on."""
    if _reduced_motion:
        return 0
    return MOTION_MS.get(name, MOTION_MS["base"])


def set_transcript_px(value: int) -> int:
    """Set the transcript text size in px (clamped) and return the value in use."""
    global _transcript_px
    _transcript_px = max(TRANSCRIPT_PX_MIN, min(TRANSCRIPT_PX_MAX, int(value)))
    return _transcript_px


def transcript_px() -> int:
    return _transcript_px
