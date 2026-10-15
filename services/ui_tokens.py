"""UI tokens shared by the Qt app and the Gradio listener (no Qt imports).

Colours now come from ``services.ui_themes``; this module keeps the font constants and the default
(Iron-gall) palettes under the names the UIs already import.
"""

from __future__ import annotations

from pathlib import Path

from services.ui_themes import (
    DEFAULT_SPEAKERS,
    DEFAULT_THEME_ID,
    PRESET_BY_ID,
    Palette,
    derive_palette,
    resolve_theme,
)

FONT_FAMILY = "Atkinson Hyperlegible Next"
FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
FONT_FILES = {
    400: "AtkinsonHyperlegibleNext-Regular.ttf",
    500: "AtkinsonHyperlegibleNext-Medium.ttf",
    600: "AtkinsonHyperlegibleNext-SemiBold.ttf",
    700: "AtkinsonHyperlegibleNext-Bold.ttf",
}

_DEFAULT_THEME = PRESET_BY_ID[DEFAULT_THEME_ID]

# Default-theme palettes, kept for code that doesn't need a user-selected theme.
PALETTES: dict[str, Palette] = {
    "light": derive_palette(_DEFAULT_THEME.light, "light"),
    "dark": derive_palette(_DEFAULT_THEME.dark, "dark"),
}

# Stage colours as (light, dark) per stage, and the default speaker colours.
STAGE_COLORS: dict[str, tuple[str, str]] = {
    "transcribe": (_DEFAULT_THEME.light.stage_transcribe, _DEFAULT_THEME.dark.stage_transcribe),
    "speakers": (_DEFAULT_THEME.light.stage_speakers, _DEFAULT_THEME.dark.stage_speakers),
    "visuals": (_DEFAULT_THEME.light.stage_visuals, _DEFAULT_THEME.dark.stage_visuals),
}
SPEAKER_COLORS = DEFAULT_SPEAKERS


def speaker_color(label: str, mode: str) -> str:
    """Colour for a speaker label such as ``S1`` in the default theme; unknown speakers use muted text."""
    digits = "".join(ch for ch in str(label) if ch.isdigit())
    resolved = resolve_theme(_DEFAULT_THEME, mode)
    if not digits:
        return resolved.palette.muted
    return resolved.speaker_colors[(int(digits) - 1) % len(resolved.speaker_colors)]
