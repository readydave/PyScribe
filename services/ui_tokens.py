"""UI colour and font tokens shared by the Qt app and the Gradio listener (no Qt imports)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

FONT_FAMILY = "Atkinson Hyperlegible Next"
FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
FONT_FILES = {
    400: "AtkinsonHyperlegibleNext-Regular.ttf",
    500: "AtkinsonHyperlegibleNext-Medium.ttf",
    600: "AtkinsonHyperlegibleNext-SemiBold.ttf",
    700: "AtkinsonHyperlegibleNext-Bold.ttf",
}


@dataclass(frozen=True)
class Palette:
    """Colour tokens. Rubric red is reserved for the primary action and live recording."""

    page: str
    surface: str
    sidebar: str
    card: str
    input_bg: str
    rule: str
    ink: str
    muted: str
    accent: str
    accent_hover: str
    accent_text: str
    rubric: str
    rubric_hover: str
    done: str
    done_text: str
    bar_active: str
    disabled_bg: str
    disabled_text: str
    log_bg: str
    log_text: str


PALETTES: dict[str, Palette] = {
    "light": Palette(
        page="#EEF1F5",
        surface="#F6F8FB",
        sidebar="#E6EAF1",
        card="#FFFFFF",
        input_bg="#FFFFFF",
        rule="#D3D9E3",
        ink="#26304A",
        muted="#566079",
        accent="#26304A",
        accent_hover="#38456A",
        accent_text="#FFFFFF",
        rubric="#C2412D",
        rubric_hover="#A83624",
        done="#2F7D6B",
        done_text="#FFFFFF",
        bar_active="#9DB2E3",
        disabled_bg="#C5CCD9",
        disabled_text="#6B7488",
        log_bg="#1B2133",
        log_text="#C9D3EC",
    ),
    "dark": Palette(
        page="#141821",
        surface="#191E2A",
        sidebar="#10131A",
        card="#1D2230",
        input_bg="#171B26",
        rule="#2E3546",
        ink="#E3E7F0",
        muted="#98A2B8",
        accent="#3B4A70",
        accent_hover="#4B5D8C",
        accent_text="#F2F5FA",
        rubric="#D2513B",
        rubric_hover="#E2614C",
        done="#4FB39B",
        done_text="#0F1A17",
        bar_active="#4A6AB5",
        disabled_bg="#2A3144",
        disabled_text="#7B859C",
        log_bg="#0E1119",
        log_text="#B9C5E3",
    ),
}


# Colours for the per-stage traces in the hardware panel: (light, dark).
STAGE_COLORS: dict[str, tuple[str, str]] = {
    "transcribe": ("#3D5A99", "#7E9BE0"),
    "speakers": ("#7A5BA8", "#B39AE0"),
    "visuals": ("#A8741F", "#E0B15A"),
}
