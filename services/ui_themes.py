"""Colour themes shared by the Qt app and the Gradio listener (no Qt imports).

A theme stores a small *core* of colours per light/dark mode. ``derive_palette`` fills in the
remaining tokens and nudges text-role colours until they meet WCAG AA, so user-edited themes stay
readable. Every colour that can reach a stylesheet is validated as ``#RRGGBB``.
"""

from __future__ import annotations

import colorsys
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields

AA = 4.5
MAX_CUSTOM_THEMES = 20
MAX_NAME_LENGTH = 40
DEFAULT_THEME_ID = "iron-gall"
SPEAKER_SLOTS = 8

_HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class Palette:
    """Every colour token the UIs use. Rubric (the primary colour) marks the main action and live recording."""

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
    rubric_text: str
    failed_text: str
    done: str
    done_text: str
    bar_active: str
    disabled_bg: str
    disabled_text: str
    log_bg: str
    log_text: str


@dataclass(frozen=True)
class ThemeColors:
    """The editable core for one mode."""

    page: str
    card: str
    ink: str
    muted: str
    rule: str
    accent: str
    primary: str
    done: str
    stage_transcribe: str
    stage_speakers: str
    stage_visuals: str


CORE_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(ThemeColors))

# Speaker label colours (light, dark), cycled by speaker number.
DEFAULT_SPEAKERS: tuple[tuple[str, str], ...] = (
    ("#3D5A99", "#7E9BE0"),
    ("#B3402A", "#F0806C"),
    ("#2B7A5F", "#4FB39B"),
    ("#7A4FA8", "#B39AE0"),
    ("#8A590C", "#E0B15A"),
    ("#1B6F80", "#5CC0D6"),
    ("#A8326B", "#E07AA8"),
    ("#5C6B1E", "#A9C25A"),
)


@dataclass(frozen=True)
class Theme:
    id: str
    name: str
    light: ThemeColors
    dark: ThemeColors
    speakers: tuple[tuple[str, str], ...] = DEFAULT_SPEAKERS
    builtin: bool = False


def _colors(*values: str) -> ThemeColors:
    return ThemeColors(*values)


PRESETS: tuple[Theme, ...] = (
    Theme(
        id="iron-gall",
        name="Iron-gall",
        builtin=True,
        light=_colors("#EEF1F5", "#FFFFFF", "#26304A", "#566079", "#D3D9E3", "#26304A", "#C2412D", "#286B5B",
                      "#3D5A99", "#7A5BA8", "#A8741F"),
        dark=_colors("#141821", "#1D2230", "#E3E7F0", "#98A2B8", "#2E3546", "#3B4A70", "#C4452F", "#4FB39B",
                     "#7E9BE0", "#B39AE0", "#E0B15A"),
    ),
    Theme(
        id="verdigris",
        name="Verdigris",
        builtin=True,
        light=_colors("#E8F0EE", "#FFFFFF", "#1B3632", "#4A6660", "#CBDAD6", "#1F5F55", "#B5432E", "#2C6E4A",
                      "#2F6F8F", "#6B5BA8", "#A8741F"),
        dark=_colors("#101A19", "#172423", "#DDEBE8", "#94ADA8", "#2A3D3A", "#2E6B62", "#C4452F", "#55B58E",
                     "#6FB3D4", "#B39AE0", "#E0B15A"),
    ),
    Theme(
        id="ochre",
        name="Ochre",
        builtin=True,
        light=_colors("#F3EDE0", "#FFFCF5", "#3A2D1E", "#66573F", "#DDD2BC", "#7A5120", "#A93C2A", "#4F6B26",
                      "#2F5E8A", "#7A4FA8", "#A8741F"),
        dark=_colors("#1B1610", "#251E16", "#EFE4D0", "#B3A185", "#3C3224", "#8A5E2A", "#C4452F", "#8FB35A",
                     "#7FAFD9", "#B79AD9", "#E0B15A"),
    ),
    Theme(
        id="graphite",
        name="Graphite",
        builtin=True,
        light=_colors("#EFEFF1", "#FFFFFF", "#202226", "#565A63", "#D6D7DB", "#3B4252", "#2457C5", "#2F7D4F",
                      "#2457C5", "#7A5BA8", "#A8741F"),
        dark=_colors("#17181B", "#202226", "#E6E7EA", "#9A9DA6", "#33353B", "#4A5266", "#2F63D6", "#5CBF84",
                     "#7E9BE0", "#B39AE0", "#E0B15A"),
    ),
)
PRESET_BY_ID: dict[str, Theme] = {theme.id: theme for theme in PRESETS}


# --- colour maths ---------------------------------------------------------------------------


def is_hex(value: object) -> bool:
    return isinstance(value, str) and bool(_HEX_RE.match(value))


def _rgb(hex_color: str) -> tuple[float, float, float]:
    return tuple(int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))  # type: ignore[return-value]


def _hex(rgb: Iterable[float]) -> str:
    r, g, b = (max(0, min(255, round(c * 255))) for c in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"


def luminance(hex_color: str) -> float:
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in _rgb(hex_color)]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a: str, b: str) -> float:
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def mix(a: str, b: str, amount: float) -> str:
    """Blend ``a`` toward ``b`` by ``amount`` (0 = a, 1 = b)."""
    ra, rb = _rgb(a), _rgb(b)
    return _hex(x + (y - x) * amount for x, y in zip(ra, rb))


def ensure_contrast(color: str, against: str | Sequence[str], minimum: float = AA) -> str:
    """Shift ``color``'s lightness (keeping hue and saturation) until it reaches ``minimum`` against every background.

    Returns the best reachable colour if the target is impossible (for example a mid-grey background).
    """
    backgrounds = [against] if isinstance(against, str) else list(against)

    def worst(candidate: str) -> float:
        return min(contrast(candidate, bg) for bg in backgrounds)

    if worst(color) >= minimum:
        return color
    # Search both directions and keep the passing colour closest to the original lightness.
    # If neither direction can reach the target, return the best ratio found.
    h, light0, s = colorsys.rgb_to_hls(*_rgb(color))
    best, best_ratio = color, worst(color)
    passing: list[tuple[float, str]] = []
    for step in (-0.01, 0.01):
        light = light0
        while 0.0 <= light + step <= 1.0:
            light += step
            candidate = _hex(colorsys.hls_to_rgb(h, light, s))
            ratio = worst(candidate)
            if ratio >= minimum:
                passing.append((abs(light - light0), candidate))
                break
            if ratio > best_ratio:
                best, best_ratio = candidate, ratio
    if passing:
        return min(passing)[1]
    for extreme in ("#000000", "#FFFFFF"):
        if worst(extreme) > best_ratio:
            best, best_ratio = extreme, worst(extreme)
    return best


def _best_text(background: str, preferred: Sequence[str]) -> str:
    """First preferred text colour that passes AA on ``background``; otherwise the better of black/white."""
    for candidate in preferred:
        if contrast(candidate, background) >= AA:
            return candidate
    return max(("#FFFFFF", "#000000"), key=lambda c: contrast(c, background))


# --- palette derivation ---------------------------------------------------------------------


def derive_palette(colors: ThemeColors, mode: str) -> Palette:
    """Build the full palette for ``mode`` from a theme's core colours."""
    dark = mode == "dark"
    page, card = colors.page, colors.card
    surface = mix(page, card, 0.5)
    sidebar = mix(page, "#000000", 0.25) if dark else mix(page, colors.ink, 0.04)
    input_bg = mix(page, card, 0.4) if dark else card
    text_backgrounds = [page, card, surface, input_bg, sidebar]

    ink = ensure_contrast(colors.ink, text_backgrounds)
    muted = ensure_contrast(colors.muted, text_backgrounds)
    accent = colors.accent
    accent_hover = mix(accent, "#FFFFFF", 0.14 if dark else 0.12)
    on_accent = ([ink, "#FFFFFF"] if dark else ["#FFFFFF", ink])
    accent_text = _best_text(accent, on_accent)
    if contrast(accent_text, accent_hover) < AA:
        accent_hover = mix(accent, accent_text, 0.0)

    rubric = colors.primary
    rubric_hover = mix(rubric, "#FFFFFF", 0.06) if dark else mix(rubric, "#000000", 0.14)
    rubric_text = _best_text(rubric, ["#FFFFFF"])
    if contrast(rubric_text, rubric_hover) < AA:
        rubric_hover = rubric
    failed_text = ensure_contrast(rubric, [card, input_bg, page])

    done = ensure_contrast(colors.done, [page, card, surface])
    done_text = _best_text(done, ["#FFFFFF", "#0F1A17"])

    bar_active = ensure_contrast(colors.stage_transcribe, ink)
    disabled_bg = colors.rule
    disabled_text = ensure_contrast(muted, disabled_bg)
    log_bg = mix(page, "#000000", 0.45) if dark else mix(ink, "#000000", 0.3)
    log_text = ensure_contrast(mix("#FFFFFF", accent, 0.2), log_bg, 7.0)

    return Palette(
        page=page, surface=surface, sidebar=sidebar, card=card, input_bg=input_bg, rule=colors.rule,
        ink=ink, muted=muted, accent=accent, accent_hover=accent_hover, accent_text=accent_text,
        rubric=rubric, rubric_hover=rubric_hover, rubric_text=rubric_text, failed_text=failed_text,
        done=done, done_text=done_text, bar_active=bar_active, disabled_bg=disabled_bg,
        disabled_text=disabled_text, log_bg=log_bg, log_text=log_text,
    )


# (foreground token, background token) pairs that carry normal-size text.
TEXT_PAIRS: tuple[tuple[str, str], ...] = (
    ("ink", "page"), ("ink", "card"), ("ink", "input_bg"), ("ink", "surface"),
    ("muted", "page"), ("muted", "card"), ("muted", "surface"),
    ("accent_text", "accent"), ("accent_text", "accent_hover"),
    ("done_text", "done"), ("ink", "bar_active"),
    ("disabled_text", "disabled_bg"), ("log_text", "log_bg"),
    ("done", "page"), ("done", "card"), ("failed_text", "card"),
    ("rubric_text", "rubric"), ("rubric_text", "rubric_hover"),
)


def check_palette(palette: Palette) -> list[str]:
    """Human-readable descriptions of text pairs below AA (empty when the palette is readable)."""
    failures = []
    for fg, bg in TEXT_PAIRS:
        ratio = contrast(getattr(palette, fg), getattr(palette, bg))
        if ratio < AA:
            failures.append(f"{fg} on {bg}: {ratio:.1f}:1")
    return failures


# --- resolving a theme for a mode -----------------------------------------------------------


@dataclass(frozen=True)
class ResolvedTheme:
    id: str
    name: str
    mode: str
    palette: Palette
    stage_colors: dict[str, str]
    speaker_colors: tuple[str, ...]


def _speaker_colors(theme: Theme, mode: str, palette: Palette) -> tuple[str, ...]:
    index = 1 if mode == "dark" else 0
    backgrounds = [palette.input_bg, palette.card, palette.page if mode == "light" else palette.surface]
    return tuple(ensure_contrast(pair[index], backgrounds) for pair in theme.speakers)


def resolve_theme(theme: Theme, mode: str) -> ResolvedTheme:
    mode = "dark" if mode == "dark" else "light"
    colors = theme.dark if mode == "dark" else theme.light
    palette = derive_palette(colors, mode)
    stages = {
        "transcribe": colors.stage_transcribe,
        "speakers": colors.stage_speakers,
        "visuals": colors.stage_visuals,
    }
    return ResolvedTheme(theme.id, theme.name, mode, palette, stages, _speaker_colors(theme, mode, palette))


def resolve(theme_id: str | None, custom_themes: Iterable[dict] | None, mode: str) -> ResolvedTheme:
    """Resolve ``theme_id`` (preset or custom) for ``mode``; unknown ids fall back to Iron-gall."""
    for raw in custom_themes or ():
        theme = theme_from_dict(raw)
        if theme is not None and theme.id == theme_id:
            return resolve_theme(theme, mode)
    return resolve_theme(PRESET_BY_ID.get(str(theme_id), PRESET_BY_ID[DEFAULT_THEME_ID]), mode)


def all_themes(custom_themes: Iterable[dict] | None) -> list[Theme]:
    """Presets followed by valid custom themes (custom ids that clash with presets are ignored)."""
    themes = list(PRESETS)
    taken = {t.id for t in themes}
    for raw in custom_themes or ():
        theme = theme_from_dict(raw)
        if theme is not None and theme.id not in taken:
            themes.append(theme)
            taken.add(theme.id)
    return themes


# --- serialisation and validation -----------------------------------------------------------


def slugify(name: str) -> str:
    slug = _SLUG_RE.sub("-", str(name).lower()).strip("-")
    return slug[:MAX_NAME_LENGTH] or "theme"


def _colors_from_dict(raw: object) -> ThemeColors | None:
    if not isinstance(raw, dict):
        return None
    values = [raw.get(name) for name in CORE_FIELDS]
    if not all(is_hex(v) for v in values):
        return None
    return ThemeColors(*[str(v).upper() for v in values])


def theme_from_dict(raw: object) -> Theme | None:
    """Validate and build a custom theme; returns None if anything is malformed."""
    if not isinstance(raw, dict):
        return None
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    name = " ".join(name.split())[:MAX_NAME_LENGTH]
    light, dark = _colors_from_dict(raw.get("light")), _colors_from_dict(raw.get("dark"))
    if light is None or dark is None:
        return None
    speakers_raw = raw.get("speakers", DEFAULT_SPEAKERS)
    speakers: list[tuple[str, str]] = []
    if isinstance(speakers_raw, (list, tuple)) and len(speakers_raw) == SPEAKER_SLOTS:
        for pair in speakers_raw:
            if isinstance(pair, (list, tuple)) and len(pair) == 2 and is_hex(pair[0]) and is_hex(pair[1]):
                speakers.append((str(pair[0]).upper(), str(pair[1]).upper()))
    if len(speakers) != SPEAKER_SLOTS:
        return None
    theme_id = slugify(str(raw.get("id") or name))
    return Theme(id=theme_id, name=name, light=light, dark=dark, speakers=tuple(speakers), builtin=False)


def theme_to_dict(theme: Theme) -> dict:
    return {
        "id": theme.id,
        "name": theme.name,
        "light": {name: getattr(theme.light, name) for name in CORE_FIELDS},
        "dark": {name: getattr(theme.dark, name) for name in CORE_FIELDS},
        "speakers": [list(pair) for pair in theme.speakers],
    }


def sanitize_custom_themes(raw: object) -> list[dict]:
    """Keep at most ``MAX_CUSTOM_THEMES`` valid, uniquely-identified custom themes (as plain dicts)."""
    if not isinstance(raw, (list, tuple)):
        return []
    result: list[dict] = []
    taken = set(PRESET_BY_ID)
    for item in raw:
        theme = theme_from_dict(item)
        if theme is None or theme.id in taken:
            continue
        taken.add(theme.id)
        result.append(theme_to_dict(theme))
        if len(result) >= MAX_CUSTOM_THEMES:
            break
    return result


def unique_theme_id(name: str, existing: Iterable[str]) -> str:
    """A slug for ``name`` that doesn't clash with ``existing`` ids."""
    taken = set(existing) | set(PRESET_BY_ID)
    base = slugify(name)
    candidate, counter = base, 2
    while candidate in taken:
        candidate = f"{base}-{counter}"[:MAX_NAME_LENGTH + 3]
        counter += 1
    return candidate
