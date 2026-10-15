"""Tests for the Listener theme (tokens shared with the Qt app) and its inlined font."""

from __future__ import annotations

import unittest

import pytest

pytest.importorskip("gradio")

import app  # noqa: E402
from services.ui_themes import DEFAULT_SPEAKERS, resolve  # noqa: E402
from services.ui_tokens import FONT_FAMILY, PALETTES  # noqa: E402

LIGHT = dict(page="#ABCDEF", card="#FFFFFF", ink="#26304A", muted="#566079", rule="#D3D9E3", accent="#26304A",
             primary="#C2412D", done="#286B5B", stage_transcribe="#3D5A99", stage_speakers="#7A5BA8",
             stage_visuals="#A8741F")
DARK = dict(LIGHT, page="#141821", card="#1D2230", ink="#E3E7F0", muted="#98A2B8", rule="#2E3546")


class ListenerThemeTests(unittest.TestCase):
    def test_theme_builds_from_shared_palette(self) -> None:
        theme = app.build_theme()
        self.assertEqual(theme.button_primary_background_fill, PALETTES["light"].rubric)
        self.assertEqual(theme.button_primary_background_fill_dark, PALETTES["dark"].rubric)
        self.assertEqual(theme.body_background_fill_dark, PALETTES["dark"].page)

    def test_font_is_inlined_without_leaking_filesystem_paths(self) -> None:
        css = app.CUSTOM_CSS
        self.assertIn(f'font-family: "{FONT_FAMILY}"', css)
        self.assertIn("data:font/ttf;base64,", css)
        self.assertNotIn("/gradio_api/file=", css)
        self.assertNotIn(str(app.FONT_DIR), css)

    def test_progress_uses_calm_colours_not_percentage_ramp(self) -> None:
        css = app._progress_css()
        for cls in ("red", "orange", "yellow", "blue"):
            self.assertIn(f"html.pyscribe-prog-{cls} progress", css)
        self.assertIn("var(--pyscribe-bar-done)", css)
        for legacy in ("#dc2626", "#f97316", "#facc15", "#2563eb", "#16a34a"):
            self.assertNotIn(legacy, css)

    def test_chosen_preset_reaches_theme_and_css(self) -> None:
        light, dark = resolve("ochre", None, "light").palette, resolve("ochre", None, "dark").palette
        theme = app.build_theme("ochre")
        self.assertEqual(theme.body_background_fill, light.page)
        self.assertEqual(theme.body_background_fill_dark, dark.page)
        self.assertEqual(theme.button_primary_text_color, light.rubric_text)
        css = app.build_css("ochre")
        self.assertIn(f"--pyscribe-bar-done: {light.done}", css)
        self.assertIn(f"--pyscribe-bar-active: {dark.bar_active}", css)

    def test_custom_theme_is_used_and_unsafe_ones_are_ignored(self) -> None:
        custom = {"name": "Mine", "light": dict(LIGHT), "dark": dict(DARK),
                  "speakers": [list(p) for p in DEFAULT_SPEAKERS]}
        self.assertEqual(app.build_theme("mine", [custom]).body_background_fill, LIGHT["page"])
        evil = {**custom, "light": {**LIGHT, "page": "#FFF; } body { display:none"}}
        css = app.build_css("mine", [evil])
        self.assertNotIn("display:none", css)
        self.assertEqual(app.build_theme("mine", [evil]).body_background_fill, resolve(None, None, "light").palette.page)

    def test_default_css_constant_matches_default_theme(self) -> None:
        self.assertEqual(app.CUSTOM_CSS, app.build_css())


if __name__ == "__main__":
    unittest.main()
