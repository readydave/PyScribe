"""Tests for the Listener theme (tokens shared with the Qt app) and its inlined font."""

from __future__ import annotations

import unittest

import pytest

pytest.importorskip("gradio")

import app  # noqa: E402
from services.ui_tokens import FONT_FAMILY, PALETTES  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
