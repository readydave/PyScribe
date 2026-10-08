"""Guards the shared palette against contrast regressions (WCAG AA, 4.5:1 for normal text)."""

from __future__ import annotations

import unittest

from services.ui_tokens import PALETTES

AA = 4.5

# (foreground token or literal, background token) pairs that carry normal-size text.
TEXT_PAIRS = [
    ("ink", "page"),
    ("ink", "card"),
    ("ink", "input_bg"),
    ("muted", "page"),
    ("muted", "card"),
    ("muted", "surface"),
    ("accent_text", "accent"),
    ("accent_text", "accent_hover"),
    ("done_text", "done"),
    ("ink", "bar_active"),
    ("disabled_text", "disabled_bg"),
    ("log_text", "log_bg"),
    ("done", "page"),
    ("done", "card"),
    ("failed_text", "card"),
    ("#FFFFFF", "rubric"),
    ("#FFFFFF", "rubric_hover"),
]


def _luminance(hex_color: str) -> float:
    value = hex_color.lstrip("#")
    channels = [int(value[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a: str, b: str) -> float:
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


class PaletteContrastTests(unittest.TestCase):
    def test_text_pairs_meet_aa_in_both_themes(self) -> None:
        for mode, palette in PALETTES.items():
            for fg, bg in TEXT_PAIRS:
                fg_hex = fg if fg.startswith("#") else getattr(palette, fg)
                ratio = contrast(fg_hex, getattr(palette, bg))
                self.assertGreaterEqual(ratio, AA, f"{mode}: {fg} on {bg} is {ratio:.2f}")


if __name__ == "__main__":
    unittest.main()
