from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from services import config_service
from services.ui_themes import (
    AA,
    CORE_FIELDS,
    DEFAULT_SPEAKERS,
    MAX_CUSTOM_THEMES,
    PRESETS,
    ThemeColors,
    all_themes,
    check_palette,
    contrast,
    derive_palette,
    ensure_contrast,
    is_hex,
    resolve,
    sanitize_custom_themes,
    theme_from_dict,
    theme_to_dict,
    unique_theme_id,
)

LIGHT = dict(page="#EEF1F5", card="#FFFFFF", ink="#26304A", muted="#566079", rule="#D3D9E3", accent="#26304A",
             primary="#C2412D", done="#286B5B", stage_transcribe="#3D5A99", stage_speakers="#7A5BA8",
             stage_visuals="#A8741F")
DARK = dict(page="#141821", card="#1D2230", ink="#E3E7F0", muted="#98A2B8", rule="#2E3546", accent="#3B4A70",
            primary="#C4452F", done="#4FB39B", stage_transcribe="#7E9BE0", stage_speakers="#B39AE0",
            stage_visuals="#E0B15A")


def custom(name: str = "Mine", **overrides: object) -> dict:
    light = {**LIGHT, **overrides.get("light", {})}  # type: ignore[arg-type]
    dark = {**DARK, **overrides.get("dark", {})}  # type: ignore[arg-type]
    return {"name": name, "light": light, "dark": dark, "speakers": [list(p) for p in DEFAULT_SPEAKERS]}


class PresetTests(unittest.TestCase):
    def test_every_preset_is_readable_in_both_modes(self) -> None:
        for theme in PRESETS:
            for mode in ("light", "dark"):
                palette = resolve(theme.id, None, mode).palette
                self.assertEqual(check_palette(palette), [], f"{theme.id} {mode}")

    def test_preset_speaker_colours_meet_aa(self) -> None:
        for theme in PRESETS:
            for mode in ("light", "dark"):
                resolved = resolve(theme.id, None, mode)
                for color in resolved.speaker_colors:
                    for bg in (resolved.palette.input_bg, resolved.palette.card):
                        self.assertGreaterEqual(contrast(color, bg), AA, f"{theme.id} {mode} {color}")

    def test_unknown_theme_falls_back_to_iron_gall(self) -> None:
        self.assertEqual(resolve("nope", None, "light").id, "iron-gall")
        self.assertEqual(resolve(None, None, "dark").id, "iron-gall")

    def test_presets_have_distinct_looks(self) -> None:
        pages = {resolve(t.id, None, "light").palette.page for t in PRESETS}
        self.assertEqual(len(pages), len(PRESETS))


class ContrastTests(unittest.TestCase):
    def test_ensure_contrast_lifts_low_contrast_text(self) -> None:
        fixed = ensure_contrast("#CCCCCC", "#FFFFFF")
        self.assertGreaterEqual(contrast(fixed, "#FFFFFF"), AA)
        fixed_dark = ensure_contrast("#333333", "#101010")
        self.assertGreaterEqual(contrast(fixed_dark, "#101010"), AA)

    def test_ensure_contrast_leaves_passing_colours_alone(self) -> None:
        self.assertEqual(ensure_contrast("#000000", "#FFFFFF"), "#000000")

    def test_ensure_contrast_handles_several_backgrounds(self) -> None:
        fixed = ensure_contrast("#999999", ["#FFFFFF", "#EEEEEE"])
        self.assertGreaterEqual(min(contrast(fixed, "#FFFFFF"), contrast(fixed, "#EEEEEE")), AA)

    def test_random_cores_always_derive_valid_readable_palettes(self) -> None:
        rng = random.Random(7)
        for _ in range(60):
            values = {name: f"#{rng.randrange(0x1000000):06X}" for name in CORE_FIELDS}
            for mode in ("light", "dark"):
                palette = derive_palette(ThemeColors(**values), mode)
                for value in vars(palette).values():
                    self.assertTrue(is_hex(value), value)
                # Backgrounds come from the user. When one text colour can pass on all of them,
                # the derived ink and muted colours must too.
                backgrounds = [palette.page, palette.card, palette.surface, palette.input_bg, palette.sidebar]
                feasible = any(
                    all(contrast(text, bg) >= AA for bg in backgrounds) for text in ("#FFFFFF", "#000000")
                )
                if feasible:
                    for fg in ("ink", "muted"):
                        for bg in backgrounds:
                            self.assertGreaterEqual(contrast(getattr(palette, fg), bg), AA, f"{mode} {fg} on {bg}")


class SerialisationTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        theme = theme_from_dict(custom("Round Trip"))
        assert theme is not None
        again = theme_from_dict(theme_to_dict(theme))
        self.assertEqual(theme, again)
        self.assertEqual(theme.id, "round-trip")

    def test_rejects_unsafe_or_malformed_colours(self) -> None:
        for bad in ("red", "#12345", "#GGGGGG", "url(http://x)", "#FFFFFF; } * { display:none", "#FFFFFFF", 5, None):
            self.assertIsNone(theme_from_dict(custom(light={"ink": bad})), repr(bad))
            self.assertIsNone(theme_from_dict(custom(dark={"page": bad})), repr(bad))
        raw = custom()
        raw["speakers"][0][0] = "expression(alert(1))"
        self.assertIsNone(theme_from_dict(raw))

    def test_rejects_bad_shapes_and_names(self) -> None:
        for bad in (None, [], "x", {}, {"name": ""}, {"name": "ok"}, {**custom(), "speakers": [["#FFFFFF", "#000000"]]}):
            self.assertIsNone(theme_from_dict(bad))

    def test_name_is_trimmed_and_never_a_colour_source(self) -> None:
        theme = theme_from_dict(custom("  " + "x" * 80))
        assert theme is not None
        self.assertLessEqual(len(theme.name), 40)

    def test_sanitize_limits_count_and_drops_preset_ids_and_duplicates(self) -> None:
        many = [custom(f"Theme {i}") for i in range(MAX_CUSTOM_THEMES + 5)]
        self.assertEqual(len(sanitize_custom_themes(many)), MAX_CUSTOM_THEMES)
        clash = sanitize_custom_themes([custom("Ochre"), custom("Mine"), custom("Mine")])
        self.assertEqual([t["id"] for t in clash], ["mine"])
        self.assertEqual(sanitize_custom_themes("nonsense"), [])

    def test_unique_theme_id_avoids_clashes(self) -> None:
        self.assertEqual(unique_theme_id("Mine", ["mine"]), "mine-2")
        self.assertEqual(unique_theme_id("Ochre", []), "ochre-2")

    def test_all_themes_lists_presets_then_custom(self) -> None:
        themes = all_themes([custom("Mine")])
        self.assertEqual([t.id for t in themes][-1], "mine")
        self.assertEqual([t.builtin for t in themes][:4], [True] * 4)


class ConfigTests(unittest.TestCase):
    def _load(self, payload: dict) -> config_service.AppConfig:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return config_service.load_config(path)

    def test_old_config_loads_as_iron_gall(self) -> None:
        cfg = self._load({"theme_mode": "dark"})
        self.assertEqual(cfg.theme_id, "iron-gall")
        self.assertEqual(cfg.custom_themes, [])

    def test_custom_theme_selected_by_id_survives_loading(self) -> None:
        cfg = self._load({"theme_id": "mine", "custom_themes": [custom("Mine")]})
        self.assertEqual(cfg.theme_id, "mine")
        self.assertEqual(len(cfg.custom_themes), 1)

    def test_unknown_or_dropped_theme_id_falls_back(self) -> None:
        bad = custom("Mine", light={"ink": "red"})
        cfg = self._load({"theme_id": "mine", "custom_themes": [bad]})
        self.assertEqual(cfg.theme_id, "iron-gall")
        self.assertEqual(cfg.custom_themes, [])

    def test_save_sanitizes_custom_themes(self) -> None:
        cfg = config_service.AppConfig(custom_themes=[custom("Mine"), custom("Evil", light={"ink": "url(x)"})])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            config_service.save_config(cfg, path)
            saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual([t["id"] for t in saved["custom_themes"]], ["mine"])


if __name__ == "__main__":
    unittest.main()
