"""Tests for prompt template loading and additive config fields."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from services.config_service import AppConfig, load_config, save_config
from services.prompt_template_service import get_default_prompt_template_id, get_prompt_template, load_prompt_templates


class PromptTemplateServiceTests(unittest.TestCase):
    def test_load_prompt_templates_returns_builtins(self) -> None:
        templates, default_template_id = load_prompt_templates()
        template_ids = {template.id for template in templates}
        self.assertIn("meeting-summary", template_ids)
        self.assertIn("action-items", template_ids)
        self.assertEqual(default_template_id, "meeting-summary")

    def test_get_prompt_template_by_id(self) -> None:
        template = get_prompt_template("decision-log")
        self.assertIsNotNone(template)
        assert template is not None
        self.assertEqual(template.name, "Decision Log")
        self.assertEqual(template.output_format, "markdown")

    def test_get_default_prompt_template_id(self) -> None:
        self.assertEqual(get_default_prompt_template_id(), "meeting-summary")


class ConfigServiceAdditiveFieldsTests(unittest.TestCase):
    def test_load_config_defaults_additive_llm_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.json"
            path.write_text("{}", encoding="utf-8")
            cfg = load_config(path)

        self.assertEqual(cfg.llm_default_template_id, "meeting-summary")
        self.assertTrue(cfg.llm_include_user_notes_default)
        self.assertTrue(cfg.llm_include_images_default)
        self.assertTrue(cfg.llm_ocr_fallback_for_images_default)
        self.assertTrue(cfg.llm_payload_preview_required)
        self.assertFalse(cfg.llm_allow_remote_lan)
        self.assertEqual(cfg.llm_profiles, [])

    def test_save_and_reload_additive_llm_fields(self) -> None:
        cfg = AppConfig(
            llm_profiles=[{"name": "Local", "provider": "ollama"}],
            llm_default_profile="Local",
            llm_default_template_id="incident-summary",
            llm_include_user_notes_default=False,
            llm_include_images_default=False,
            llm_ocr_fallback_for_images_default=True,
            llm_payload_preview_required=False,
            llm_allow_remote_lan=True,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.json"
            save_config(cfg, path)
            reloaded = load_config(path)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(reloaded.llm_default_profile, "Local")
        self.assertEqual(reloaded.llm_default_template_id, "incident-summary")
        self.assertFalse(reloaded.llm_include_user_notes_default)
        self.assertFalse(reloaded.llm_include_images_default)
        self.assertTrue(reloaded.llm_ocr_fallback_for_images_default)
        self.assertFalse(reloaded.llm_payload_preview_required)
        self.assertTrue(reloaded.llm_allow_remote_lan)
        self.assertEqual(len(reloaded.llm_profiles), 1)
        self.assertIn("llm_profiles", payload)


if __name__ == "__main__":
    unittest.main()
