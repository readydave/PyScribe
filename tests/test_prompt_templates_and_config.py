"""Tests for prompt template loading and additive config fields."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from services.config_service import AppConfig, load_config, save_config
from services.prompt_template_service import (
    create_user_prompt_template,
    delete_user_prompt_template,
    get_default_prompt_template_id,
    get_prompt_template,
    load_prompt_templates,
    update_user_prompt_template,
)


class PromptTemplateServiceTests(unittest.TestCase):
    def test_load_prompt_templates_returns_builtins(self) -> None:
        templates, default_template_id = load_prompt_templates(include_user_templates=False)
        template_ids = {template.id for template in templates}
        self.assertIn("meeting-summary", template_ids)
        self.assertIn("action-items", template_ids)
        self.assertEqual(default_template_id, "meeting-summary")

    def test_get_prompt_template_by_id(self) -> None:
        template = get_prompt_template("decision-log", include_user_templates=False)
        self.assertIsNotNone(template)
        assert template is not None
        self.assertEqual(template.name, "Decision Log")
        self.assertEqual(template.output_format, "markdown")

    def test_get_default_prompt_template_id(self) -> None:
        self.assertEqual(get_default_prompt_template_id(include_user_templates=False), "meeting-summary")

    def test_user_template_crud(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            user_index = Path(temp_dir) / "index.yaml"
            created = create_user_prompt_template(
                name="Weekly Team Summary",
                description="Custom summary template",
                tags=["weekly", "team"],
                output_format="markdown",
                system_prompt="You are concise.",
                user_prompt_scaffold="Summarize this update.",
                enabled=True,
                user_index_path=user_index,
            )
            self.assertFalse(created.built_in)
            self.assertEqual(created.id, "weekly-team-summary")

            templates, default_template_id = load_prompt_templates(
                include_user_templates=True,
                user_index_path=user_index,
            )
            self.assertIn("weekly-team-summary", {template.id for template in templates})
            self.assertEqual(default_template_id, "weekly-team-summary")

            updated = update_user_prompt_template(
                template_id=created.id,
                name="Weekly Team Summary v2",
                description="Updated",
                tags=["weekly"],
                output_format="json",
                system_prompt="System v2",
                user_prompt_scaffold="Scaffold v2",
                enabled=True,
                user_index_path=user_index,
            )
            self.assertEqual(updated.output_format, "json")
            self.assertGreaterEqual(updated.version, 2)

            removed = delete_user_prompt_template(created.id, user_index_path=user_index)
            self.assertTrue(removed)
            templates_after_delete, _ = load_prompt_templates(
                include_user_templates=True,
                user_index_path=user_index,
            )
            self.assertNotIn("weekly-team-summary", {template.id for template in templates_after_delete})


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
        self.assertFalse(cfg.llm_payload_preview_required)
        self.assertFalse(cfg.llm_allow_remote_lan)
        self.assertEqual(cfg.llm_profiles, [])
        self.assertEqual(cfg.visual_ocr_backend, "auto")

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

    def test_save_config_strips_plaintext_llm_api_keys(self) -> None:
        cfg = AppConfig(
            llm_profiles=[
                {"name": "Plain", "provider": "openai_compatible", "api_key": "super-secret", "api_key_runtime": "runtime"},
                {"name": "EnvRef", "provider": "openai_compatible", "api_key": "env:PYSCRIBE_API_KEY"},
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.json"
            save_config(cfg, path)
            payload = json.loads(path.read_text(encoding="utf-8"))
        profiles = payload.get("llm_profiles", [])
        self.assertEqual(len(profiles), 2)
        self.assertEqual(profiles[0].get("api_key"), "")
        self.assertNotIn("api_key_runtime", profiles[0])
        self.assertEqual(profiles[1].get("api_key"), "env:PYSCRIBE_API_KEY")


if __name__ == "__main__":
    unittest.main()
