"""Tests for the consolidated model catalog."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from services.catalog_service import (
    BASE_MODEL_CHOICES,
    _hf_hub_cache_dir,
    get_model_choices,
)
import models


class ModelCatalogDriftTests(unittest.TestCase):
    def test_every_tier_key_is_a_selectable_choice(self) -> None:
        """models.TIERS metadata and catalog choices must not drift apart."""
        choices = set(get_model_choices())
        missing = sorted(set(models.TIERS) - choices)
        self.assertEqual(missing, [], f"TIERS keys missing from model choices: {missing}")

    def test_curated_list_has_no_duplicates(self) -> None:
        self.assertEqual(len(BASE_MODEL_CHOICES), len(set(BASE_MODEL_CHOICES)))


class ModelCatalogCacheScanTests(unittest.TestCase):
    def test_cache_scan_respects_hf_env_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hub = Path(tmp) / "hub"
            (hub / "models--Systran--faster-whisper-testonly").mkdir(parents=True)
            (hub / "models--other--not-whisper").mkdir(parents=True)
            with patch.dict(os.environ, {"HUGGINGFACE_HUB_CACHE": str(hub)}):
                choices = get_model_choices()
            self.assertIn("Systran/faster-whisper-testonly", choices)
            self.assertNotIn("other/not-whisper", choices)

    def test_hf_home_fallback_used_when_hub_cache_unset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = {"HF_HOME": tmp}
            with patch.dict(os.environ, env):
                os.environ.pop("HUGGINGFACE_HUB_CACHE", None)
                self.assertEqual(_hf_hub_cache_dir(), Path(tmp) / "hub")

    def test_missing_cache_dir_is_harmless(self) -> None:
        with patch.dict(os.environ, {"HUGGINGFACE_HUB_CACHE": "/nonexistent/pyscribe-test"}):
            choices = get_model_choices()
        self.assertEqual(sorted(set(BASE_MODEL_CHOICES)), choices)


if __name__ == "__main__":
    unittest.main()
