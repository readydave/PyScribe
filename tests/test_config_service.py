"""Tests for atomic config persistence and corruption quarantine."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from services.config_service import AppConfig, load_config, save_config


class ConfigAtomicSaveTests(unittest.TestCase):
    def test_save_and_reload_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            config = AppConfig(last_model="small", use_diarization=True, max_speakers=3)
            save_config(config, path)
            loaded = load_config(path)
            self.assertEqual(loaded.last_model, "small")
            self.assertTrue(loaded.use_diarization)
            self.assertEqual(loaded.max_speakers, 3)

    def test_save_leaves_no_tmp_file_behind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            save_config(AppConfig(), path)
            leftovers = [p.name for p in Path(tmp).iterdir() if p.name != "config.json"]
            self.assertEqual(leftovers, [])

    def test_save_failure_preserves_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing-dir" / "config.json"
            # Parent directory does not exist: save should log and not raise.
            save_config(AppConfig(last_model="tiny"), path)
            self.assertFalse(path.exists())

    def test_save_preserves_path_prefs_from_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            first = AppConfig()
            first.last_open_dir = "/media/files"
            save_config(first, path)
            save_config(AppConfig(), path)
            loaded = load_config(path)
            self.assertEqual(loaded.last_open_dir, "/media/files")


class ConfigQuarantineTests(unittest.TestCase):
    def test_missing_file_returns_defaults_without_quarantine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            loaded = load_config(path)
            self.assertIsInstance(loaded, AppConfig)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_corrupt_file_is_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text("{ not valid json", encoding="utf-8")
            loaded = load_config(path)
            self.assertIsInstance(loaded, AppConfig)
            self.assertFalse(path.exists())
            bad_files = [p for p in Path(tmp).iterdir() if p.name.startswith("config.json.bad-")]
            self.assertEqual(len(bad_files), 1)
            self.assertEqual(bad_files[0].read_text(encoding="utf-8"), "{ not valid json")

    def test_non_object_json_is_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(["a", "list"]), encoding="utf-8")
            load_config(path)
            self.assertFalse(path.exists())
            bad_files = [p for p in Path(tmp).iterdir() if p.name.startswith("config.json.bad-")]
            self.assertEqual(len(bad_files), 1)

    def test_save_after_quarantine_writes_fresh_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text("garbage", encoding="utf-8")
            load_config(path)
            save_config(AppConfig(last_model="base"), path)
            loaded = load_config(path)
            self.assertEqual(loaded.last_model, "base")

    def test_unknown_keys_in_old_config_are_ignored_without_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps({"last_model": "small", "some_future_key": {"nested": True}}),
                encoding="utf-8",
            )
            loaded = load_config(path)
            self.assertEqual(loaded.last_model, "small")
            # Unknown keys must not trigger quarantine.
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
