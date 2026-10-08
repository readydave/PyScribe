from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6.QtWidgets import QApplication

from services import config_service
from services.config_service import AppConfig
from services.model_service import RuntimeInfo
from ui_qt.main_window import MainWindow


class ConfigDefaultsTests(unittest.TestCase):
    def _load(self, payload: dict) -> AppConfig:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return config_service.load_config(path)

    def test_new_fields_have_safe_defaults(self) -> None:
        cfg = self._load({})
        self.assertIsNone(cfg.default_model)
        self.assertEqual(cfg.default_input_mode, "file")
        self.assertEqual(cfg.default_hotwords, "")
        self.assertFalse(cfg.default_batched)
        self.assertFalse(cfg.sidebar_collapsed)
        self.assertIsNone(cfg.window_geometry)

    def test_bad_values_are_sanitised(self) -> None:
        cfg = self._load({"default_input_mode": "bogus", "default_hotwords": "x" * 900, "default_batched": "yes",
                          "default_model": 5, "sidebar_collapsed": "maybe"})
        self.assertEqual(cfg.default_input_mode, "file")
        self.assertEqual(len(cfg.default_hotwords), 500)
        self.assertIsNone(cfg.default_model)

    def test_values_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            config_service.save_config(
                AppConfig(default_model="owner/repo", default_input_mode="live", default_hotwords="Kubernetes",
                          default_batched=True, sidebar_collapsed=True, window_geometry="AAAA"),
                path,
            )
            cfg = config_service.load_config(path)
        self.assertEqual((cfg.default_model, cfg.default_input_mode, cfg.default_hotwords), ("owner/repo", "live", "Kubernetes"))
        self.assertTrue(cfg.default_batched and cfg.sidebar_collapsed)
        self.assertEqual(cfg.window_geometry, "AAAA")


class StartupDefaultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls._app = QApplication.instance() or QApplication([])

    def _build(self, config: AppConfig | None = None) -> tuple[MainWindow, list[AppConfig]]:
        saved: list[AppConfig] = []
        runtime = RuntimeInfo(device="cpu", compute_type="int8", gpu_name="N/A", vram_gb=0.0, cpu_count=8)
        patches = [
            patch("ui_qt.main_window.detect_runtime", return_value=runtime),
            patch("ui_qt.main_window.load_config", return_value=config or AppConfig()),
            patch("ui_qt.main_window.save_config", side_effect=lambda cfg: saved.append(AppConfig(**cfg.__dict__))),
            patch("ui_qt.main_window.list_live_audio_inputs", return_value=[]),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        window = MainWindow()
        window.show()
        QApplication.processEvents()
        self.addCleanup(window.close)
        return window, saved

    def test_defaults_are_applied_at_startup(self) -> None:
        cfg = AppConfig(default_model="owner/custom-repo", default_hotwords="Kubernetes, Okafor",
                        default_batched=True, sidebar_collapsed=True, default_input_mode="live")
        win, _ = self._build(cfg)
        self.assertEqual(win.model_combo.currentText(), "owner/custom-repo")
        self.assertEqual(win.hotwords_input.text(), "Kubernetes, Okafor")
        self.assertTrue(win.batched_checkbox.isChecked())
        self.assertTrue(win._sidebar_collapsed)
        self.assertEqual(win.input_mode_combo.currentIndex(), 1)
        self.assertEqual(win.default_model_input.text(), "owner/custom-repo")

    def test_without_a_default_model_the_last_used_model_is_kept(self) -> None:
        win, _ = self._build(AppConfig(last_model="small"))
        self.assertEqual(win.model_combo.currentText(), "small")
        self.assertEqual(win.default_model_input.text(), "")

    def test_settings_page_edits_are_saved(self) -> None:
        win, saved = self._build()
        win.default_model_input.setText(" large-v3 ")
        win._on_default_model_edited()
        self.assertEqual(saved[-1].default_model, "large-v3")
        win.default_model_input.setText("")
        win._on_default_model_edited()
        self.assertIsNone(saved[-1].default_model)
        win.default_input_combo.setCurrentIndex(1)
        self.assertEqual(saved[-1].default_input_mode, "live")
        win.default_hotwords_input.setText("PyScribe")
        win._on_default_hotwords_edited()
        self.assertEqual(saved[-1].default_hotwords, "PyScribe")
        win.default_batched_checkbox.setChecked(True)
        self.assertTrue(saved[-1].default_batched)

    def test_invalid_open_folder_is_reverted(self) -> None:
        win, _ = self._build()
        original = win.last_open_dir
        win.default_path_input.setText("/definitely/not/a/folder")
        win._on_default_path_edited()
        self.assertEqual(win.last_open_dir, original)
        self.assertEqual(win.default_path_input.text(), original)

    def test_use_current_settings_as_defaults(self) -> None:
        win, saved = self._build()
        win.model_combo.setEditText("small")
        win.hotwords_input.setText("Okafor")
        win.batched_checkbox.setChecked(True)
        win._use_current_as_defaults()
        self.assertEqual(
            (saved[-1].default_model, saved[-1].default_hotwords, saved[-1].default_batched),
            ("small", "Okafor", True),
        )
        self.assertEqual(win.default_model_input.text(), "small")

    def test_sidebar_toggle_and_window_size_persist(self) -> None:
        win, saved = self._build()
        win._toggle_sidebar_collapsed()
        self.assertTrue(saved[-1].sidebar_collapsed)
        # The offscreen screen is 800x800, so pick a height the default fit (656) would not produce.
        win.resize(900, 700)
        QApplication.processEvents()
        win.close()
        geometry = saved[-1].window_geometry
        self.assertTrue(geometry)
        restored, _ = self._build(AppConfig(window_geometry=geometry))
        self.assertEqual(restored.height(), 700)


if __name__ == "__main__":
    unittest.main()
