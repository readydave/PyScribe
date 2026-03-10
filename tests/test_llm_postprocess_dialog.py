"""Focused tests for the Qt LLM post-process dialog."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PySide6.QtWidgets import QApplication

from services.config_service import AppConfig
from ui_qt.llm_postprocess_dialog import LLMPostprocessDialog


class LLMPostprocessDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls._app = QApplication.instance() or QApplication([])

    def _build_dialog(self) -> LLMPostprocessDialog:
        return LLMPostprocessDialog(
            config=AppConfig(),
            current_transcript_text="current transcript",
            current_ocr_text="",
            is_transcription_running=lambda: False,
        )

    def test_default_output_save_path_prefers_loaded_transcript_directory_and_name(self) -> None:
        dialog = self._build_dialog()
        try:
            dialog._loaded_transcript_path = r"C:\Users\dave.kahlbaugh\Documents\meeting_notes.txt"
            dialog.template_combo.setCurrentIndex(0)
            self.assertEqual(
                dialog._default_output_save_path(),
                r"C:\Users\dave.kahlbaugh\Documents\meeting_notes_postprocess_meeting-summary.md",
            )
        finally:
            dialog.close()

    def test_default_output_save_path_falls_back_to_last_save_dir(self) -> None:
        dialog = self._build_dialog()
        try:
            dialog._config.last_save_dir = r"C:\Users\dave.kahlbaugh\Documents\Exports"
            dialog.template_combo.setCurrentIndex(0)
            self.assertEqual(
                dialog._default_output_save_path(),
                r"C:\Users\dave.kahlbaugh\Documents\Exports\postprocess_meeting-summary.md",
            )
        finally:
            dialog.close()

    def test_save_output_updates_last_save_dir(self) -> None:
        dialog = self._build_dialog()
        try:
            dialog._last_output_text = "example output"
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / "meeting_postprocess_meeting-summary.md"
                with patch(
                    "ui_qt.llm_postprocess_dialog.QFileDialog.getSaveFileName",
                    return_value=(str(output_path), "Markdown Files (*.md)"),
                ):
                    dialog._on_save_output()
                self.assertEqual(dialog._config.last_save_dir, str(output_path.resolve().parent))
                self.assertEqual(output_path.read_text(encoding="utf-8"), "example output")
        finally:
            dialog.close()


if __name__ == "__main__":
    unittest.main()
