from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6.QtWidgets import QApplication, QMessageBox

from services.ui_themes import PRESETS, contrast
from ui_qt import theme
from ui_qt.theme_dialog import ThemeEditorDialog


class ThemeEditorDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.addCleanup(theme.apply_theme, self._app, "system")

    def _dialog(self, custom: list[dict] | None = None, selected: str = "iron-gall") -> ThemeEditorDialog:
        dialog = ThemeEditorDialog(None, custom or [], selected, "light")
        self.addCleanup(dialog.deleteLater)
        return dialog

    def test_lists_presets_first(self) -> None:
        dialog = self._dialog()
        self.assertEqual(dialog.theme_list.count(), len(PRESETS))
        self.assertIn("(preset)", dialog.theme_list.item(0).text())

    def test_selecting_a_preset_previews_it(self) -> None:
        dialog = self._dialog(selected="ochre")
        self.assertEqual(theme.active_theme().id, "ochre")
        self.assertFalse(dialog.delete_btn.isEnabled())
        self.assertFalse(dialog.rename_btn.isEnabled())

    def test_editing_a_preset_makes_a_copy_and_leaves_the_preset_alone(self) -> None:
        dialog = self._dialog(selected="verdigris")
        with patch.object(dialog, "_pick_color", return_value="#123456"):
            dialog._edit_core("light", "accent")
        self.assertEqual(len(dialog.custom_themes), 1)
        copy = dialog.custom_themes[0]
        self.assertEqual(copy["light"]["accent"], "#123456")
        self.assertEqual(copy["name"], "Verdigris copy")
        self.assertEqual(dialog.selected_theme_id, copy["id"])
        self.assertNotEqual(next(p for p in PRESETS if p.id == "verdigris").light.accent, "#123456")
        self.assertEqual(theme.active_theme().id, copy["id"])

    def test_cancelled_colour_pick_changes_nothing(self) -> None:
        dialog = self._dialog()
        with patch.object(dialog, "_pick_color", return_value=None):
            dialog._edit_core("light", "page")
        self.assertEqual(dialog.custom_themes, [])

    def test_hard_to_read_text_colour_is_adjusted_and_reported(self) -> None:
        dialog = self._dialog()
        with patch.object(dialog, "_pick_color", return_value="#F0F0F0"):
            dialog._edit_core("light", "ink")  # near-white text on a light page
        palette = theme.active_palette()
        self.assertGreaterEqual(contrast(palette.ink, palette.page), 4.5)
        self.assertIn("was adjusted", dialog.contrast_notes())

    def test_rename_delete_and_duplicate_rules(self) -> None:
        dialog = self._dialog()
        self.assertTrue(dialog._duplicate())
        copy_id = dialog.selected_theme_id
        with patch("ui_qt.theme_dialog.QInputDialog.getText", return_value=("Night shift", True)):
            dialog._rename()
        self.assertEqual(dialog.custom_themes[0]["name"], "Night shift")
        self.assertEqual(dialog.custom_themes[0]["id"], copy_id)
        with patch("ui_qt.theme_dialog.QMessageBox.question", return_value=QMessageBox.Yes):
            dialog._delete()
        self.assertEqual(dialog.custom_themes, [])
        self.assertEqual(dialog.selected_theme_id, "iron-gall")

    def test_export_then_import_round_trips(self) -> None:
        dialog = self._dialog()
        dialog._duplicate()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "theme.json"
            self.assertTrue(dialog.export_file(path))
            fresh = self._dialog()
            self.assertTrue(fresh.import_file(path))
        self.assertEqual(len(fresh.custom_themes), 1)
        self.assertEqual(fresh.custom_themes[0]["light"], dialog.custom_themes[0]["light"])

    def test_import_rejects_unsafe_files(self) -> None:
        dialog = self._dialog()
        bad = {"name": "Evil", "light": {}, "dark": {}}
        with tempfile.TemporaryDirectory() as tmp, patch("ui_qt.theme_dialog.QMessageBox.warning") as warn:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps(bad), encoding="utf-8")
            self.assertFalse(dialog.import_file(path))
            path.write_text("not json", encoding="utf-8")
            self.assertFalse(dialog.import_file(path))
            path.write_text("x" * (70 * 1024), encoding="utf-8")
            self.assertFalse(dialog.import_file(path))
            self.assertEqual(warn.call_count, 3)
        self.assertEqual(dialog.custom_themes, [])


if __name__ == "__main__":
    unittest.main()
