"""Colour theme editor: pick a preset, duplicate it, and edit every colour with a live preview."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from services.ui_themes import (
    DEFAULT_THEME_ID,
    MAX_CUSTOM_THEMES,
    MAX_NAME_LENGTH,
    SPEAKER_SLOTS,
    Theme,
    all_themes,
    check_palette,
    derive_palette,
    sanitize_custom_themes,
    theme_from_dict,
    theme_to_dict,
    unique_theme_id,
)
from ui_qt import theme

LOGGER = logging.getLogger(__name__)

MAX_IMPORT_BYTES = 64 * 1024

COLOR_LABELS: tuple[tuple[str, str], ...] = (
    ("page", "Window background"),
    ("card", "Panels and cards"),
    ("ink", "Text"),
    ("muted", "Secondary text"),
    ("rule", "Borders"),
    ("accent", "Buttons and selection"),
    ("primary", "Main action and recording"),
    ("done", "Finished and success"),
    ("stage_transcribe", "Transcribe trace"),
    ("stage_speakers", "Speakers trace"),
    ("stage_visuals", "Visuals trace"),
)


def _text_on(hex_color: str) -> str:
    color = QColor(hex_color)
    return "#000000" if color.lightness() > 140 else "#FFFFFF"


class _Swatch(QPushButton):
    """A button showing a colour; the colour is set programmatically, never parsed from user text."""

    def __init__(self, hex_color: str) -> None:
        super().__init__()
        self.setMinimumWidth(110)
        self.set_color(hex_color)

    def set_color(self, hex_color: str) -> None:
        self._hex = hex_color
        self.setText(hex_color)
        # hex_color is always validated (#RRGGBB) before it reaches a swatch.
        self.setStyleSheet(
            f"QPushButton {{ background: {hex_color}; color: {_text_on(hex_color)}; "
            f"border: 1px solid #808080; padding: 6px 10px; }}"
        )

    @property
    def hex(self) -> str:
        return self._hex


class ThemeEditorDialog(QDialog):
    """Edits custom themes. Presets are read-only: editing one first makes a copy."""

    previewApplied = Signal()

    def __init__(
        self,
        parent: QWidget | None,
        custom_themes: list[dict],
        selected_id: str,
        theme_mode: str,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Colour themes")
        self.resize(860, 640)
        self._theme_mode = theme_mode
        self._themes: list[dict] = sanitize_custom_themes(custom_themes)
        known = {t.id for t in all_themes(self._themes)}
        self._selected_id = selected_id if selected_id in known else DEFAULT_THEME_ID
        self._swatches: dict[tuple[str, str], _Swatch] = {}
        self._speaker_swatches: dict[tuple[int, int], _Swatch] = {}
        self._building = False

        root = QHBoxLayout(self)
        root.addLayout(self._build_list_column(), 0)
        root.addLayout(self._build_editor_column(), 1)
        self._refresh_list()
        self._load_selected()

    # --- results ---------------------------------------------------------------------------

    @property
    def selected_theme_id(self) -> str:
        return self._selected_id

    @property
    def custom_themes(self) -> list[dict]:
        return list(self._themes)

    # --- layout ----------------------------------------------------------------------------

    def _build_list_column(self) -> QVBoxLayout:
        column = QVBoxLayout()
        column.addWidget(QLabel("Themes"))
        self.theme_list = QListWidget()
        self.theme_list.setMinimumWidth(210)
        self.theme_list.currentItemChanged.connect(self._on_list_changed)
        column.addWidget(self.theme_list, 1)
        self.duplicate_btn = QPushButton("Duplicate")
        self.rename_btn = QPushButton("Rename...")
        self.delete_btn = QPushButton("Delete")
        self.import_btn = QPushButton("Import...")
        self.export_btn = QPushButton("Export...")
        self.duplicate_btn.clicked.connect(self._duplicate)
        self.rename_btn.clicked.connect(self._rename)
        self.delete_btn.clicked.connect(self._delete)
        self.import_btn.clicked.connect(self._import)
        self.export_btn.clicked.connect(self._export)
        for button in (self.duplicate_btn, self.rename_btn, self.delete_btn, self.import_btn, self.export_btn):
            column.addWidget(button)
        return column

    def _build_editor_column(self) -> QVBoxLayout:
        column = QVBoxLayout()
        self.title_label = QLabel("")
        self.title_label.setObjectName("PageSubtitle")
        column.addWidget(self.title_label)
        self.tabs = QTabWidget()
        for mode, label in (("light", "Light"), ("dark", "Dark")):
            self.tabs.addTab(self._build_mode_tab(mode), label)
        column.addWidget(self.tabs, 1)
        self.notes_label = QLabel("")
        self.notes_label.setObjectName("hint")
        self.notes_label.setWordWrap(True)
        column.addWidget(self.notes_label)
        hint = QLabel(
            "Changes preview right away. Text colours are adjusted automatically when they would be hard to read. "
            "The preview follows View > Theme (System, Light or Dark)."
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        column.addWidget(hint)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        column.addWidget(buttons)
        return column

    def _build_mode_tab(self, mode: str) -> QWidget:
        tab = QWidget()
        grid = QGridLayout(tab)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        for row, (name, label) in enumerate(COLOR_LABELS):
            swatch = _Swatch("#000000")
            swatch.clicked.connect(lambda _=False, m=mode, n=name: self._edit_core(m, n))
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(swatch, row, 1)
            self._swatches[(mode, name)] = swatch
        speakers_row = len(COLOR_LABELS)
        grid.addWidget(QLabel("Speaker labels"), speakers_row, 0, Qt.AlignTop)
        holder = QGridLayout()
        mode_index = 0 if mode == "light" else 1
        for slot in range(SPEAKER_SLOTS):
            swatch = _Swatch("#000000")
            swatch.setMinimumWidth(80)
            swatch.clicked.connect(lambda _=False, s=slot, i=mode_index: self._edit_speaker(s, i))
            holder.addWidget(swatch, slot // 4, slot % 4)
            self._speaker_swatches[(slot, mode_index)] = swatch
        grid.addLayout(holder, speakers_row, 1)
        grid.setRowStretch(speakers_row + 1, 1)
        return tab

    # --- list and selection ----------------------------------------------------------------

    def _refresh_list(self) -> None:
        self._building = True
        self.theme_list.clear()
        for item in all_themes(self._themes):
            entry = QListWidgetItem(item.name + ("" if not item.builtin else "  (preset)"))
            entry.setData(Qt.UserRole, item.id)
            self.theme_list.addItem(entry)
            if item.id == self._selected_id:
                self.theme_list.setCurrentItem(entry)
        self._building = False

    def _current_theme(self) -> Theme:
        for item in all_themes(self._themes):
            if item.id == self._selected_id:
                return item
        return all_themes(self._themes)[0]

    def _on_list_changed(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if self._building or current is None:
            return
        self._selected_id = str(current.data(Qt.UserRole))
        self._load_selected()

    def _load_selected(self) -> None:
        item = self._current_theme()
        for mode in ("light", "dark"):
            colors = item.light if mode == "light" else item.dark
            for name, _label in COLOR_LABELS:
                self._swatches[(mode, name)].set_color(getattr(colors, name))
        for slot, pair in enumerate(item.speakers):
            for index in (0, 1):
                self._speaker_swatches[(slot, index)].set_color(pair[index])
        self.delete_btn.setEnabled(not item.builtin)
        self.rename_btn.setEnabled(not item.builtin)
        self.title_label.setText(
            f"{item.name}" + (" is a preset. Change any colour to save your own copy." if item.builtin else "")
        )
        self._preview()

    # --- editing ---------------------------------------------------------------------------

    def _pick_color(self, initial: str, title: str) -> str | None:
        """Ask for a colour; returns ``#RRGGBB`` or None if cancelled. Overridden in tests."""
        chosen = QColorDialog.getColor(QColor(initial), self, title)
        if not chosen.isValid():
            return None
        return chosen.name().upper()

    def _ensure_editable(self) -> bool:
        """Editing a preset duplicates it first so the preset itself never changes."""
        if not self._current_theme().builtin:
            return True
        return self._duplicate()

    def _store(self, edited: dict) -> None:
        for index, existing in enumerate(self._themes):
            if existing.get("id") == edited["id"]:
                self._themes[index] = edited
                break
        self._themes = sanitize_custom_themes(self._themes)

    def _edit_core(self, mode: str, name: str) -> None:
        current = getattr(self._current_theme().light if mode == "light" else self._current_theme().dark, name)
        label = dict(COLOR_LABELS)[name]
        chosen = self._pick_color(current, f"{label} ({mode})")
        if chosen is None or chosen == current:
            return
        if not self._ensure_editable():
            return
        data = theme_to_dict(self._current_theme())
        data[mode][name] = chosen
        self._store(data)
        self._load_selected()

    def _edit_speaker(self, slot: int, index: int) -> None:
        current = self._current_theme().speakers[slot][index]
        chosen = self._pick_color(current, f"Speaker {slot + 1}")
        if chosen is None or chosen == current:
            return
        if not self._ensure_editable():
            return
        data = theme_to_dict(self._current_theme())
        data["speakers"][slot][index] = chosen
        self._store(data)
        self._load_selected()

    # --- list actions ----------------------------------------------------------------------

    def _duplicate(self) -> bool:
        if len(self._themes) >= MAX_CUSTOM_THEMES:
            QMessageBox.information(self, "Colour themes", f"You can keep up to {MAX_CUSTOM_THEMES} custom themes.")
            return False
        source = self._current_theme()
        name = f"{source.name} copy"[:MAX_NAME_LENGTH]
        data = theme_to_dict(source)
        data["name"] = name
        data["id"] = unique_theme_id(name, [t["id"] for t in self._themes])
        self._themes.append(data)
        self._themes = sanitize_custom_themes(self._themes)
        self._selected_id = data["id"]
        self._refresh_list()
        self._load_selected()
        return True

    def _rename(self) -> None:
        item = self._current_theme()
        if item.builtin:
            return
        text, ok = QInputDialog.getText(self, "Rename theme", "Theme name:", text=item.name)
        text = " ".join(str(text).split())[:MAX_NAME_LENGTH]
        if not ok or not text:
            return
        data = theme_to_dict(item)
        data["name"] = text
        self._store(data)
        self._refresh_list()
        self._load_selected()

    def _delete(self) -> None:
        item = self._current_theme()
        if item.builtin:
            return
        answer = QMessageBox.question(self, "Delete theme", f"Delete the theme \"{item.name}\"?")
        if answer != QMessageBox.Yes:
            return
        self._themes = [t for t in self._themes if t.get("id") != item.id]
        self._selected_id = DEFAULT_THEME_ID
        self._refresh_list()
        self._load_selected()

    def _import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import theme", "", "Theme files (*.json)")
        if path:
            self.import_file(Path(path))

    def import_file(self, path: Path) -> bool:
        """Import one theme from a JSON file; invalid files are rejected with a message."""
        try:
            if path.stat().st_size > MAX_IMPORT_BYTES:
                raise ValueError("The file is too large to be a theme.")
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Import theme", f"Could not read that file: {exc}")
            return False
        imported = theme_from_dict(raw)
        if imported is None:
            QMessageBox.warning(
                self,
                "Import theme",
                "That file isn't a valid PyScribe theme. Colours must be written like #1A2B3C.",
            )
            return False
        if len(self._themes) >= MAX_CUSTOM_THEMES:
            QMessageBox.information(self, "Import theme", f"You can keep up to {MAX_CUSTOM_THEMES} custom themes.")
            return False
        data = theme_to_dict(imported)
        data["id"] = unique_theme_id(imported.name, [t["id"] for t in self._themes])
        self._themes.append(data)
        self._themes = sanitize_custom_themes(self._themes)
        self._selected_id = data["id"]
        self._refresh_list()
        self._load_selected()
        return True

    def _export(self) -> None:
        item = self._current_theme()
        path, _ = QFileDialog.getSaveFileName(self, "Export theme", f"{item.id}.json", "Theme files (*.json)")
        if path:
            self.export_file(Path(path))

    def export_file(self, path: Path) -> bool:
        try:
            path.write_text(json.dumps(theme_to_dict(self._current_theme()), indent=2), encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(self, "Export theme", f"Could not save the theme: {exc}")
            return False
        return True

    # --- preview and contrast notes --------------------------------------------------------

    def _preview(self) -> None:
        app = QApplication.instance()
        if app is not None:
            theme.apply_theme(app, self._theme_mode, self._selected_id, self._themes)
            self.previewApplied.emit()
        self.notes_label.setText(self.contrast_notes())

    def contrast_notes(self) -> str:
        """Describe text colours that were adjusted for readability, and anything still too faint."""
        item = self._current_theme()
        lines: list[str] = []
        for mode, colors in (("light", item.light), ("dark", item.dark)):
            palette = derive_palette(colors, mode)
            for name, label in (("ink", "Text"), ("muted", "Secondary text")):
                if getattr(palette, name).upper() != getattr(colors, name).upper():
                    lines.append(
                        f"{mode.capitalize()}: {label} {getattr(colors, name)} was adjusted to "
                        f"{getattr(palette, name)} so it stays readable."
                    )
            for failure in check_palette(palette):
                lines.append(f"{mode.capitalize()}: still hard to read ({failure}). Try a different background colour.")
        return "\n".join(lines)
