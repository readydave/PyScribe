"""Templates: browse, preview and manage the prompt templates used for summaries, action items and notes."""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from services.prompt_template_service import (
    PromptTemplate,
    create_user_prompt_template,
    delete_user_prompt_template,
    get_default_prompt_template_id,
    load_prompt_templates,
    set_user_default_prompt_template,
    update_user_prompt_template,
)
from ui_qt.components import AppButton, Chip
from ui_qt.llm_postprocess_dialog import PromptTemplateEditorDialog

LOGGER = logging.getLogger(__name__)


class TemplatesView(QWidget):
    use_requested = Signal(str)
    connections_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._templates: list[PromptTemplate] = []
        self._default_id = ""
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 16)
        root.setSpacing(12)

        header = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(2)
        title = QLabel("Templates")
        title.setProperty("type", "title")
        subtitle = QLabel("Reusable prompts for summaries, action items, meeting notes and your own outputs.")
        subtitle.setProperty("type", "caption")
        titles.addWidget(title)
        titles.addWidget(subtitle)
        header.addLayout(titles, 1)
        self.connections_button = AppButton("AI connections…", "secondary")
        self.connections_button.clicked.connect(self.connections_requested)
        self.new_button = AppButton("New template…", "primary")
        self.new_button.clicked.connect(self.new_template)
        header.addWidget(self.connections_button)
        header.addWidget(self.new_button)
        root.addLayout(header)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        self.list = QListWidget()
        self.list.setAccessibleName("Templates")
        self.list.currentRowChanged.connect(self._show_current)
        splitter.addWidget(self.list)

        detail = QFrame()
        detail.setProperty("studioCard", True)
        column = QVBoxLayout(detail)
        column.setContentsMargins(16, 14, 16, 14)
        column.setSpacing(8)
        top = QHBoxLayout()
        self.name_label = QLabel("")
        self.name_label.setProperty("type", "heading")
        top.addWidget(self.name_label, 1)
        self.builtin_chip = Chip("Built-in", "neutral")
        self.default_chip = Chip("Default", "accent")
        top.addWidget(self.builtin_chip)
        top.addWidget(self.default_chip)
        column.addLayout(top)
        self.description_label = QLabel("")
        self.description_label.setWordWrap(True)
        self.description_label.setProperty("type", "body")
        self.facts_label = QLabel("")
        self.facts_label.setProperty("type", "caption")
        column.addWidget(self.description_label)
        column.addWidget(self.facts_label)
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setAccessibleName("Template preview")
        column.addWidget(self.preview, 1)
        actions = QHBoxLayout()
        self.use_button = AppButton("Use on a transcript…", "primary")
        self.use_button.clicked.connect(self._use_current)
        self.default_button = AppButton("Set as default", "secondary")
        self.default_button.clicked.connect(self.set_default)
        self.edit_button = AppButton("Edit…", "secondary")
        self.edit_button.clicked.connect(self.edit_template)
        self.delete_button = AppButton("Delete…", "ghost")
        self.delete_button.clicked.connect(self.delete_template)
        for button in (self.use_button, self.default_button, self.edit_button, self.delete_button):
            actions.addWidget(button)
        actions.addStretch(1)
        column.addLayout(actions)
        splitter.addWidget(detail)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([280, 640])
        root.addWidget(splitter, 1)
        self.refresh()

    # --- data ------------------------------------------------------------------------------------

    def refresh(self, select_id: str | None = None) -> None:
        keep = select_id or self.current_id()
        try:
            self._templates, _ = load_prompt_templates(include_disabled=True)
            self._default_id = get_default_prompt_template_id() or ""
        except Exception:
            LOGGER.warning("Could not load prompt templates.", exc_info=True)
            self._templates, self._default_id = [], ""
        self.list.blockSignals(True)
        self.list.clear()
        for template in self._templates:
            label = template.name + ("" if template.enabled else " (disabled)")
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, template.id)
            self.list.addItem(item)
        self.list.blockSignals(False)
        row = next((i for i, t in enumerate(self._templates) if t.id == keep), 0 if self._templates else -1)
        self.list.setCurrentRow(row)
        self._show_current(row)

    def current(self) -> PromptTemplate | None:
        row = self.list.currentRow()
        return self._templates[row] if 0 <= row < len(self._templates) else None

    def current_id(self) -> str:
        template = self.current()
        return template.id if template else ""

    def _show_current(self, _row: int) -> None:
        template = self.current()
        has = template is not None
        self.use_button.setEnabled(has)
        self.default_button.setEnabled(has and not template.built_in)
        self.default_button.setToolTip("Only your own templates can be set as the default." if has and template.built_in else "")
        self.edit_button.setEnabled(has and not template.built_in)
        self.delete_button.setEnabled(has and not template.built_in)
        self.builtin_chip.setVisible(has and template.built_in)
        self.default_chip.setVisible(has and template.id == self._default_id)
        if template is None:
            self.name_label.setText("No templates")
            self.description_label.setText("")
            self.facts_label.setText("")
            self.preview.setPlainText("")
            return
        self.name_label.setText(template.name)
        self.description_label.setText(template.description)
        tags = ", ".join(template.tags)
        self.facts_label.setText(f"{template.output_format.title()} output" + (f"  •  {tags}" if tags else ""))
        self.preview.setPlainText(f"System prompt\n{template.system_prompt}\n\nUser prompt\n{template.user_prompt_scaffold}")

    # --- actions ---------------------------------------------------------------------------------

    def _use_current(self) -> None:
        if self.current() is not None:
            self.use_requested.emit(self.current_id())

    def new_template(self) -> None:
        editor = PromptTemplateEditorDialog(parent=self)
        if editor.exec() != PromptTemplateEditorDialog.Accepted:
            return
        payload = editor.payload()
        try:
            created = create_user_prompt_template(
                name=str(payload["name"]),
                description=str(payload["description"]),
                tags=list(payload["tags"]),  # type: ignore[arg-type]
                output_format=str(payload["output_format"]),
                system_prompt=str(payload["system_prompt"]),
                user_prompt_scaffold=str(payload["user_prompt_scaffold"]),
                enabled=bool(payload["enabled"]),
                template_id=str(payload["template_id"] or ""),
            )
        except Exception as exc:
            QMessageBox.warning(self, "Template create failed", str(exc))
            return
        self.refresh(created.id)

    def edit_template(self) -> None:
        template = self.current()
        if template is None or template.built_in:
            return
        editor = PromptTemplateEditorDialog(template=template, parent=self)
        if editor.exec() != PromptTemplateEditorDialog.Accepted:
            return
        payload = editor.payload()
        try:
            updated = update_user_prompt_template(
                template_id=template.id,
                name=str(payload["name"]),
                description=str(payload["description"]),
                tags=list(payload["tags"]),  # type: ignore[arg-type]
                output_format=str(payload["output_format"]),
                system_prompt=str(payload["system_prompt"]),
                user_prompt_scaffold=str(payload["user_prompt_scaffold"]),
                enabled=bool(payload["enabled"]),
            )
        except Exception as exc:
            QMessageBox.warning(self, "Template update failed", str(exc))
            return
        self.refresh(updated.id)

    def delete_template(self) -> None:
        template = self.current()
        if template is None or template.built_in:
            return
        answer = QMessageBox.question(
            self, "Delete template", f"Delete the template \"{template.name}\"?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if answer != QMessageBox.Yes:
            return
        if not delete_user_prompt_template(template.id):
            QMessageBox.warning(self, "Delete failed", "The template could not be deleted.")
            return
        self.refresh()

    def set_default(self) -> None:
        template = self.current()
        if template is None:
            return
        try:
            changed = set_user_default_prompt_template(template.id)
        except Exception as exc:
            QMessageBox.warning(self, "Could not set default", str(exc))
            return
        if not changed:
            QMessageBox.information(self, "Could not set default", "Only your own templates can be set as the default.")
            return
        self.refresh(template.id)
