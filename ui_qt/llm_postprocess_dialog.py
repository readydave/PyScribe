"""Qt dialog for LLM post-processing of current or saved transcripts."""

from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from services import (
    AppConfig,
    LLMConnectionProfile,
    LLMPostprocessRequest,
    PromptTemplate,
    build_llm_payload_preview,
    create_user_prompt_template,
    delete_user_prompt_template,
    get_enabled_llm_profiles,
    get_prompt_template,
    load_prompt_templates,
    run_llm_postprocess,
    test_connection,
    update_user_prompt_template,
)


class PromptTemplateEditorDialog(QDialog):
    def __init__(self, *, template: PromptTemplate | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._editing = template is not None
        self._template = template
        self.setWindowTitle("Edit Template" if self._editing else "New Template")
        self.resize(760, 640)

        root = QVBoxLayout(self)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.id_input = QLineEdit()
        self.id_input.setPlaceholderText("meeting-summary-custom")
        self.name_input = QLineEdit()
        self.description_box = QTextEdit()
        self.description_box.setMaximumHeight(90)
        self.tags_input = QLineEdit()
        self.tags_input.setPlaceholderText("meetings, summary")
        self.output_format_combo = QComboBox()
        self.output_format_combo.addItems(["markdown", "json"])
        self.enabled_checkbox = QCheckBox("Template enabled")

        form.addRow("Template ID", self.id_input)
        form.addRow("Name", self.name_input)
        form.addRow("Description", self.description_box)
        form.addRow("Tags (comma-separated)", self.tags_input)
        form.addRow("Output format", self.output_format_combo)
        form.addRow("", self.enabled_checkbox)

        root.addLayout(form)
        root.addWidget(QLabel("System Prompt"))
        self.system_prompt_box = QTextEdit()
        self.system_prompt_box.setMinimumHeight(130)
        root.addWidget(self.system_prompt_box)
        root.addWidget(QLabel("User Prompt Scaffold"))
        self.user_prompt_box = QTextEdit()
        self.user_prompt_box.setMinimumHeight(130)
        root.addWidget(self.user_prompt_box)

        actions = QHBoxLayout()
        actions.addStretch(1)
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        actions.addWidget(save_btn)
        actions.addWidget(cancel_btn)
        root.addLayout(actions)

        if template is not None:
            self.id_input.setText(template.id)
            self.id_input.setEnabled(False)
            self.name_input.setText(template.name)
            self.description_box.setPlainText(template.description)
            self.tags_input.setText(", ".join(template.tags))
            self.output_format_combo.setCurrentText(template.output_format)
            self.enabled_checkbox.setChecked(template.enabled)
            self.system_prompt_box.setPlainText(template.system_prompt)
            self.user_prompt_box.setPlainText(template.user_prompt_scaffold)
        else:
            self.enabled_checkbox.setChecked(True)

    def payload(self) -> dict[str, object]:
        raw_tags = [part.strip() for part in (self.tags_input.text() or "").split(",") if part.strip()]
        return {
            "template_id": (self.id_input.text() or "").strip().lower(),
            "name": (self.name_input.text() or "").strip(),
            "description": (self.description_box.toPlainText() or "").strip(),
            "tags": raw_tags,
            "output_format": (self.output_format_combo.currentText() or "markdown").strip().lower(),
            "enabled": self.enabled_checkbox.isChecked(),
            "system_prompt": (self.system_prompt_box.toPlainText() or "").strip(),
            "user_prompt_scaffold": (self.user_prompt_box.toPlainText() or "").strip(),
        }


class LLMPostprocessDialog(QDialog):
    def __init__(
        self,
        *,
        config: AppConfig,
        current_transcript_text: str,
        current_ocr_text: str,
        is_transcription_running: Callable[[], bool],
        prefer_loaded_transcript: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("LLM Post-Process")
        self.resize(980, 760)

        self._config = config
        self._profiles: list[LLMConnectionProfile] = get_enabled_llm_profiles(config.llm_profiles)
        self._templates, self._default_template_id = load_prompt_templates()
        self._template_by_id: dict[str, PromptTemplate] = {template.id: template for template in self._templates}
        self._current_transcript_text = (current_transcript_text or "").strip()
        self._current_ocr_text = (current_ocr_text or "").strip()
        self._loaded_transcript_path: str | None = None
        self._loaded_transcript_text: str = ""
        self._loaded_ocr_path: str | None = None
        self._loaded_ocr_text: str = ""
        self._is_transcription_running = is_transcription_running
        self._prefer_loaded_transcript = bool(prefer_loaded_transcript)
        self._payload_preview_required = bool(config.llm_payload_preview_required)
        self._last_payload_preview: str = ""
        self._last_output_text: str = ""

        self._build_ui()
        self._populate_profiles(default_name=config.llm_default_profile)
        self._populate_templates(default_template_id=config.llm_default_template_id or self._default_template_id)
        if self._prefer_loaded_transcript:
            self.use_current_checkbox.setChecked(False)
        self._refresh_source_labels()

        if not self._profiles:
            self.run_btn.setEnabled(False)
            self.connection_status.setText("No enabled LLM profiles. Configure one in Tools > LLM Connections...")

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(10)

        config_form = QFormLayout()
        config_form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self.profile_combo = QComboBox()
        self.profile_combo.currentIndexChanged.connect(self._on_profile_changed)
        self.template_combo = QComboBox()
        self.template_combo.currentIndexChanged.connect(self._on_template_changed)
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.refresh_btn = QPushButton("Refresh Connection + Models")
        self.refresh_btn.clicked.connect(self._on_refresh_connection)
        self.template_new_btn = QPushButton("New...")
        self.template_new_btn.clicked.connect(self._on_new_template)
        self.template_edit_btn = QPushButton("Edit...")
        self.template_edit_btn.clicked.connect(self._on_edit_template)
        self.template_delete_btn = QPushButton("Delete")
        self.template_delete_btn.clicked.connect(self._on_delete_template)

        profile_row = QHBoxLayout()
        profile_row.addWidget(self.profile_combo, 1)
        profile_row.addWidget(self.refresh_btn)

        template_row = QHBoxLayout()
        template_row.addWidget(self.template_combo, 1)
        template_row.addWidget(self.template_new_btn)
        template_row.addWidget(self.template_edit_btn)
        template_row.addWidget(self.template_delete_btn)

        config_form.addRow("Profile", self._wrap_layout(profile_row))
        config_form.addRow("Template", self._wrap_layout(template_row))
        config_form.addRow("Model", self.model_combo)
        root.addLayout(config_form)

        source_row = QHBoxLayout()
        self.use_current_checkbox = QCheckBox("Use current transcript from main window")
        self.use_current_checkbox.setChecked(bool(self._current_transcript_text))
        self.use_current_checkbox.setEnabled(bool(self._current_transcript_text))
        self.load_transcript_btn = QPushButton("Load Transcript File...")
        self.load_transcript_btn.clicked.connect(self._on_load_transcript)
        self.load_ocr_btn = QPushButton("Load OCR File (optional)...")
        self.load_ocr_btn.clicked.connect(self._on_load_ocr)
        source_row.addWidget(self.use_current_checkbox)
        source_row.addWidget(self.load_transcript_btn)
        source_row.addWidget(self.load_ocr_btn)
        root.addLayout(source_row)

        self.source_label = QLabel("")
        self.ocr_label = QLabel("")
        root.addWidget(self.source_label)
        root.addWidget(self.ocr_label)

        root.addWidget(QLabel("Additional Notes"))
        self.notes_box = QTextEdit()
        self.notes_box.setPlaceholderText("Short notes/instructions to include in post-processing (optional).")
        self.notes_box.setMinimumHeight(80)
        root.addWidget(self.notes_box)

        root.addWidget(QLabel("Pasted Context"))
        self.extra_context_box = QTextEdit()
        self.extra_context_box.setPlaceholderText("Paste additional context text (optional).")
        self.extra_context_box.setMinimumHeight(90)
        root.addWidget(self.extra_context_box)

        self.connection_status = QLabel("Connection test not run yet.")
        self.connection_status.setWordWrap(True)
        root.addWidget(self.connection_status)

        preview_row = QHBoxLayout()
        preview_row.addWidget(QLabel("Payload Preview"))
        preview_row.addStretch(1)
        self.preview_btn = QPushButton("Preview Payload")
        self.preview_btn.clicked.connect(self._on_preview_payload)
        preview_row.addWidget(self.preview_btn)
        root.addLayout(preview_row)

        self.preview_box = QTextEdit()
        self.preview_box.setReadOnly(True)
        self.preview_box.setPlaceholderText("Preview of request payload sent to the LLM.")
        self.preview_box.setMinimumHeight(140)
        root.addWidget(self.preview_box)

        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("LLM Output"))
        output_row.addStretch(1)
        self.copy_btn = QPushButton("Copy Output")
        self.copy_btn.setEnabled(False)
        self.copy_btn.clicked.connect(self._on_copy_output)
        self.save_btn = QPushButton("Save Output...")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._on_save_output)
        output_row.addWidget(self.copy_btn)
        output_row.addWidget(self.save_btn)
        root.addLayout(output_row)

        self.output_box = QTextEdit()
        self.output_box.setReadOnly(True)
        self.output_box.setPlaceholderText("Generated post-processing output appears here.")
        root.addWidget(self.output_box, 1)

        actions = QHBoxLayout()
        actions.addStretch(1)
        self.run_btn = QPushButton("Run Post-Process")
        self.run_btn.clicked.connect(self._on_run_postprocess)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        actions.addWidget(self.run_btn)
        actions.addWidget(close_btn)
        root.addLayout(actions)

    def _wrap_layout(self, layout: QHBoxLayout) -> QWidget:
        wrapper = QWidget()
        wrapper.setLayout(layout)
        return wrapper

    def _populate_profiles(self, *, default_name: str | None) -> None:
        self.profile_combo.clear()
        for profile in self._profiles:
            self.profile_combo.addItem(profile.name)
        if default_name:
            idx = self.profile_combo.findText(default_name)
            if idx >= 0:
                self.profile_combo.setCurrentIndex(idx)
                return
        if self.profile_combo.count() > 0:
            self.profile_combo.setCurrentIndex(0)
        self._on_profile_changed()

    def _populate_templates(self, *, default_template_id: str | None) -> None:
        self.template_combo.clear()
        for template in self._templates:
            suffix = " (built-in)" if template.built_in else ""
            self.template_combo.addItem(f"{template.name}{suffix}", template.id)
        target = (default_template_id or "").strip().lower()
        if target:
            for idx in range(self.template_combo.count()):
                if self.template_combo.itemData(idx) == target:
                    self.template_combo.setCurrentIndex(idx)
                    self._on_template_changed()
                    return
        if self.template_combo.count() > 0:
            self.template_combo.setCurrentIndex(0)
        self._on_template_changed()

    def _selected_template_id(self) -> str | None:
        value = self.template_combo.currentData()
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None

    def _selected_template(self) -> PromptTemplate | None:
        template_id = self._selected_template_id()
        if not template_id:
            return None
        return self._template_by_id.get(template_id)

    def _refresh_templates(self, *, selected_template_id: str | None = None) -> None:
        self._templates, self._default_template_id = load_prompt_templates()
        self._template_by_id = {template.id: template for template in self._templates}
        self._populate_templates(default_template_id=selected_template_id or self._default_template_id)

    def _refresh_source_labels(self) -> None:
        if self._current_transcript_text:
            self.source_label.setText(f"Current transcript available ({len(self._current_transcript_text)} chars).")
        else:
            self.source_label.setText("Current transcript unavailable. Load a transcript file to continue.")
        if self._loaded_transcript_path:
            self.source_label.setText(
                f"Loaded transcript: {self._loaded_transcript_path} ({len(self._loaded_transcript_text)} chars)"
            )
        if self._loaded_ocr_path:
            self.ocr_label.setText(f"Loaded OCR report: {self._loaded_ocr_path} ({len(self._loaded_ocr_text)} chars)")
        elif self._current_ocr_text and self.use_current_checkbox.isChecked():
            self.ocr_label.setText(f"Current OCR context available ({len(self._current_ocr_text)} chars).")
        else:
            self.ocr_label.setText("No OCR context selected.")

    def _selected_profile(self) -> LLMConnectionProfile | None:
        name = self.profile_combo.currentText().strip()
        for profile in self._profiles:
            if profile.name == name:
                return profile
        return None

    def _resolve_input_context(self) -> tuple[str, str]:
        if self.use_current_checkbox.isChecked() and self._current_transcript_text:
            transcript_text = self._current_transcript_text
            ocr_text = self._loaded_ocr_text or self._current_ocr_text
            return transcript_text, ocr_text
        transcript_text = self._loaded_transcript_text.strip()
        if not transcript_text and self._current_transcript_text:
            transcript_text = self._current_transcript_text
        ocr_text = self._loaded_ocr_text.strip() or ""
        return transcript_text, ocr_text

    def _build_request(self, transcript_text: str, ocr_text: str) -> LLMPostprocessRequest:
        return LLMPostprocessRequest(
            transcript_text=transcript_text,
            ocr_text=ocr_text,
            notes_text=(self.notes_box.toPlainText() or "").strip(),
            selected_model=(self.model_combo.currentText() or "").strip() or None,
            extra_context_text=(self.extra_context_box.toPlainText() or "").strip(),
        )

    def _render_payload_preview(self) -> tuple[PromptTemplate | None, LLMPostprocessRequest | None]:
        transcript_text, ocr_text = self._resolve_input_context()
        if not transcript_text:
            QMessageBox.warning(
                self,
                "Transcript required",
                "No transcript text is available. Use the current transcript or load a transcript file.",
            )
            return None, None
        template_id = self._selected_template_id()
        if not template_id:
            QMessageBox.warning(self, "Template required", "Choose a prompt template.")
            return None, None
        template = get_prompt_template(template_id)
        if template is None:
            QMessageBox.warning(self, "Template unavailable", f"Template '{template_id}' could not be loaded.")
            return None, None
        request = self._build_request(transcript_text, ocr_text)
        payload_preview = build_llm_payload_preview(template=template, request=request)
        self._last_payload_preview = payload_preview
        self.preview_box.setPlainText(payload_preview)
        self._config.llm_default_template_id = template.id
        return template, request

    @Slot()
    def _on_profile_changed(self) -> None:
        profile = self._selected_profile()
        current_text = (self.model_combo.currentText() or "").strip()
        self.model_combo.clear()
        if profile is None:
            return
        if profile.default_model:
            self.model_combo.addItem(profile.default_model)
            self.model_combo.setCurrentText(profile.default_model)
            return
        if current_text:
            self.model_combo.setEditText(current_text)

    @Slot()
    def _on_template_changed(self) -> None:
        template = self._selected_template()
        editable = bool(template is not None and not template.built_in)
        self.template_edit_btn.setEnabled(editable)
        self.template_delete_btn.setEnabled(editable)

    @Slot()
    def _on_refresh_connection(self) -> None:
        profile = self._selected_profile()
        if profile is None:
            self.connection_status.setText("No profile selected.")
            return
        self.setCursor(Qt.WaitCursor)
        try:
            result = test_connection(profile)
        finally:
            self.unsetCursor()

        if result.detected_models:
            current_text = (self.model_combo.currentText() or "").strip()
            self.model_combo.clear()
            for model_name in result.detected_models:
                self.model_combo.addItem(model_name)
            preferred = current_text or result.selected_model or profile.default_model
            if preferred:
                idx = self.model_combo.findText(preferred)
                if idx >= 0:
                    self.model_combo.setCurrentIndex(idx)
                else:
                    self.model_combo.setEditText(preferred)
        elif profile.default_model:
            if self.model_combo.findText(profile.default_model) < 0:
                self.model_combo.addItem(profile.default_model)
            self.model_combo.setCurrentText(profile.default_model)

        if result.status == "pass":
            self.connection_status.setText("Connection test: PASS")
            return
        lines = [f"Connection test: FAIL ({result.failure_code})"]
        for stage in result.stages:
            if stage.status == "fail":
                lines.append(f"- {stage.stage}: {stage.detail}")
                for suggestion in stage.suggestions:
                    lines.append(f"  * {suggestion}")
        self.connection_status.setText("\n".join(lines))

    @Slot()
    def _on_load_transcript(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Transcript File", "", "Text Files (*.txt *.md);;All Files (*.*)")
        if not path:
            return
        try:
            text = Path(path).read_text(encoding="utf-8")
        except Exception as exc:
            QMessageBox.critical(self, "Load transcript failed", str(exc))
            return
        self._loaded_transcript_path = path
        self._loaded_transcript_text = text.strip()
        if self._loaded_transcript_text:
            self.use_current_checkbox.setChecked(False)
        self._refresh_source_labels()

    @Slot()
    def _on_load_ocr(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select OCR Context File", "", "Text Files (*.txt *.md);;All Files (*.*)")
        if not path:
            return
        try:
            text = Path(path).read_text(encoding="utf-8")
        except Exception as exc:
            QMessageBox.critical(self, "Load OCR file failed", str(exc))
            return
        self._loaded_ocr_path = path
        self._loaded_ocr_text = text.strip()
        self._refresh_source_labels()

    @Slot()
    def _on_new_template(self) -> None:
        editor = PromptTemplateEditorDialog(parent=self)
        if editor.exec() != QDialog.Accepted:
            return
        payload = editor.payload()
        try:
            template = create_user_prompt_template(
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
        self._refresh_templates(selected_template_id=template.id)
        self.connection_status.setText(f"Created template '{template.id}'.")

    @Slot()
    def _on_edit_template(self) -> None:
        template = self._selected_template()
        if template is None:
            return
        if template.built_in:
            QMessageBox.information(
                self,
                "Built-in template",
                "Built-in templates are read-only. Create a new template to customize prompts.",
            )
            return
        editor = PromptTemplateEditorDialog(template=template, parent=self)
        if editor.exec() != QDialog.Accepted:
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
        self._refresh_templates(selected_template_id=updated.id)
        self.connection_status.setText(f"Updated template '{updated.id}'.")

    @Slot()
    def _on_delete_template(self) -> None:
        template = self._selected_template()
        if template is None or template.built_in:
            return
        answer = QMessageBox.question(
            self,
            "Delete template",
            f"Delete user template '{template.id}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        if not delete_user_prompt_template(template.id):
            QMessageBox.warning(self, "Delete failed", "Template could not be deleted.")
            return
        self._refresh_templates()
        self.connection_status.setText(f"Deleted template '{template.id}'.")

    @Slot()
    def _on_preview_payload(self) -> None:
        template, request = self._render_payload_preview()
        if template is None or request is None:
            return
        self.connection_status.setText(f"Payload preview ready for template '{template.id}'.")

    @Slot()
    def _on_run_postprocess(self) -> None:
        profile = self._selected_profile()
        if profile is None:
            QMessageBox.warning(self, "Profile required", "Choose an enabled LLM profile.")
            return
        transcription_running = self._is_transcription_running()
        if transcription_running:
            if profile.scope == "local":
                QMessageBox.warning(
                    self,
                    "Post-process blocked",
                    (
                        "A local transcription/analysis job is currently running.\n\n"
                        "Local LLM profiles cannot run concurrently because they compete for local compute/GPU. "
                        "Choose a LAN profile or wait for transcription to finish."
                    ),
                )
                return
            if not profile.allow_concurrent_with_local_transcription:
                QMessageBox.warning(
                    self,
                    "Post-process blocked",
                    (
                        "A transcription job is currently running.\n\n"
                        "This profile is not configured for concurrent use with local transcription. "
                        "Enable concurrent mode for this remote profile or wait for transcription to finish."
                    ),
                )
                return

        template, request = self._render_payload_preview()
        if template is None or request is None:
            return
        if self._payload_preview_required:
            answer = QMessageBox.question(
                self,
                "Confirm payload",
                "Payload preview is required. Continue and send this payload to the model?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if answer != QMessageBox.Yes:
                self.connection_status.setText("Post-processing canceled before send.")
                return

        self.setCursor(Qt.WaitCursor)
        try:
            result = run_llm_postprocess(profile, template, request)
        finally:
            self.unsetCursor()

        if result.status != "pass":
            QMessageBox.warning(
                self,
                "LLM post-processing failed",
                f"{result.error_code or 'error'}: {result.error_detail or 'Unknown error'}",
            )
            return
        self._last_output_text = result.output_text
        self.output_box.setPlainText(result.output_text)
        self.copy_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.connection_status.setText(f"Post-processing complete with model '{result.model}'.")

    @Slot()
    def _on_copy_output(self) -> None:
        text = self._last_output_text.strip()
        if not text:
            return
        QApplication.clipboard().setText(text)
        self.connection_status.setText("Output copied to clipboard.")

    @Slot()
    def _on_save_output(self) -> None:
        text = self._last_output_text.strip()
        if not text:
            return
        template_id = self._selected_template_id() or "template"
        suggested = f"postprocess_{template_id}.md"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Post-Processed Output",
            suggested,
            "Markdown Files (*.md);;Text Files (*.txt)",
        )
        if not path:
            return
        try:
            Path(path).write_text(text, encoding="utf-8")
        except Exception as exc:
            QMessageBox.critical(self, "Save output failed", str(exc))
            return
        self.connection_status.setText(f"Saved output: {os.path.basename(path)}")
