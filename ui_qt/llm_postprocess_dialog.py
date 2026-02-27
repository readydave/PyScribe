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
    get_enabled_llm_profiles,
    get_prompt_template,
    load_prompt_templates,
    run_llm_postprocess,
    test_connection,
)


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
        self.resize(980, 700)

        self._profiles: list[LLMConnectionProfile] = get_enabled_llm_profiles(config.llm_profiles)
        self._templates, self._default_template_id = load_prompt_templates()
        self._current_transcript_text = (current_transcript_text or "").strip()
        self._current_ocr_text = (current_ocr_text or "").strip()
        self._loaded_transcript_path: str | None = None
        self._loaded_transcript_text: str = ""
        self._loaded_ocr_path: str | None = None
        self._loaded_ocr_text: str = ""
        self._is_transcription_running = is_transcription_running
        self._prefer_loaded_transcript = bool(prefer_loaded_transcript)
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
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.refresh_btn = QPushButton("Refresh Connection + Models")
        self.refresh_btn.clicked.connect(self._on_refresh_connection)

        profile_row = QHBoxLayout()
        profile_row.addWidget(self.profile_combo, 1)
        profile_row.addWidget(self.refresh_btn)

        config_form.addRow("Profile", self._wrap_layout(profile_row))
        config_form.addRow("Template", self.template_combo)
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

        self.notes_box = QTextEdit()
        self.notes_box.setPlaceholderText("Additional notes/context to include in post-processing (optional).")
        self.notes_box.setMinimumHeight(100)
        root.addWidget(QLabel("Additional Notes"))
        root.addWidget(self.notes_box)

        self.connection_status = QLabel("Connection test not run yet.")
        self.connection_status.setWordWrap(True)
        root.addWidget(self.connection_status)

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
            self.template_combo.addItem(template.name, template.id)
        if default_template_id:
            for idx in range(self.template_combo.count()):
                if self.template_combo.itemData(idx) == default_template_id:
                    self.template_combo.setCurrentIndex(idx)
                    return
        if self.template_combo.count() > 0:
            self.template_combo.setCurrentIndex(0)

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

    def _selected_template_id(self) -> str | None:
        value = self.template_combo.currentData()
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None

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

        transcript_text, ocr_text = self._resolve_input_context()
        if not transcript_text:
            QMessageBox.warning(
                self,
                "Transcript required",
                "No transcript text is available. Use the current transcript or load a transcript file.",
            )
            return
        template_id = self._selected_template_id()
        if not template_id:
            QMessageBox.warning(self, "Template required", "Choose a prompt template.")
            return
        template = get_prompt_template(template_id)
        if template is None:
            QMessageBox.warning(self, "Template unavailable", f"Template '{template_id}' could not be loaded.")
            return

        selected_model = (self.model_combo.currentText() or "").strip() or None
        notes_text = (self.notes_box.toPlainText() or "").strip()
        req = LLMPostprocessRequest(
            transcript_text=transcript_text,
            ocr_text=ocr_text,
            notes_text=notes_text,
            selected_model=selected_model,
        )
        self.setCursor(Qt.WaitCursor)
        try:
            result = run_llm_postprocess(profile, template, req)
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
        path, _ = QFileDialog.getSaveFileName(self, "Save Post-Processed Output", suggested, "Markdown Files (*.md);;Text Files (*.txt)")
        if not path:
            return
        try:
            Path(path).write_text(text, encoding="utf-8")
        except Exception as exc:
            QMessageBox.critical(self, "Save output failed", str(exc))
            return
        self.connection_status.setText(f"Saved output: {os.path.basename(path)}")
