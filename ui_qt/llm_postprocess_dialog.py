"""Qt dialog for LLM post-processing of current or saved transcripts."""

from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtGui import QCloseEvent, QDragEnterEvent, QDragLeaveEvent, QDropEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from services import (
    AppConfig,
    LLMConnectionProfile,
    LLMPreparedPayload,
    LLMPostprocessRequest,
    LLMPostprocessResult,
    LLMRunControl,
    PromptTemplate,
    create_user_prompt_template,
    delete_user_prompt_template,
    get_enabled_llm_profiles,
    get_prompt_template,
    load_prompt_templates,
    prepare_llm_postprocess_payload,
    run_llm_postprocess,
    test_connection,
    update_user_prompt_template,
)

_TEXT_FILE_EXTENSIONS = {".txt", ".md", ".markdown", ".log", ".rtf"}
_IMAGE_FILE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tif", ".tiff"}


class FileDropTarget(QLabel):
    files_dropped: Signal = Signal(list)

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(82)
        self._active = False
        self._apply_style()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        mime = event.mimeData()
        if mime.hasUrls() and any(url.isLocalFile() for url in mime.urls()):
            self._active = True
            self._apply_style()
            event.acceptProposedAction()
            return
        event.ignore()

    def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:  # noqa: N802
        self._active = False
        self._apply_style()
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        self._active = False
        self._apply_style()
        paths: list[str] = []
        for url in event.mimeData().urls():
            if not url.isLocalFile():
                continue
            candidate = str(url.toLocalFile() or "").strip()
            if not candidate or not os.path.isfile(candidate):
                continue
            if candidate not in paths:
                paths.append(candidate)
        if paths:
            self.files_dropped.emit(paths)
            event.acceptProposedAction()
            return
        event.ignore()

    def _apply_style(self) -> None:
        border = "#2563eb" if self._active else "#94a3b8"
        background = "#eff6ff" if self._active else "#f8fafc"
        text = "#1d4ed8" if self._active else "#334155"
        self.setStyleSheet(
            f"border: 2px dashed {border}; border-radius: 10px; background: {background}; color: {text}; font-weight: 600;"
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


class LLMPostprocessWorker(QObject):
    output_chunk: Signal = Signal(str)
    finished: Signal = Signal(object)
    failed: Signal = Signal(str)

    def __init__(
        self,
        *,
        profile: LLMConnectionProfile,
        template: PromptTemplate,
        request: LLMPostprocessRequest,
        prepared_payload: LLMPreparedPayload,
        run_control: LLMRunControl,
    ) -> None:
        super().__init__()
        self._profile = profile
        self._template = template
        self._request = request
        self._prepared_payload = prepared_payload
        self._run_control = run_control

    @Slot()
    def run(self) -> None:
        try:
            result = run_llm_postprocess(
                self._profile,
                self._template,
                self._request,
                prepared_payload=self._prepared_payload,
                on_output_chunk=self.output_chunk.emit,
                run_control=self._run_control,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished.emit(result)


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
        self._loaded_image_paths: tuple[str, ...] = ()
        self._is_transcription_running = is_transcription_running
        self._prefer_loaded_transcript = bool(prefer_loaded_transcript)
        self._payload_preview_required = bool(config.llm_payload_preview_required)
        self._connection_test_state: str = "not_run"
        self._last_payload_preview: str = ""
        self._last_output_text: str = ""
        self._postprocess_thread: QThread | None = None
        self._postprocess_worker: LLMPostprocessWorker | None = None
        self._run_control: LLMRunControl | None = None
        self._postprocess_active: bool = False
        self._streamed_output_chunks: list[str] = []
        self._close_after_cancel: bool = False

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
        root.setSpacing(8)
        root.setContentsMargins(10, 10, 10, 10)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        left_panel = QWidget()
        left_panel.setMinimumWidth(360)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.setSpacing(10)

        config_group = QGroupBox("Configuration")
        config_layout = QVBoxLayout(config_group)
        config_layout.setContentsMargins(10, 12, 10, 12)
        config_layout.setSpacing(8)
        self.profile_combo = QComboBox()
        self.profile_combo.setMinimumContentsLength(20)
        self.profile_combo.currentIndexChanged.connect(self._on_profile_changed)
        self.profile_combo.currentTextChanged.connect(lambda _: self._sync_combo_tooltip(self.profile_combo))
        self.template_combo = QComboBox()
        self.template_combo.setMinimumContentsLength(20)
        self.template_combo.currentIndexChanged.connect(self._on_template_changed)
        self.template_combo.currentTextChanged.connect(lambda _: self._sync_combo_tooltip(self.template_combo))
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setMinimumContentsLength(20)
        self.model_combo.currentTextChanged.connect(lambda _: self._sync_combo_tooltip(self.model_combo))
        self.refresh_btn = QPushButton("Refresh Connection + Models")
        self.refresh_btn.clicked.connect(self._on_refresh_connection)
        self.template_new_btn = QPushButton("New...")
        self.template_new_btn.clicked.connect(self._on_new_template)
        self.template_edit_btn = QPushButton("Edit...")
        self.template_edit_btn.clicked.connect(self._on_edit_template)
        self.template_delete_btn = QPushButton("Delete")
        self.template_delete_btn.clicked.connect(self._on_delete_template)

        profile_row = QHBoxLayout()
        profile_row.setSpacing(8)
        profile_row.addWidget(self.template_new_btn, 1)
        profile_row.addWidget(self.template_edit_btn, 1)
        profile_row.addWidget(self.template_delete_btn, 1)

        config_layout.addWidget(QLabel("Profile"))
        config_layout.addWidget(self.profile_combo)
        config_layout.addWidget(self.refresh_btn)
        config_layout.addWidget(QLabel("Template"))
        config_layout.addWidget(self.template_combo)
        config_layout.addLayout(profile_row)
        config_layout.addWidget(QLabel("Model"))
        config_layout.addWidget(self.model_combo)
        left_layout.addWidget(config_group)

        attachments_group = QGroupBox("Attachments")
        attachments_layout = QVBoxLayout(attachments_group)
        attachments_layout.setContentsMargins(10, 12, 10, 12)
        attachments_layout.setSpacing(8)

        self.use_current_checkbox = QCheckBox("Use current transcript")
        self.use_current_checkbox.setChecked(bool(self._current_transcript_text))
        self.use_current_checkbox.setEnabled(bool(self._current_transcript_text))
        self.load_transcript_btn = QPushButton("Load Transcript...")
        self.load_transcript_btn.clicked.connect(self._on_load_transcript)
        self.load_ocr_btn = QPushButton("Load OCR File...")
        self.load_ocr_btn.setToolTip("Optional OCR context file.")
        self.load_ocr_btn.clicked.connect(self._on_load_ocr)
        attachments_layout.addWidget(self.use_current_checkbox)
        attachments_layout.addWidget(self.load_transcript_btn)
        attachments_layout.addWidget(self.load_ocr_btn)

        self.source_label = QLabel("")
        self.ocr_label = QLabel("")
        self.source_label.setWordWrap(True)
        self.ocr_label.setWordWrap(True)
        attachments_layout.addWidget(self.source_label)
        attachments_layout.addWidget(self.ocr_label)

        image_row = QHBoxLayout()
        image_row.setSpacing(8)
        self.include_images_checkbox = QCheckBox("Include image attachments")
        self.include_images_checkbox.setChecked(bool(self._config.llm_include_images_default))
        self.load_images_btn = QPushButton("Load Images...")
        self.load_images_btn.clicked.connect(self._on_load_images)
        self.clear_images_btn = QPushButton("Clear")
        self.clear_images_btn.clicked.connect(self._on_clear_images)
        image_row.addWidget(self.load_images_btn, 1)
        image_row.addWidget(self.clear_images_btn)
        attachments_layout.addWidget(self.include_images_checkbox)
        attachments_layout.addLayout(image_row)

        attachments_layout.addWidget(QLabel("Image OCR fallback backend"))
        self.image_ocr_backend_combo = QComboBox()
        self.image_ocr_backend_combo.setMinimumContentsLength(16)
        self.image_ocr_backend_combo.addItems(["auto", "rapidocr", "paddleocr", "surya", "pytesseract"])
        self.image_ocr_backend_combo.setCurrentText("auto")
        attachments_layout.addWidget(self.image_ocr_backend_combo)
        self.image_ocr_fallback_checkbox = QCheckBox("OCR fallback for text-only models")
        self.image_ocr_fallback_checkbox.setChecked(bool(self._config.llm_ocr_fallback_for_images_default))
        attachments_layout.addWidget(self.image_ocr_fallback_checkbox)

        self.images_label = QLabel("No image attachments selected.")
        self.images_label.setWordWrap(True)
        attachments_layout.addWidget(self.images_label)

        self.drop_target = FileDropTarget("Drop transcript/OCR text files and images here")
        self.drop_target.files_dropped.connect(self._on_files_dropped)
        attachments_layout.addWidget(self.drop_target)
        left_layout.addWidget(attachments_group, 1)

        left_layout.addStretch(1)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(6, 6, 6, 6)
        right_layout.setSpacing(10)

        input_card = QFrame()
        input_card.setObjectName("Card")
        input_layout = QVBoxLayout(input_card)
        input_layout.setContentsMargins(10, 10, 10, 10)
        input_layout.setSpacing(6)
        input_layout.addWidget(QLabel("Input Context"))

        input_layout.addWidget(QLabel("Additional Notes"))
        self.notes_box = QPlainTextEdit()
        self.notes_box.setPlaceholderText("Short notes/instructions to include in post-processing (optional).")
        self.notes_box.setMinimumHeight(80)
        input_layout.addWidget(self.notes_box)

        input_layout.addWidget(QLabel("Pasted Context"))
        self.extra_context_box = QPlainTextEdit()
        self.extra_context_box.setPlaceholderText("Paste additional context text (optional).")
        self.extra_context_box.setMinimumHeight(90)
        input_layout.addWidget(self.extra_context_box)
        right_layout.addWidget(input_card, 1)

        preview_card = QFrame()
        preview_card.setObjectName("Card")
        preview_layout = QVBoxLayout(preview_card)
        preview_layout.setContentsMargins(10, 10, 10, 10)
        preview_layout.setSpacing(6)
        preview_row = QHBoxLayout()
        preview_row.addWidget(QLabel("Payload Preview"))
        self.preview_required_checkbox = QCheckBox("Require confirmation before send")
        self.preview_required_checkbox.setChecked(self._payload_preview_required)
        self.preview_required_checkbox.toggled.connect(self._on_payload_preview_requirement_changed)
        preview_row.addWidget(self.preview_required_checkbox)
        preview_row.addStretch(1)
        self.preview_btn = QPushButton("Preview Payload")
        self.preview_btn.clicked.connect(self._on_preview_payload)
        preview_row.addWidget(self.preview_btn)
        preview_layout.addLayout(preview_row)

        self.preview_box = QPlainTextEdit()
        self.preview_box.setReadOnly(True)
        self.preview_box.setPlaceholderText("Preview of request payload sent to the LLM.")
        self.preview_box.setMinimumHeight(140)
        preview_layout.addWidget(self.preview_box)
        right_layout.addWidget(preview_card, 1)

        output_card = QFrame()
        output_card.setObjectName("Card")
        output_layout = QVBoxLayout(output_card)
        output_layout.setContentsMargins(10, 10, 10, 10)
        output_layout.setSpacing(6)
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
        output_layout.addLayout(output_row)

        self.output_box = QPlainTextEdit()
        self.output_box.setReadOnly(True)
        self.output_box.setPlaceholderText("Generated post-processing output appears here.")
        self.output_box.setMinimumHeight(180)
        output_layout.addWidget(self.output_box, 1)
        right_layout.addWidget(output_card, 1)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 560])
        root.addWidget(splitter, 1)

        footer = QFrame()
        footer.setObjectName("Card")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(10, 10, 10, 10)
        footer_layout.setSpacing(8)
        self.connection_status = QLabel("Connection test optional. Click 'Refresh Connection + Models' to validate.")
        self.connection_status.setWordWrap(True)
        footer_layout.addWidget(self.connection_status, 1)
        actions = QHBoxLayout()
        self.run_btn = QPushButton("Run Post-Process")
        self.run_btn.clicked.connect(self._on_run_postprocess)
        self.cancel_run_btn = QPushButton("Cancel Generation")
        self.cancel_run_btn.setEnabled(False)
        self.cancel_run_btn.clicked.connect(self._on_cancel_postprocess)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.reject)
        actions.addWidget(self.run_btn)
        actions.addWidget(self.cancel_run_btn)
        actions.addWidget(self.close_btn)
        footer_layout.addLayout(actions)
        root.addWidget(footer)

    def _wrap_layout(self, layout: QHBoxLayout) -> QWidget:
        wrapper = QWidget()
        wrapper.setLayout(layout)
        return wrapper

    def _sync_combo_tooltip(self, combo: QComboBox) -> None:
        combo.setToolTip((combo.currentText() or "").strip())

    def _default_output_save_path(self) -> str:
        template_id = self._selected_template_id() or "template"
        if self._loaded_transcript_path:
            transcript_path = Path(self._loaded_transcript_path)
            base_name = f"{transcript_path.stem}_postprocess_{template_id}.md"
            return str(transcript_path.with_name(base_name))
        last_save_dir = str(self._config.last_save_dir or "").strip()
        if last_save_dir:
            return str(Path(last_save_dir) / f"postprocess_{template_id}.md")
        return f"postprocess_{template_id}.md"

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
        self._sync_combo_tooltip(self.profile_combo)

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
        self._sync_combo_tooltip(self.template_combo)

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
        if self._loaded_image_paths:
            names = ", ".join(os.path.basename(path) for path in self._loaded_image_paths[:4])
            extra = f" (+{len(self._loaded_image_paths) - 4} more)" if len(self._loaded_image_paths) > 4 else ""
            self.images_label.setText(f"Images attached ({len(self._loaded_image_paths)}): {names}{extra}")
        else:
            self.images_label.setText("No image attachments selected.")

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
            image_paths=self._loaded_image_paths,
            include_images=self.include_images_checkbox.isChecked(),
            image_ocr_backend=(self.image_ocr_backend_combo.currentText() or "auto").strip().lower(),
            ocr_fallback_for_images=self.image_ocr_fallback_checkbox.isChecked(),
        )

    def _render_payload_preview(self) -> tuple[PromptTemplate | None, LLMPostprocessRequest | None, LLMPreparedPayload | None]:
        profile = self._selected_profile()
        if profile is None:
            QMessageBox.warning(self, "Profile required", "Choose an enabled LLM profile.")
            return None, None, None
        transcript_text, ocr_text = self._resolve_input_context()
        if not transcript_text:
            QMessageBox.warning(
                self,
                "Transcript required",
                "No transcript text is available. Use the current transcript or load a transcript file.",
            )
            return None, None, None
        template_id = self._selected_template_id()
        if not template_id:
            QMessageBox.warning(self, "Template required", "Choose a prompt template.")
            return None, None, None
        template = get_prompt_template(template_id)
        if template is None:
            QMessageBox.warning(self, "Template unavailable", f"Template '{template_id}' could not be loaded.")
            return None, None, None
        request = self._build_request(transcript_text, ocr_text)
        prepared = prepare_llm_postprocess_payload(profile, template, request)
        if prepared.status != "pass":
            QMessageBox.warning(
                self,
                "Payload preview failed",
                f"{prepared.error_code or 'error'}: {prepared.error_detail or 'Unknown error'}",
            )
            return None, None, None
        payload_preview = prepared.payload_text
        self._last_payload_preview = payload_preview
        self.preview_box.setPlainText(payload_preview)
        self._config.llm_include_images_default = self.include_images_checkbox.isChecked()
        self._config.llm_ocr_fallback_for_images_default = self.image_ocr_fallback_checkbox.isChecked()
        self._config.llm_default_template_id = template.id
        return template, request, prepared

    @Slot()
    def _on_profile_changed(self) -> None:
        profile = self._selected_profile()
        current_text = (self.model_combo.currentText() or "").strip()
        self.model_combo.clear()
        self._sync_combo_tooltip(self.profile_combo)
        if profile is None:
            return
        self._connection_test_state = "not_run"
        self.connection_status.setText("Connection test optional. Click 'Refresh Connection + Models' to validate.")
        if profile.default_model:
            self.model_combo.addItem(profile.default_model)
            self.model_combo.setCurrentText(profile.default_model)
            self._sync_combo_tooltip(self.model_combo)
            return
        if current_text:
            self.model_combo.setEditText(current_text)
        self._sync_combo_tooltip(self.model_combo)

    @Slot(bool)
    def _on_payload_preview_requirement_changed(self, enabled: bool) -> None:
        self._payload_preview_required = bool(enabled)
        self._config.llm_payload_preview_required = bool(enabled)

    @Slot()
    def _on_template_changed(self) -> None:
        template = self._selected_template()
        editable = bool(template is not None and not template.built_in)
        self.template_edit_btn.setEnabled(editable)
        self.template_delete_btn.setEnabled(editable)
        self._sync_combo_tooltip(self.template_combo)

    @Slot()
    def _on_refresh_connection(self) -> None:
        if self._postprocess_active:
            return
        profile = self._selected_profile()
        if profile is None:
            self.connection_status.setText("No profile selected.")
            return
        refresh_label = self.refresh_btn.text()
        self.refresh_btn.setText("Testing Connection...")
        self.refresh_btn.setEnabled(False)
        self.connection_status.setText("Testing connection profile...")
        QApplication.processEvents()
        self.setCursor(Qt.WaitCursor)
        try:
            result = test_connection(profile)
        finally:
            self.unsetCursor()
            self.refresh_btn.setText(refresh_label)
            self.refresh_btn.setEnabled(True)

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
        self._sync_combo_tooltip(self.model_combo)

        if result.status == "pass":
            selected = f" Selected model: {result.selected_model}." if result.selected_model else ""
            self._connection_test_state = "pass"
            self.connection_status.setText(f"Connection test: PASS.{selected}")
            return
        self._connection_test_state = "fail"
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
    def _on_load_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Image Attachments",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All Files (*.*)",
        )
        if not paths:
            return
        normalized: list[str] = []
        for path in paths:
            candidate = str(path or "").strip()
            if not candidate or not os.path.isfile(candidate):
                continue
            if candidate not in normalized:
                normalized.append(candidate)
        self._loaded_image_paths = tuple(normalized)
        self._refresh_source_labels()

    @Slot()
    def _on_clear_images(self) -> None:
        self._loaded_image_paths = ()
        self._refresh_source_labels()

    @Slot(list)
    def _on_files_dropped(self, paths: list[str]) -> None:
        normalized: list[str] = []
        for path in paths:
            candidate = str(path or "").strip()
            if not candidate or not os.path.isfile(candidate):
                continue
            if candidate not in normalized:
                normalized.append(candidate)
        if not normalized:
            return

        text_paths = [path for path in normalized if Path(path).suffix.lower() in _TEXT_FILE_EXTENSIONS]
        image_paths = [path for path in normalized if Path(path).suffix.lower() in _IMAGE_FILE_EXTENSIONS]
        ignored_count = len(normalized) - len(text_paths) - len(image_paths)

        loaded_parts: list[str] = []
        if text_paths:
            transcript_path = text_paths[0]
            try:
                transcript_text = Path(transcript_path).read_text(encoding="utf-8").strip()
            except Exception as exc:
                QMessageBox.critical(self, "Load transcript failed", str(exc))
                transcript_text = ""
            if transcript_text:
                self._loaded_transcript_path = transcript_path
                self._loaded_transcript_text = transcript_text
                self.use_current_checkbox.setChecked(False)
                loaded_parts.append("transcript")

            ocr_path: str | None = None
            if len(text_paths) > 1:
                for candidate in text_paths[1:]:
                    name = os.path.basename(candidate).lower()
                    if "ocr" in name or "visual" in name or "slide" in name or "chat" in name:
                        ocr_path = candidate
                        break
                if ocr_path is None:
                    ocr_path = text_paths[1]
            if ocr_path is not None:
                try:
                    ocr_text = Path(ocr_path).read_text(encoding="utf-8").strip()
                except Exception as exc:
                    QMessageBox.critical(self, "Load OCR file failed", str(exc))
                    ocr_text = ""
                if ocr_text:
                    self._loaded_ocr_path = ocr_path
                    self._loaded_ocr_text = ocr_text
                    loaded_parts.append("OCR")

        if image_paths:
            merged = list(self._loaded_image_paths)
            for path in image_paths:
                if path not in merged:
                    merged.append(path)
            self._loaded_image_paths = tuple(merged)
            self.include_images_checkbox.setChecked(True)
            loaded_parts.append(f"{len(image_paths)} image(s)")

        self._refresh_source_labels()
        status = "Dropped files loaded."
        if loaded_parts:
            status = f"Dropped files loaded: {', '.join(loaded_parts)}."
        if ignored_count > 0:
            status = f"{status} Ignored {ignored_count} unsupported file(s)."
        self.connection_status.setText(status)

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
        if self._postprocess_active:
            return
        preview_label = self.preview_btn.text()
        preview_enabled = self.preview_btn.isEnabled()
        run_enabled = self.run_btn.isEnabled()
        refresh_enabled = self.refresh_btn.isEnabled()
        self.preview_btn.setText("Building Preview...")
        self.preview_btn.setEnabled(False)
        self.run_btn.setEnabled(False)
        self.refresh_btn.setEnabled(False)
        self.connection_status.setText("Preparing payload preview...")
        QApplication.processEvents()
        self.setCursor(Qt.WaitCursor)
        try:
            template, request, prepared = self._render_payload_preview()
            if template is None or request is None or prepared is None:
                return
            status = f"Payload preview ready for template '{template.id}'."
            if prepared.info_note:
                status = f"{status} {prepared.info_note}"
            self.connection_status.setText(status)
        finally:
            self.unsetCursor()
            self.preview_btn.setText(preview_label)
            self.preview_btn.setEnabled(preview_enabled)
            self.run_btn.setEnabled(run_enabled)
            self.refresh_btn.setEnabled(refresh_enabled)

    @Slot()
    def _on_run_postprocess(self) -> None:
        if self._postprocess_active:
            return
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

        self.run_btn.setText("Preparing Request...")
        self.run_btn.setEnabled(False)
        self.connection_status.setText("Preparing payload for post-processing...")
        QApplication.processEvents()

        template, request, prepared = self._render_payload_preview()
        if template is None or request is None or prepared is None:
            self.run_btn.setText("Run Post-Process")
            self.run_btn.setEnabled(True)
            return
        if self.preview_required_checkbox.isChecked():
            answer = QMessageBox.question(
                self,
                "Confirm payload",
                "Payload preview is required. Continue and send this payload to the model?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if answer != QMessageBox.Yes:
                self.connection_status.setText("Post-processing canceled before send.")
                self.run_btn.setText("Run Post-Process")
                self.run_btn.setEnabled(True)
                return

        self._streamed_output_chunks = []
        self._last_output_text = ""
        self.output_box.clear()
        self.copy_btn.setEnabled(False)
        self.save_btn.setEnabled(False)

        self._run_control = LLMRunControl()
        self._postprocess_thread = QThread(self)
        self._postprocess_worker = LLMPostprocessWorker(
            profile=profile,
            template=template,
            request=request,
            prepared_payload=prepared,
            run_control=self._run_control,
        )
        self._postprocess_worker.moveToThread(self._postprocess_thread)
        self._postprocess_thread.started.connect(self._postprocess_worker.run)
        self._postprocess_worker.output_chunk.connect(self._on_postprocess_output_chunk)
        self._postprocess_worker.finished.connect(self._on_postprocess_finished)
        self._postprocess_worker.failed.connect(self._on_postprocess_failed)
        self._postprocess_worker.finished.connect(self._postprocess_thread.quit)
        self._postprocess_worker.failed.connect(self._postprocess_thread.quit)
        self._postprocess_worker.finished.connect(self._postprocess_worker.deleteLater)
        self._postprocess_worker.failed.connect(self._postprocess_worker.deleteLater)
        self._postprocess_thread.finished.connect(self._on_postprocess_thread_finished)
        self._postprocess_thread.finished.connect(self._postprocess_thread.deleteLater)

        self._postprocess_active = True
        self.run_btn.setText("Running Post-Process...")
        self.run_btn.setEnabled(False)
        self.preview_btn.setEnabled(False)
        self.refresh_btn.setEnabled(False)
        self.cancel_run_btn.setEnabled(True)
        self.cancel_run_btn.setText("Cancel Generation")
        self.connection_status.setText("Running post-processing request...")
        self._postprocess_thread.start()

    @Slot()
    def _on_cancel_postprocess(self) -> None:
        if not self._postprocess_active:
            return
        answer = QMessageBox.question(
            self,
            "Cancel generation",
            "Stop LLM generation now? Partial output received so far will be kept.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self._request_postprocess_cancel()

    def _request_postprocess_cancel(self) -> None:
        if not self._postprocess_active:
            return
        self.cancel_run_btn.setEnabled(False)
        self.cancel_run_btn.setText("Cancelling...")
        self.run_btn.setText("Cancelling...")
        self.connection_status.setText("Cancelling post-processing request...")
        if self._run_control is not None:
            self._run_control.request_cancel()

    @Slot(str)
    def _on_postprocess_output_chunk(self, chunk: str) -> None:
        if not chunk:
            return
        self._streamed_output_chunks.append(chunk)
        text = "".join(self._streamed_output_chunks)
        self._last_output_text = text
        self.output_box.setPlainText(text)
        scrollbar = self.output_box.verticalScrollBar()
        if scrollbar is not None:
            scrollbar.setValue(scrollbar.maximum())
        if self.cancel_run_btn.text() != "Cancelling...":
            self.run_btn.setText("Streaming Output...")
            self.connection_status.setText(f"Streaming model output... {len(text)} chars received.")

    @Slot(object)
    def _on_postprocess_finished(self, result_obj: object) -> None:
        result = result_obj if isinstance(result_obj, LLMPostprocessResult) else None
        if result is None:
            self._on_postprocess_failed("Invalid post-process worker result.")
            return

        if result.status != "pass":
            if result.error_code == "cancelled":
                partial_text = (result.output_text or self._last_output_text or "").strip()
                if partial_text:
                    self._last_output_text = partial_text
                    self.output_box.setPlainText(partial_text)
                    self.copy_btn.setEnabled(True)
                    self.save_btn.setEnabled(True)
                    self.connection_status.setText("Generation cancelled. Partial output retained.")
                else:
                    self.connection_status.setText("Generation cancelled.")
            else:
                detail = result.error_detail or "Unknown error"
                profile = self._selected_profile()
                if result.error_code == "timeout" and profile is not None:
                    detail = (
                        f"{detail}\n\n"
                        f"Current timeout: {profile.timeout_seconds:.1f}s.\n"
                        "If your model is cold-starting, raise timeout in Tools > LLM Connections "
                        "(for example 30-60 seconds)."
                    )
                partial_text = (result.output_text or self._last_output_text or "").strip()
                if partial_text:
                    self._last_output_text = partial_text
                    self.output_box.setPlainText(partial_text)
                    self.copy_btn.setEnabled(True)
                    self.save_btn.setEnabled(True)
                    detail = f"{detail}\n\nPartial output was received and kept in the output box."
                    self.connection_status.setText(
                        f"Post-processing failed ({result.error_code or 'error'}). Partial output is available."
                    )
                else:
                    self.connection_status.setText(
                        f"Post-processing failed ({result.error_code or 'error'}). "
                        "Connection test pass does not guarantee generation latency."
                    )
                QMessageBox.warning(
                    self,
                    "LLM post-processing failed",
                    f"{result.error_code or 'error'}: {detail}",
                )
        else:
            self._last_output_text = (result.output_text or self._last_output_text or "").strip()
            self.output_box.setPlainText(self._last_output_text)
            self.copy_btn.setEnabled(bool(self._last_output_text))
            self.save_btn.setEnabled(bool(self._last_output_text))
            status = f"Post-processing complete with model '{result.model}'."
            if result.info_note:
                status = f"{status} {result.info_note}"
            self.connection_status.setText(status)

        self._finalize_postprocess_run()

    @Slot(str)
    def _on_postprocess_failed(self, detail: str) -> None:
        self.connection_status.setText("Post-processing failed before completion.")
        QMessageBox.warning(self, "LLM post-processing failed", detail)
        self._finalize_postprocess_run()

    @Slot()
    def _on_postprocess_thread_finished(self) -> None:
        self._postprocess_thread = None
        self._postprocess_worker = None
        self._complete_pending_close_if_ready()

    def _finalize_postprocess_run(self) -> None:
        self._postprocess_active = False
        self._run_control = None
        self.run_btn.setText("Run Post-Process")
        self.run_btn.setEnabled(bool(self._profiles))
        self.preview_btn.setEnabled(bool(self._profiles))
        self.refresh_btn.setEnabled(bool(self._profiles))
        self.cancel_run_btn.setEnabled(False)
        self.cancel_run_btn.setText("Cancel Generation")
        self._complete_pending_close_if_ready()

    def _complete_pending_close_if_ready(self) -> None:
        if not self._close_after_cancel:
            return
        if self._postprocess_active:
            return
        if self._postprocess_thread is not None and self._postprocess_thread.isRunning():
            return
        self._close_after_cancel = False
        self.reject()

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
        suggested = self._default_output_save_path()
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
        resolved_parent = str(Path(path).resolve().parent)
        self._config.last_save_dir = resolved_parent
        self.connection_status.setText(f"Saved output: {os.path.basename(path)}")

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if not self._postprocess_active:
            super().closeEvent(event)
            return
        answer = QMessageBox.question(
            self,
            "Generation in progress",
            "LLM generation is still running. Cancel generation and close this window?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            event.ignore()
            return
        self._close_after_cancel = True
        self._request_postprocess_cancel()
        event.ignore()
