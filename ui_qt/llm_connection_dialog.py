"""Qt dialog for configuring and testing LLM connection profiles."""

from __future__ import annotations

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from services import AppConfig, load_llm_profiles, test_connection


class LLMConnectionsDialog(QDialog):
    def __init__(self, config: AppConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("LLM Connections")
        self.resize(900, 620)
        self._profiles: list[dict[str, object]] = [dict(item) for item in config.llm_profiles]
        self._default_profile: str | None = config.llm_default_profile
        self._suspend_field_events = False

        self._build_ui()
        self._refresh_profile_list()
        self._refresh_default_profile_combo()
        if self.profile_list.count() > 0:
            self.profile_list.setCurrentRow(0)
        else:
            self._set_form_enabled(False)
            self._result_box.setPlainText("No profiles configured. Click 'Add Profile' to begin.")

    def profiles(self) -> list[dict[str, object]]:
        return [dict(item) for item in self._profiles]

    def default_profile(self) -> str | None:
        return self._default_profile

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(10)

        top = QHBoxLayout()
        self.profile_list = QListWidget()
        self.profile_list.currentRowChanged.connect(self._on_profile_selected)
        top.addWidget(self.profile_list, 1)

        form_wrap = QWidget()
        form = QFormLayout(form_wrap)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.name_input = QLineEdit()
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["ollama", "openai_compatible"])
        self.scope_combo = QComboBox()
        self.scope_combo.addItems(["local", "lan"])
        self.base_url_input = QLineEdit()
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.default_model_input = QLineEdit()
        self.timeout_input = QLineEdit()
        self.timeout_input.setPlaceholderText("8.0")
        self.allowed_cidrs_input = QLineEdit()
        self.allowed_cidrs_input.setPlaceholderText("192.168.0.0/16,10.0.0.0/8")
        self.verify_tls_check = QCheckBox("Verify TLS certificates")
        self.enabled_check = QCheckBox("Profile enabled")
        self.concurrent_check = QCheckBox("Allow concurrent run with local transcription")

        form.addRow("Name", self.name_input)
        form.addRow("Provider", self.provider_combo)
        form.addRow("Scope", self.scope_combo)
        form.addRow("Base URL", self.base_url_input)
        form.addRow("API Key", self.api_key_input)
        form.addRow("Default Model", self.default_model_input)
        form.addRow("Timeout (seconds)", self.timeout_input)
        form.addRow("Allowed CIDRs", self.allowed_cidrs_input)
        form.addRow("", self.verify_tls_check)
        form.addRow("", self.enabled_check)
        form.addRow("", self.concurrent_check)
        top.addWidget(form_wrap, 2)

        root.addLayout(top, 2)

        row_buttons = QHBoxLayout()
        self.add_btn = QPushButton("Add Profile")
        self.delete_btn = QPushButton("Delete Profile")
        self.apply_btn = QPushButton("Apply Changes")
        self.test_btn = QPushButton("Test Connection")
        self.add_btn.clicked.connect(self._on_add_profile)
        self.delete_btn.clicked.connect(self._on_delete_profile)
        self.apply_btn.clicked.connect(self._on_apply_profile)
        self.test_btn.clicked.connect(self._on_test_connection)
        row_buttons.addWidget(self.add_btn)
        row_buttons.addWidget(self.delete_btn)
        row_buttons.addWidget(self.apply_btn)
        row_buttons.addWidget(self.test_btn)
        row_buttons.addStretch(1)
        root.addLayout(row_buttons)

        defaults_row = QHBoxLayout()
        defaults_row.addWidget(QLabel("Default profile"))
        self.default_profile_combo = QComboBox()
        self.default_profile_combo.currentTextChanged.connect(self._on_default_profile_changed)
        defaults_row.addWidget(self.default_profile_combo, 1)
        root.addLayout(defaults_row)

        self._result_box = QTextEdit()
        self._result_box.setReadOnly(True)
        self._result_box.setPlaceholderText("Connection test results appear here.")
        root.addWidget(self._result_box, 1)

        actions = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        save_btn = QPushButton("Save and Close")
        cancel_btn.clicked.connect(self.reject)
        save_btn.clicked.connect(self._on_save_and_close)
        actions.addStretch(1)
        actions.addWidget(cancel_btn)
        actions.addWidget(save_btn)
        root.addLayout(actions)

    def _set_form_enabled(self, enabled: bool) -> None:
        widgets = [
            self.name_input,
            self.provider_combo,
            self.scope_combo,
            self.base_url_input,
            self.api_key_input,
            self.default_model_input,
            self.timeout_input,
            self.allowed_cidrs_input,
            self.verify_tls_check,
            self.enabled_check,
            self.concurrent_check,
            self.delete_btn,
            self.apply_btn,
            self.test_btn,
        ]
        for widget in widgets:
            widget.setEnabled(enabled)

    @Slot()
    def _on_add_profile(self) -> None:
        profile = {
            "name": f"profile-{len(self._profiles) + 1}",
            "provider": "ollama",
            "scope": "local",
            "base_url": "http://127.0.0.1:11434",
            "api_key": "",
            "default_model": "",
            "timeout_seconds": 8.0,
            "verify_tls": True,
            "enabled": True,
            "allow_concurrent_with_local_transcription": False,
            "allowed_cidrs": ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"],
        }
        self._profiles.append(profile)
        self._refresh_profile_list()
        self.profile_list.setCurrentRow(len(self._profiles) - 1)
        self._refresh_default_profile_combo()

    @Slot()
    def _on_delete_profile(self) -> None:
        row = self.profile_list.currentRow()
        if row < 0 or row >= len(self._profiles):
            return
        name = str(self._profiles[row].get("name", ""))
        answer = QMessageBox.question(
            self,
            "Delete profile",
            f"Delete profile '{name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        del self._profiles[row]
        if self._default_profile == name:
            self._default_profile = None
        self._refresh_profile_list()
        self._refresh_default_profile_combo()
        if self._profiles:
            self.profile_list.setCurrentRow(max(0, min(row, len(self._profiles) - 1)))
            self._set_form_enabled(True)
        else:
            self._set_form_enabled(False)
            self._result_box.setPlainText("No profiles configured. Click 'Add Profile' to begin.")

    @Slot(int)
    def _on_profile_selected(self, row: int) -> None:
        if row < 0 or row >= len(self._profiles):
            self._set_form_enabled(False)
            return
        self._set_form_enabled(True)
        profile = self._profiles[row]
        self._suspend_field_events = True
        try:
            self.name_input.setText(str(profile.get("name", "")))
            self.provider_combo.setCurrentText(str(profile.get("provider", "ollama")))
            self.scope_combo.setCurrentText(str(profile.get("scope", "local")))
            self.base_url_input.setText(str(profile.get("base_url", "")))
            self.api_key_input.setText(str(profile.get("api_key", "")))
            self.default_model_input.setText(str(profile.get("default_model", "")))
            timeout = profile.get("timeout_seconds", 8.0)
            self.timeout_input.setText(str(timeout))
            cidrs = profile.get("allowed_cidrs", [])
            if isinstance(cidrs, list):
                self.allowed_cidrs_input.setText(",".join(str(item) for item in cidrs))
            else:
                self.allowed_cidrs_input.setText("")
            self.verify_tls_check.setChecked(bool(profile.get("verify_tls", True)))
            self.enabled_check.setChecked(bool(profile.get("enabled", True)))
            self.concurrent_check.setChecked(bool(profile.get("allow_concurrent_with_local_transcription", False)))
        finally:
            self._suspend_field_events = False

    @Slot()
    def _on_apply_profile(self) -> None:
        row = self.profile_list.currentRow()
        if row < 0 or row >= len(self._profiles):
            return
        try:
            timeout = float(self.timeout_input.text().strip() or "8.0")
        except ValueError:
            QMessageBox.warning(self, "Invalid timeout", "Timeout must be a number.")
            return
        cidr_values = [
            part.strip()
            for part in (self.allowed_cidrs_input.text() or "").split(",")
            if part.strip()
        ]
        profile = {
            "name": (self.name_input.text() or "").strip() or f"profile-{row + 1}",
            "provider": self.provider_combo.currentText().strip(),
            "scope": self.scope_combo.currentText().strip(),
            "base_url": (self.base_url_input.text() or "").strip(),
            "api_key": (self.api_key_input.text() or "").strip(),
            "default_model": (self.default_model_input.text() or "").strip(),
            "timeout_seconds": timeout,
            "verify_tls": self.verify_tls_check.isChecked(),
            "enabled": self.enabled_check.isChecked(),
            "allow_concurrent_with_local_transcription": self.concurrent_check.isChecked(),
            "allowed_cidrs": cidr_values,
        }
        self._profiles[row] = profile
        self._refresh_profile_list()
        self.profile_list.setCurrentRow(row)
        self._refresh_default_profile_combo()
        self._result_box.setPlainText("Profile changes applied.")

    @Slot()
    def _on_test_connection(self) -> None:
        self._on_apply_profile()
        row = self.profile_list.currentRow()
        if row < 0 or row >= len(self._profiles):
            return
        parsed_profiles = load_llm_profiles([self._profiles[row]])
        if not parsed_profiles:
            self._result_box.setPlainText("Profile is invalid. Check provider/scope/base URL fields.")
            return
        profile = parsed_profiles[0]
        self.setCursor(Qt.WaitCursor)
        try:
            result = test_connection(profile)
        finally:
            self.unsetCursor()
        lines: list[str] = []
        lines.append(f"Overall: {result.status.upper()}")
        lines.append(f"Provider: {result.provider}")
        lines.append(f"Base URL: {result.base_url}")
        if result.selected_model:
            lines.append(f"Selected model: {result.selected_model}")
        if result.loaded_model:
            lines.append(f"Loaded model: {result.loaded_model}")
        if result.detected_models:
            lines.append("Detected models:")
            for model_name in result.detected_models:
                lines.append(f"- {model_name}")
        lines.append("")
        lines.append("Stages:")
        for stage in result.stages:
            lines.append(f"- [{stage.status.upper()}] {stage.stage}: {stage.detail}")
            if stage.suggestions:
                for suggestion in stage.suggestions:
                    lines.append(f"  * {suggestion}")
        if result.failure_code:
            lines.append("")
            lines.append(f"Failure code: {result.failure_code}")
            lines.append(f"Failure detail: {result.failure_detail}")
        self._result_box.setPlainText("\n".join(lines))

    @Slot(str)
    def _on_default_profile_changed(self, value: str) -> None:
        if self._suspend_field_events:
            return
        text = value.strip()
        self._default_profile = text or None

    @Slot()
    def _on_save_and_close(self) -> None:
        # Ensure active edits are applied before saving.
        self._on_apply_profile()
        if self._default_profile and not any(self._default_profile == str(item.get("name", "")) for item in self._profiles):
            self._default_profile = None
        self.accept()

    def _refresh_profile_list(self) -> None:
        current = self.profile_list.currentRow()
        self.profile_list.blockSignals(True)
        self.profile_list.clear()
        for profile in self._profiles:
            name = str(profile.get("name", "unnamed"))
            provider = str(profile.get("provider", ""))
            scope = str(profile.get("scope", ""))
            enabled = bool(profile.get("enabled", True))
            label = f"{name} ({provider}, {scope})"
            if not enabled:
                label += " [disabled]"
            self.profile_list.addItem(QListWidgetItem(label))
        self.profile_list.blockSignals(False)
        if self._profiles and 0 <= current < len(self._profiles):
            self.profile_list.setCurrentRow(current)

    def _refresh_default_profile_combo(self) -> None:
        self.default_profile_combo.blockSignals(True)
        self.default_profile_combo.clear()
        self.default_profile_combo.addItem("")
        for profile in self._profiles:
            self.default_profile_combo.addItem(str(profile.get("name", "")))
        if self._default_profile:
            index = self.default_profile_combo.findText(self._default_profile)
            if index >= 0:
                self.default_profile_combo.setCurrentIndex(index)
            else:
                self.default_profile_combo.setCurrentIndex(0)
                self._default_profile = None
        else:
            self.default_profile_combo.setCurrentIndex(0)
        self.default_profile_combo.blockSignals(False)

    def keyPressEvent(self, event) -> None:  # noqa: ANN001
        if event.key() == Qt.Key_Escape:
            event.ignore()
            return
        super().keyPressEvent(event)
