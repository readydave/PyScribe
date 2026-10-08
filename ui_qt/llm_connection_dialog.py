"""Qt dialog for configuring and testing LLM connection profiles."""

from __future__ import annotations

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
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

from services import AppConfig, discover_local_networks, load_llm_profiles, scan_lan_for_llm_instances, run_connection_test


# Hosted providers offered as one-click starting points. Model names are left empty (except Anthropic)
# because each provider's own model list changes; "Test Connection" lists what the key can use.
CLOUD_PRESETS: tuple[dict[str, object], ...] = (
    {
        "label": "Anthropic (Claude)",
        "name": "anthropic",
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com",
        "api_key": "env:ANTHROPIC_API_KEY",
        "default_model": "claude-sonnet-5-5",
    },
    {
        "label": "OpenAI",
        "name": "openai",
        "provider": "openai_compatible",
        "base_url": "https://api.openai.com",
        "api_key": "env:OPENAI_API_KEY",
        "default_model": "",
    },
    {
        "label": "Google Gemini",
        "name": "gemini",
        "provider": "openai_compatible",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "api_key": "env:GEMINI_API_KEY",
        "default_model": "",
    },
    {
        "label": "OpenRouter (many models)",
        "name": "openrouter",
        "provider": "openai_compatible",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "env:OPENROUTER_API_KEY",
        "default_model": "",
    },
)


class LLMConnectionsDialog(QDialog):
    def __init__(self, config: AppConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("LLM Connections")
        self.resize(900, 620)
        self._profiles: list[dict[str, object]] = [dict(item) for item in config.llm_profiles]
        self._default_profile: str | None = config.llm_default_profile
        self._suspend_field_events = False
        self._local_networks: list[object] = []
        self._scan_results: list[object] = []

        self._build_ui()
        self._on_refresh_networks()
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
        self.provider_combo.addItems(["ollama", "lm_studio", "openai_compatible", "anthropic"])
        self.provider_combo.currentTextChanged.connect(self._on_provider_changed)
        self.scope_combo = QComboBox()
        self.scope_combo.addItems(["local", "lan", "cloud"])
        self.scope_combo.currentTextChanged.connect(self._on_scope_changed)
        self.base_url_input = QLineEdit()
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setPlaceholderText("env:MY_API_KEY (recommended), or a key kept for this session only")
        self.api_key_input.textChanged.connect(self._on_api_key_text_changed)
        self.default_model_input = QLineEdit()
        self.timeout_input = QLineEdit()
        self.timeout_input.setPlaceholderText("8.0")
        self.allowed_cidrs_input = QLineEdit()
        self.allowed_cidrs_input.setPlaceholderText("192.168.0.0/16,10.0.0.0/8")
        self.verify_tls_check = QCheckBox("Verify TLS certificates")
        self.verify_tls_check.setToolTip(
            "Required for HTTPS LAN endpoints. Only disable certificate verification for "
            "localhost/loopback development endpoints."
        )
        self.context_tokens_input = QLineEdit()
        self.context_tokens_input.setPlaceholderText("Automatic")
        self.context_tokens_input.setToolTip(
            "How many tokens the model can read at once. Transcripts longer than this are split into parts "
            "and merged automatically. For LM Studio, set this to the context length you loaded the model with."
        )
        self.max_output_input = QLineEdit()
        self.max_output_input.setPlaceholderText("Automatic")
        self.max_output_input.setToolTip("The longest reply to allow, in tokens.")
        self.temperature_input = QLineEdit()
        self.temperature_input.setPlaceholderText("Provider default")
        self.temperature_input.setToolTip(
            "0 = most literal, higher = more varied. Leave empty to use the provider's default "
            "(some frontier models reject a custom value)."
        )
        self.cloud_ack_check = QCheckBox("I understand transcripts and images will be sent to this provider")
        self.cloud_ack_check.setToolTip("Required for cloud profiles. Meeting transcripts may contain sensitive information.")
        self.enabled_check = QCheckBox("Profile enabled")
        self.concurrent_check = QCheckBox("Allow concurrent run with local transcription")

        form.addRow("Name", self.name_input)
        form.addRow("Provider", self.provider_combo)
        form.addRow("Scope", self.scope_combo)
        form.addRow("Base URL", self.base_url_input)
        form.addRow("API Key", self.api_key_input)
        form.addRow("Default Model", self.default_model_input)
        form.addRow("Timeout (seconds)", self.timeout_input)
        form.addRow("Context tokens", self.context_tokens_input)
        form.addRow("Max output tokens", self.max_output_input)
        form.addRow("Temperature", self.temperature_input)
        form.addRow("Allowed CIDRs", self.allowed_cidrs_input)
        form.addRow("", self.cloud_ack_check)
        form.addRow("", self.verify_tls_check)
        form.addRow("", self.enabled_check)
        form.addRow("", self.concurrent_check)
        top.addWidget(form_wrap, 2)

        root.addLayout(top, 2)

        row_buttons = QHBoxLayout()
        self.add_btn = QPushButton("Add Profile")
        self.rename_btn = QPushButton("Rename Profile")
        self.delete_btn = QPushButton("Delete Profile")
        self.apply_btn = QPushButton("Apply Changes")
        self.test_btn = QPushButton("Test Connection")
        self.add_btn.clicked.connect(self._on_add_profile)
        self.rename_btn.clicked.connect(self._on_rename_profile)
        self.delete_btn.clicked.connect(self._on_delete_profile)
        self.apply_btn.clicked.connect(self._on_apply_profile)
        self.test_btn.clicked.connect(self._on_test_connection)
        self.preset_combo = QComboBox()
        for preset in CLOUD_PRESETS:
            self.preset_combo.addItem(str(preset["label"]), preset)
        self.add_preset_btn = QPushButton("Add Cloud Profile")
        self.add_preset_btn.setToolTip("Add a hosted provider such as Claude, OpenAI, Gemini, or OpenRouter.")
        self.add_preset_btn.clicked.connect(self._on_add_cloud_preset)
        row_buttons.addWidget(self.add_btn)
        row_buttons.addWidget(self.preset_combo)
        row_buttons.addWidget(self.add_preset_btn)
        row_buttons.addWidget(self.rename_btn)
        row_buttons.addWidget(self.delete_btn)
        row_buttons.addWidget(self.apply_btn)
        row_buttons.addWidget(self.test_btn)
        row_buttons.addStretch(1)
        root.addLayout(row_buttons)

        scan_row = QHBoxLayout()
        self.include_non_private_check = QCheckBox("Include non-private/VPN interfaces")
        self.include_loopback_check = QCheckBox("Include loopback")
        self.refresh_networks_btn = QPushButton("Detect Networks")
        self.refresh_networks_btn.clicked.connect(self._on_refresh_networks)
        self.network_combo = QComboBox()
        self.scan_btn = QPushButton("Scan Selected Network")
        self.scan_btn.clicked.connect(self._on_scan_network)
        scan_row.addWidget(self.include_non_private_check)
        scan_row.addWidget(self.include_loopback_check)
        scan_row.addWidget(self.refresh_networks_btn)
        scan_row.addWidget(self.network_combo, 1)
        scan_row.addWidget(self.scan_btn)
        root.addLayout(scan_row)

        apply_scan_row = QHBoxLayout()
        apply_scan_row.addWidget(QLabel("Discovered endpoint"))
        self.scan_results_combo = QComboBox()
        self.apply_scan_btn = QPushButton("Apply Scan Result")
        self.apply_scan_btn.clicked.connect(self._on_apply_scan_result)
        apply_scan_row.addWidget(self.scan_results_combo, 1)
        apply_scan_row.addWidget(self.apply_scan_btn)
        root.addLayout(apply_scan_row)

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
            self.context_tokens_input,
            self.max_output_input,
            self.temperature_input,
            self.allowed_cidrs_input,
            self.verify_tls_check,
            self.cloud_ack_check,
            self.enabled_check,
            self.concurrent_check,
            self.rename_btn,
            self.delete_btn,
            self.apply_btn,
            self.test_btn,
        ]
        for widget in widgets:
            widget.setEnabled(enabled)

    @Slot()
    def _on_refresh_networks(self) -> None:
        include_non_private = self.include_non_private_check.isChecked()
        include_loopback = self.include_loopback_check.isChecked()
        self._local_networks = discover_local_networks(
            include_non_private=include_non_private,
            include_loopback=include_loopback,
        )
        self.network_combo.clear()
        for info in self._local_networks:
            label = f"{info.interface_name}: {info.ip_address} ({info.scan_cidr})"
            self.network_combo.addItem(label, info.scan_cidr)
        if self.network_combo.count() == 0:
            self.network_combo.addItem("No networks detected", "")

    @Slot()
    def _on_scan_network(self) -> None:
        scan_cidr = str(self.network_combo.currentData() or "").strip()
        if not scan_cidr:
            QMessageBox.information(self, "Scan", "No network/subnet selected for scan.")
            return
        self.setCursor(Qt.WaitCursor)
        try:
            results = scan_lan_for_llm_instances(
                [scan_cidr],
                include_ollama=True,
                include_openai_compatible=True,
                include_lm_studio=True,
                timeout_seconds=0.30,
                max_hosts_per_network=256,
            )
        finally:
            self.unsetCursor()

        self._scan_results = list(results)
        self.scan_results_combo.clear()
        lines: list[str] = [f"Scan network: {scan_cidr}"]
        if not self._scan_results:
            lines.append("No endpoints discovered.")
            self._result_box.setPlainText("\n".join(lines))
            return
        lines.append(f"Discovered endpoints: {len(self._scan_results)}")
        for idx, item in enumerate(self._scan_results):
            label = f"{item.provider} | {item.base_url} | {item.network_cidr}"
            self.scan_results_combo.addItem(label, idx)
            lines.append(f"- {label}")
            if item.detected_models:
                lines.append(f"  models: {', '.join(item.detected_models[:6])}")
        self._result_box.setPlainText("\n".join(lines))

    @Slot()
    def _on_apply_scan_result(self) -> None:
        value = self.scan_results_combo.currentData()
        if value is None:
            QMessageBox.information(self, "Apply scan", "No scan result selected.")
            return
        try:
            idx = int(value)
        except (TypeError, ValueError):
            QMessageBox.information(self, "Apply scan", "No scan result selected.")
            return
        if idx < 0 or idx >= len(self._scan_results):
            QMessageBox.information(self, "Apply scan", "No scan result selected.")
            return
        item = self._scan_results[idx]
        self.provider_combo.setCurrentText(str(item.provider))
        self.base_url_input.setText(str(item.base_url))
        self.scope_combo.setCurrentText("local" if item.scope_hint == "local" else "lan")
        if item.detected_models:
            self.default_model_input.setText(str(item.detected_models[0]))
        if item.network_cidr:
            self.allowed_cidrs_input.setText(str(item.network_cidr))
        self._result_box.setPlainText(f"Applied scan result: {item.provider} at {item.base_url}")

    @Slot(str)
    def _on_api_key_text_changed(self, text: str) -> None:
        # An env:NAME reference is not a secret, so show it; anything else stays masked.
        mode = QLineEdit.Normal if (text or "").strip().lower().startswith("env:") else QLineEdit.Password
        if self.api_key_input.echoMode() != mode:
            self.api_key_input.setEchoMode(mode)

    @Slot(str)
    def _on_scope_changed(self, value: str) -> None:
        is_cloud = (value or "").strip().lower() == "cloud"
        self.cloud_ack_check.setVisible(is_cloud)
        self.allowed_cidrs_input.setEnabled(not is_cloud)
        self.concurrent_check.setEnabled(not is_cloud)
        if is_cloud:
            self.verify_tls_check.setChecked(True)
        self.verify_tls_check.setEnabled(not is_cloud)

    @Slot()
    def _on_add_cloud_preset(self) -> None:
        preset = self.preset_combo.currentData()
        if not isinstance(preset, dict):
            return
        name = str(preset["name"])
        candidate, counter = name, 2
        while self._is_profile_name_in_use(candidate):
            candidate = f"{name}-{counter}"
            counter += 1
        self._profiles.append(
            {
                "name": candidate,
                "provider": preset["provider"],
                "scope": "cloud",
                "base_url": preset["base_url"],
                "api_key": preset["api_key"],
                "api_key_runtime": "",
                "default_model": preset["default_model"],
                "timeout_seconds": 120.0,
                "verify_tls": True,
                "enabled": True,
                "allow_concurrent_with_local_transcription": False,
                "allowed_cidrs": [],
                "cloud_acknowledged": False,
                "temperature": None,
            }
        )
        self._refresh_profile_list()
        self.profile_list.setCurrentRow(len(self._profiles) - 1)
        self._refresh_default_profile_combo()
        self._result_box.setPlainText(
            f"Added '{candidate}'. Set the {preset['api_key']} environment variable (or paste a key for this "
            "session), tick the confirmation box, then press Test Connection to list the models your key can use."
        )

    @Slot(str)
    def _on_provider_changed(self, value: str) -> None:
        if self._suspend_field_events:
            return
        provider = (value or "").strip().lower()
        current_url = (self.base_url_input.text() or "").strip()
        if provider == "anthropic":
            if not current_url or "127.0.0.1" in current_url or "localhost" in current_url:
                self.base_url_input.setText("https://api.anthropic.com")
            self.scope_combo.setCurrentText("cloud")
            return
        if provider == "ollama":
            if not current_url or current_url in {"http://127.0.0.1:1234", "http://localhost:1234"}:
                self.base_url_input.setText("http://127.0.0.1:11434")
            return
        if provider == "lm_studio":
            if not current_url or current_url in {"http://127.0.0.1:11434", "http://localhost:11434"}:
                self.base_url_input.setText("http://127.0.0.1:1234")
            if (self.scope_combo.currentText() or "").strip().lower() == "lan" and "127.0.0.1" in self.base_url_input.text():
                self.scope_combo.setCurrentText("local")
            return
        # openai_compatible default
        if not current_url:
            self.base_url_input.setText("http://127.0.0.1:1234")

    @Slot()
    def _on_add_profile(self) -> None:
        profile_name = self._next_profile_name()
        profile = {
            "name": profile_name,
            "provider": "ollama",
            "scope": "local",
            "base_url": "http://127.0.0.1:11434",
            "api_key": "",
            "api_key_runtime": "",
            "default_model": "",
            "timeout_seconds": 8.0,
            "verify_tls": True,
            "enabled": True,
            "allow_concurrent_with_local_transcription": False,
            "allowed_cidrs": ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"],
            "context_tokens": 0,
            "max_output_tokens": 0,
            "temperature": 0.2,
            "cloud_acknowledged": False,
        }
        self._profiles.append(profile)
        self._refresh_profile_list()
        self.profile_list.setCurrentRow(len(self._profiles) - 1)
        self._refresh_default_profile_combo()

    @Slot()
    def _on_rename_profile(self) -> None:
        row = self.profile_list.currentRow()
        if row < 0 or row >= len(self._profiles):
            return
        current_name = str(self._profiles[row].get("name", "")).strip() or f"profile-{row + 1}"
        new_name, ok = QInputDialog.getText(
            self,
            "Rename profile",
            "New profile name:",
            QLineEdit.Normal,
            current_name,
        )
        if not ok:
            return
        candidate = (new_name or "").strip()
        if not candidate:
            QMessageBox.warning(self, "Invalid name", "Profile name cannot be empty.")
            return
        if self._is_profile_name_in_use(candidate, exclude_index=row):
            QMessageBox.warning(self, "Duplicate name", f"A profile named '{candidate}' already exists.")
            return
        self._profiles[row]["name"] = candidate
        self.name_input.setText(candidate)
        if self._default_profile and self._default_profile.strip().lower() == current_name.strip().lower():
            self._default_profile = candidate
        self._refresh_profile_list()
        self.profile_list.setCurrentRow(row)
        self._refresh_default_profile_combo()
        self._result_box.setPlainText(f"Profile renamed: {current_name} -> {candidate}")

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
            runtime_key = str(profile.get("api_key_runtime", ""))
            persisted_key = str(profile.get("api_key", ""))
            self.api_key_input.setText(runtime_key or persisted_key)
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
            self.context_tokens_input.setText(self._number_text(profile.get("context_tokens")))
            self.max_output_input.setText(self._number_text(profile.get("max_output_tokens")))
            temperature = profile.get("temperature", 0.2 if str(profile.get("scope", "local")) != "cloud" else None)
            self.temperature_input.setText("" if temperature is None else str(temperature))
            self.cloud_ack_check.setChecked(bool(profile.get("cloud_acknowledged", False)))
            self._on_scope_changed(str(profile.get("scope", "local")))
        finally:
            self._suspend_field_events = False

    @Slot()
    def _on_apply_profile(self) -> None:
        row = self.profile_list.currentRow()
        if row < 0 or row >= len(self._profiles):
            return
        old_name = str(self._profiles[row].get("name", "")).strip() or f"profile-{row + 1}"
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
        try:
            context_tokens = self._parse_optional_int(self.context_tokens_input.text(), 1000, 2_000_000)
            max_output_tokens = self._parse_optional_int(self.max_output_input.text(), 16, 200_000)
            temperature = self._parse_optional_temperature(self.temperature_input.text())
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid value", str(exc))
            return
        scope_value = self.scope_combo.currentText().strip()
        base_url_value = (self.base_url_input.text() or "").strip()
        if scope_value == "cloud":
            if not base_url_value.lower().startswith("https://"):
                QMessageBox.warning(self, "Cloud profile", "Cloud profiles must use an https:// address.")
                return
            if not self.cloud_ack_check.isChecked():
                QMessageBox.warning(
                    self,
                    "Confirm cloud use",
                    "Tick the confirmation box: transcripts (and any images you attach) are sent to this provider.",
                )
                return
        if scope_value == "lan" and ("127.0.0.1" in base_url_value or "localhost" in base_url_value):
            scope_value = "local"
            self.scope_combo.setCurrentText("local")
        new_name = (self.name_input.text() or "").strip() or f"profile-{row + 1}"
        if self._is_profile_name_in_use(new_name, exclude_index=row):
            QMessageBox.warning(self, "Duplicate name", f"A profile named '{new_name}' already exists.")
            return
        entered_api_key = (self.api_key_input.text() or "").strip()
        persisted_api_key = entered_api_key if entered_api_key.lower().startswith("env:") else ""
        runtime_api_key = entered_api_key if entered_api_key and not entered_api_key.lower().startswith("env:") else ""
        verify_tls = self.verify_tls_check.isChecked()
        if (
            scope_value == "lan"
            and base_url_value.lower().startswith("https://")
            and not verify_tls
        ):
            QMessageBox.warning(
                self,
                "TLS policy",
                "HTTPS LAN endpoints must keep certificate verification enabled. "
                "Install a trusted certificate or switch the endpoint to HTTP if that matches your deployment.",
            )
            return
        profile = {
            "name": new_name,
            "provider": self.provider_combo.currentText().strip(),
            "scope": scope_value,
            "base_url": base_url_value,
            "api_key": persisted_api_key,
            "api_key_runtime": runtime_api_key,
            "default_model": (self.default_model_input.text() or "").strip(),
            "timeout_seconds": timeout,
            "verify_tls": verify_tls,
            "enabled": self.enabled_check.isChecked(),
            "allow_concurrent_with_local_transcription": self.concurrent_check.isChecked(),
            "allowed_cidrs": cidr_values,
            "context_tokens": context_tokens,
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
            "cloud_acknowledged": bool(self.cloud_ack_check.isChecked() and scope_value == "cloud"),
        }
        self._profiles[row] = profile
        if self._default_profile and self._default_profile.strip().lower() == old_name.strip().lower():
            self._default_profile = new_name
        self._refresh_profile_list()
        self.profile_list.setCurrentRow(row)
        self._refresh_default_profile_combo()
        session_key_note = ""
        if runtime_api_key:
            session_key_note = " Using a session-only API key (not saved to disk)."
        if scope_value == "local" and ("127.0.0.1" in base_url_value or "localhost" in base_url_value):
            self._result_box.setPlainText(
                f"Profile changes applied. Scope set to local for loopback endpoint.{session_key_note}"
            )
        else:
            self._result_box.setPlainText(f"Profile changes applied.{session_key_note}")

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
            result = run_connection_test(profile)
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

    def _is_profile_name_in_use(self, name: str, *, exclude_index: int | None = None) -> bool:
        target = str(name or "").strip().lower()
        if not target:
            return False
        for idx, profile in enumerate(self._profiles):
            if exclude_index is not None and idx == exclude_index:
                continue
            existing = str(profile.get("name", "")).strip().lower()
            if existing and existing == target:
                return True
        return False

    @staticmethod
    def _number_text(value: object) -> str:
        try:
            number = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return ""
        return str(number) if number > 0 else ""

    @staticmethod
    def _parse_optional_int(text: str, minimum: int, maximum: int) -> int:
        cleaned = (text or "").strip().replace(",", "")
        if not cleaned:
            return 0
        try:
            value = int(cleaned)
        except ValueError:
            raise ValueError("Token limits must be whole numbers, or empty for automatic.") from None
        if not minimum <= value <= maximum:
            raise ValueError(f"Token limits must be between {minimum:,} and {maximum:,}.")
        return value

    @staticmethod
    def _parse_optional_temperature(text: str) -> float | None:
        cleaned = (text or "").strip()
        if not cleaned:
            return None
        try:
            value = float(cleaned)
        except ValueError:
            raise ValueError("Temperature must be a number between 0 and 2, or empty.") from None
        if not 0.0 <= value <= 2.0:
            raise ValueError("Temperature must be a number between 0 and 2, or empty.")
        return value

    def _next_profile_name(self) -> str:
        idx = len(self._profiles) + 1
        while True:
            candidate = f"profile-{idx}"
            if not self._is_profile_name_in_use(candidate):
                return candidate
            idx += 1

    def keyPressEvent(self, event) -> None:  # noqa: ANN001
        if event.key() == Qt.Key_Escape:
            event.ignore()
            return
        super().keyPressEvent(event)
