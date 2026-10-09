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

import ipaddress
from collections.abc import Callable
from urllib.parse import urlparse

from services import AppConfig, discover_local_networks, load_llm_profiles, scan_lan_for_llm_instances, run_connection_test
from services import secret_store
from ui_qt import keyring_worker

KEYRING_PLACEHOLDER = "stored in system keyring"


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

# Providers that run the user's own signed-in command-line program (no URL, no key). One row per provider:
# add or remove a row and the provider list, the "Add CLI Profile" button and the field handling follow.
CLI_PROVIDERS: tuple[dict[str, str], ...] = (
    {
        "id": "claude_cli",
        "label": "Claude Code CLI",
        "name": "claude-cli",
        "default_model": "",
        "note": "Runs your own signed-in Claude Code CLI, for personal use only. See docs/user_guide.md (LLM Connections).",
        "ack": "I understand transcripts will be sent through my signed-in CLI to the vendor",
    },
)
CLI_PROVIDER_IDS = frozenset(item["id"] for item in CLI_PROVIDERS)
CLOUD_ACK_TEXT = "I understand transcripts and images will be sent to this provider"

# Which form rows each provider shows, per scope. "offered" are the scopes the Scope box lists for a new or
# changed provider; every scope the services accept still has a row set, so a saved profile that uses an
# unusual scope (e.g. ollama on cloud, anthropic on lan) keeps showing the right rows.
_COMMON_ROWS = ("scope", "model", "timeout", "context", "max_output", "temperature")
_HTTP_ROWS = ("base_url", "api_key", "keyring")
# "discovery" is the network-scan block under the form; it only makes sense for an HTTP endpoint on this network.
_SCOPE_EXTRA_ROWS = {
    "local": ("tls", "concurrent", "discovery"),
    "lan": ("cidrs", "tls", "concurrent", "discovery"),
    "cloud": ("ack",),
}


def _http_provider(offered: tuple[str, ...]) -> dict[str, object]:
    return {
        "offered": offered,
        "fields": {scope: frozenset(_COMMON_ROWS + _HTTP_ROWS + extra) for scope, extra in _SCOPE_EXTRA_ROWS.items()},
    }


PROVIDER_FORM: dict[str, dict[str, object]] = {
    "ollama": _http_provider(("local", "lan")),
    "lm_studio": _http_provider(("local", "lan")),
    "openai_compatible": _http_provider(("local", "lan", "cloud")),
    "anthropic": _http_provider(("cloud",)),
}
for _cli in CLI_PROVIDERS:  # no URL, key, TLS, CIDRs, concurrency or temperature: the CLI only takes a model and a timeout
    PROVIDER_FORM[_cli["id"]] = {
        "offered": ("cloud",),
        "fields": {"cloud": frozenset(("scope", "model", "timeout", "context", "max_output", "ack"))},
    }
DEFAULT_CIDRS_TEXT = "10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"


def _needs_public_https(url: str) -> bool:
    """True for a URL that cannot be the hosted Anthropic API: not https, or pointing at a private/LAN host."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme.lower() != "https" or not host:
        return True
    if host == "localhost" or host.endswith((".local", ".lan", ".internal")):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_private or address.is_loopback or address.is_link_local


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
        self._keyring_available = False
        self._closed = False
        self._busy_keyring = False
        self._availability_handle: keyring_worker.TaskHandle | None = None
        self._test_handle: keyring_worker.TaskHandle | None = None
        self._created_refs: set[str] = set()  # keyring entries made in this session (removed if the dialog is cancelled)
        self._pending_deletes: set[str] = set()  # entries of replaced/deleted keys (removed once the dialog is saved)

        self._build_ui()
        self._apply_form_rules(self.provider_combo.currentText(), self.scope_combo.currentText())
        self._availability_handle = keyring_worker.start_task(
            keyring_worker.check_available(), on_done=self._on_keyring_availability
        )
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
        form = self._form = QFormLayout(form_wrap)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.name_input = QLineEdit()
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["ollama", "lm_studio", "openai_compatible", "anthropic", *(item["id"] for item in CLI_PROVIDERS)])
        self.provider_combo.currentTextChanged.connect(self._on_provider_changed)
        self.scope_combo = QComboBox()
        self.scope_combo.addItems(["local", "lan", "cloud"])
        self.scope_combo.currentTextChanged.connect(self._on_scope_changed)
        self.cli_note_label = QLabel("")
        self.cli_note_label.setWordWrap(True)
        self.cli_note_label.setVisible(False)
        font = self.cli_note_label.font()
        font.setItalic(True)
        self.cli_note_label.setFont(font)
        self.base_url_input = QLineEdit()
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setPlaceholderText("env:MY_API_KEY (recommended), or a key kept for this session only")
        self.api_key_input.textChanged.connect(self._on_api_key_text_changed)
        self.keyring_check = QCheckBox("Store in system keyring")
        self.keyring_check.setToolTip("Keep a typed API key in the operating system keyring instead of for this session only.")
        self.keyring_check.setVisible(False)  # shown once the keyring is found to be usable
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
        self.cloud_ack_check = QCheckBox(CLOUD_ACK_TEXT)
        self.cloud_ack_check.setToolTip("Required for cloud profiles. Meeting transcripts may contain sensitive information.")
        self.enabled_check = QCheckBox("Profile enabled")
        self.concurrent_check = QCheckBox("Allow concurrent run with local transcription")

        form.addRow("Name", self.name_input)
        form.addRow("Provider", self.provider_combo)
        form.addRow("", self.cli_note_label)
        form.addRow("Scope", self.scope_combo)
        form.addRow("Base URL", self.base_url_input)
        form.addRow("API Key", self.api_key_input)
        form.addRow("", self.keyring_check)
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
        self.cli_preset_combo = QComboBox()
        for item in CLI_PROVIDERS:
            self.cli_preset_combo.addItem(item["label"], item)
        self.add_cli_btn = QPushButton("Add CLI Profile")
        self.add_cli_btn.setToolTip("Add a profile that runs your own signed-in command-line AI program (personal use only).")
        self.add_cli_btn.clicked.connect(self._on_add_cli_profile)
        if CLI_PROVIDERS:
            row_buttons.addWidget(self.cli_preset_combo)
            row_buttons.addWidget(self.add_cli_btn)
        self.cli_preset_combo.setVisible(len(CLI_PROVIDERS) > 1)
        self.add_cli_btn.setVisible(bool(CLI_PROVIDERS))
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
        self.discovery_label = QLabel("Discovered endpoint")
        apply_scan_row.addWidget(self.discovery_label)
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
        self.save_btn = save_btn = QPushButton("Save and Close")
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
        stripped = (text or "").strip()
        mode = QLineEdit.Normal if stripped.lower().startswith("env:") or stripped == KEYRING_PLACEHOLDER else QLineEdit.Password
        if self.api_key_input.echoMode() != mode:
            self.api_key_input.setEchoMode(mode)

    @Slot(str)
    def _on_scope_changed(self, value: str) -> None:
        if self._suspend_field_events:
            return
        self._apply_form_rules(self.provider_combo.currentText(), value)

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

    @Slot()
    def _on_add_cli_profile(self) -> None:
        item = self.cli_preset_combo.currentData()
        if not isinstance(item, dict):
            return
        name = str(item["name"])
        candidate, counter = name, 2
        while self._is_profile_name_in_use(candidate):
            candidate = f"{name}-{counter}"
            counter += 1
        self._profiles.append(
            {
                "name": candidate,
                "provider": item["id"],
                "scope": "cloud",
                "base_url": "",
                "api_key": "",
                "api_key_runtime": "",
                "default_model": item["default_model"],
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
            f"Added '{candidate}'. It runs your own signed-in CLI (personal use only). Tick the confirmation box, "
            "then press Test Connection."
        )

    @staticmethod
    def _cli_provider(provider: str) -> dict[str, str] | None:
        key = (provider or "").strip().lower()
        return next((item for item in CLI_PROVIDERS if item["id"] == key), None)

    def _apply_form_rules(self, provider: str, scope: str) -> bool:
        """Show only the rows that apply to this provider and scope. Returns True when the scope is unusual for the provider.

        Rows are hidden (and disabled), never reset to another provider or scope here; only values that cannot apply
        to a CLI provider are cleared.
        """
        provider = (provider or "").strip().lower()
        scope = (scope or "local").strip().lower()
        spec = PROVIDER_FORM.get(provider) or PROVIDER_FORM["openai_compatible"]
        offered = list(spec["offered"])  # type: ignore[arg-type]
        flagged = scope not in offered
        if flagged:
            offered.append(scope)
        if [self.scope_combo.itemText(i) for i in range(self.scope_combo.count())] != offered:
            self.scope_combo.blockSignals(True)
            try:
                self.scope_combo.clear()
                self.scope_combo.addItems(offered)
            finally:
                self.scope_combo.blockSignals(False)
        if self.scope_combo.currentText() != scope:
            self.scope_combo.blockSignals(True)
            try:
                self.scope_combo.setCurrentText(scope)
            finally:
                self.scope_combo.blockSignals(False)
        fields_by_scope: dict[str, frozenset[str]] = spec["fields"]  # type: ignore[assignment]
        fields = fields_by_scope.get(scope) or next(iter(fields_by_scope.values()))
        cli = self._cli_provider(provider)
        self.cli_note_label.setText(cli["note"] if cli else "")
        self._form.setRowVisible(self.cli_note_label, cli is not None)
        self.cloud_ack_check.setText(cli["ack"] if cli else CLOUD_ACK_TEXT)
        if cli is not None:  # nothing typed here can apply to a CLI profile
            self.base_url_input.clear()
            self.api_key_input.clear()
            self.temperature_input.clear()
            self.keyring_check.setChecked(False)
        if "cidrs" in fields and not (self.allowed_cidrs_input.text() or "").strip():
            self.allowed_cidrs_input.setText(DEFAULT_CIDRS_TEXT)
        if scope == "cloud":
            self.verify_tls_check.setChecked(True)  # hosted endpoints always verify certificates
        active = 0 <= self.profile_list.currentRow() < len(self._profiles) and not self._busy_keyring
        rows = {
            "scope": self.scope_combo,
            "base_url": self.base_url_input,
            "api_key": self.api_key_input,
            "keyring": self.keyring_check,
            "model": self.default_model_input,
            "timeout": self.timeout_input,
            "context": self.context_tokens_input,
            "max_output": self.max_output_input,
            "temperature": self.temperature_input,
            "cidrs": self.allowed_cidrs_input,
            "ack": self.cloud_ack_check,
            "tls": self.verify_tls_check,
            "concurrent": self.concurrent_check,
        }
        for key, widget in rows.items():
            visible = key in fields and (key != "keyring" or self._keyring_available)
            self._form.setRowVisible(widget, visible)
            widget.setEnabled(visible and active)
        self.scope_combo.setEnabled(active and len(offered) > 1)  # a fixed scope is shown but not editable
        for widget in (
            self.include_non_private_check,
            self.include_loopback_check,
            self.refresh_networks_btn,
            self.network_combo,
            self.scan_btn,
            self.discovery_label,
            self.scan_results_combo,
            self.apply_scan_btn,
        ):
            widget.setVisible("discovery" in fields)
        return flagged

    @Slot(str)
    def _on_provider_changed(self, value: str) -> None:
        if self._suspend_field_events:
            return
        provider = (value or "").strip().lower()
        spec = PROVIDER_FORM.get(provider)
        current_scope = (self.scope_combo.currentText() or "local").strip().lower()
        offered = spec["offered"] if spec else ()
        scope = current_scope if (not offered or current_scope in offered) else offered[0]  # type: ignore[index]
        self._apply_form_rules(provider, scope)
        if self._cli_provider(provider) is not None:
            return
        current_url = (self.base_url_input.text() or "").strip()
        if provider != "anthropic" and "api.anthropic.com" in current_url:
            current_url = ""
            self.base_url_input.clear()
        if provider == "anthropic":
            if not current_url or _needs_public_https(current_url):
                self.base_url_input.setText("https://api.anthropic.com")
            return
        if provider == "ollama":
            if not current_url or current_url in {"http://127.0.0.1:1234", "http://localhost:1234"}:
                self.base_url_input.setText("http://127.0.0.1:11434")
            return
        if provider == "lm_studio":
            if not current_url or current_url in {"http://127.0.0.1:11434", "http://localhost:11434"}:
                self.base_url_input.setText("http://127.0.0.1:1234")
            if (self.scope_combo.currentText() or "").strip().lower() == "lan" and "127.0.0.1" in self.base_url_input.text():
                self._apply_form_rules(provider, "local")
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
        self._schedule_key_delete(secret_store.ref_id_from(self._profiles[row].get("api_key")))
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
            has_ref = secret_store.ref_id_from(persisted_key) is not None and not runtime_key
            self.api_key_input.setText(KEYRING_PLACEHOLDER if has_ref else (runtime_key or persisted_key))
            self.keyring_check.setChecked(has_ref)
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
            saved_provider = str(profile.get("provider", "ollama"))
            saved_scope = str(profile.get("scope", "local"))
            unusual_scope = self._apply_form_rules(saved_provider, saved_scope)
        finally:
            self._suspend_field_events = False
        if unusual_scope:
            self._result_box.setPlainText(
                f"Note: the scope '{saved_scope}' is unusual for provider '{saved_provider}'. The saved settings are kept as they are."
            )

    @Slot()
    def _on_apply_profile(self, _checked: bool = False, then: Callable[[], None] | None = None) -> None:
        """Validate and apply the form; ``then`` runs after the profile was applied (it may wait for the keyring)."""
        if self._busy_keyring:
            return
        row = self.profile_list.currentRow()
        if row < 0 or row >= len(self._profiles):
            if then is not None:
                then()
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
        cli_provider = self._cli_provider(self.provider_combo.currentText())
        if cli_provider is not None:
            scope_value = "cloud"
            base_url_value = ""
            if not self.cloud_ack_check.isChecked():
                QMessageBox.warning(
                    self,
                    "Confirm cloud use",
                    "Tick the confirmation box: transcripts are sent through your signed-in CLI to the vendor.",
                )
                return
        elif scope_value == "cloud":
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
            "default_model": (self.default_model_input.text() or "").strip(),
            "timeout_seconds": timeout,
            "verify_tls": verify_tls,
            "enabled": self.enabled_check.isChecked(),
            "allow_concurrent_with_local_transcription": self.concurrent_check.isChecked(),
            "allowed_cidrs": cidr_values if scope_value == "lan" else [],
            "context_tokens": context_tokens,
            "max_output_tokens": max_output_tokens,
            "temperature": None if cli_provider is not None else temperature,
            "cloud_acknowledged": bool(self.cloud_ack_check.isChecked() and scope_value == "cloud"),
        }
        old_ref_id = secret_store.ref_id_from(self._profiles[row].get("api_key"))
        if cli_provider is not None:  # no key of any kind; a keyring entry the profile held goes with the switch
            self._schedule_key_delete(old_ref_id)
            self._commit_profile(row, old_name, profile, "", "", then)
        elif old_ref_id and entered_api_key == KEYRING_PLACEHOLDER:
            self._commit_profile(row, old_name, profile, f"keyring:{old_ref_id}", "", then)
        elif entered_api_key.lower().startswith("env:"):
            self._schedule_key_delete(old_ref_id)
            self._commit_profile(row, old_name, profile, entered_api_key, "", then)
        elif entered_api_key and self.keyring_check.isChecked() and self._keyring_available:
            self._store_key_then_commit(row, old_name, profile, entered_api_key, old_ref_id, then)
        else:  # a session-only key, or no key at all
            self._schedule_key_delete(old_ref_id)
            self._commit_profile(row, old_name, profile, "", entered_api_key, then)

    def _commit_profile(
        self,
        row: int,
        old_name: str,
        profile: dict[str, object],
        persisted_api_key: str,
        runtime_api_key: str,
        then: Callable[[], None] | None,
    ) -> None:
        profile["api_key"] = persisted_api_key
        profile["api_key_runtime"] = runtime_api_key
        new_name = str(profile["name"])
        scope_value = str(profile["scope"])
        base_url_value = str(profile["base_url"])
        self._profiles[row] = profile
        if self._default_profile and self._default_profile.strip().lower() == old_name.strip().lower():
            self._default_profile = new_name
        self._refresh_profile_list()
        self.profile_list.setCurrentRow(row)
        self._refresh_default_profile_combo()
        key_note = ""
        if runtime_api_key:
            key_note = " Using a session-only API key (not saved to disk)."
        elif secret_store.ref_id_from(persisted_api_key) is not None:
            key_note = " API key is stored in the system keyring."
        if scope_value == "local" and ("127.0.0.1" in base_url_value or "localhost" in base_url_value):
            self._result_box.setPlainText(
                f"Profile changes applied. Scope set to local for loopback endpoint.{key_note}"
            )
        else:
            self._result_box.setPlainText(f"Profile changes applied.{key_note}")
        if then is not None:
            then()

    def _store_key_then_commit(
        self,
        row: int,
        old_name: str,
        profile: dict[str, object],
        key: str,
        old_ref_id: str | None,
        then: Callable[[], None] | None,
    ) -> None:
        """Store the key off the GUI thread; the profile only points at the new ref once that succeeded."""
        self._set_keyring_busy(True)
        self._result_box.setPlainText("Storing the API key in the system keyring...")

        def done(ref_id: object) -> None:
            new_ref_id = str(ref_id)
            if self._closed:  # the dialog went away while the key was being stored: do not leave an orphan
                keyring_worker.start_task(keyring_worker.delete_key_task(new_ref_id))
                return
            self._created_refs.add(new_ref_id)
            self._set_keyring_busy(False)
            self._schedule_key_delete(old_ref_id)
            if not (0 <= row < len(self._profiles)):
                return
            self._commit_profile(row, old_name, profile, f"keyring:{new_ref_id}", "", then)

        def failed(message: str) -> None:
            if self._closed:
                return
            self._set_keyring_busy(False)
            self._result_box.setPlainText(f"The API key was not stored. {message}")

        keyring_worker.start_task(keyring_worker.store_key_task(key), on_done=done, on_error=failed)

    def _schedule_key_delete(self, ref_id: str | None) -> None:
        """Remove a keyring entry for a replaced, cleared or deleted key (deferred until the dialog is saved)."""
        if not ref_id:
            return
        if ref_id in self._created_refs:  # made in this session and never saved anywhere: safe to remove now
            self._created_refs.discard(ref_id)
            keyring_worker.start_task(keyring_worker.delete_key_task(ref_id))
        else:
            self._pending_deletes.add(ref_id)

    def _set_keyring_busy(self, busy: bool) -> None:
        self._busy_keyring = busy
        for widget in (self.profile_list, self.apply_btn, self.test_btn, self.save_btn, self.delete_btn, self.api_key_input, self.keyring_check):
            widget.setEnabled(not busy)
        if not busy:
            self._apply_form_rules(self.provider_combo.currentText(), self.scope_combo.currentText())

    @Slot(object)
    def _on_keyring_availability(self, available: object) -> None:
        self._keyring_available = bool(available)
        self._apply_form_rules(self.provider_combo.currentText(), self.scope_combo.currentText())

    @Slot()
    def _on_test_connection(self) -> None:
        if self._test_handle is not None:
            return
        self._on_apply_profile(then=self._start_connection_test)

    def _start_connection_test(self) -> None:
        row = self.profile_list.currentRow()
        if row < 0 or row >= len(self._profiles):
            return
        parsed_profiles = load_llm_profiles([self._profiles[row]])
        if not parsed_profiles:
            self._result_box.setPlainText("Profile is invalid. Check provider/scope/base URL fields.")
            return
        profile = parsed_profiles[0]
        self._test_button_label = self.test_btn.text()
        self.test_btn.setText("Testing...")
        self.test_btn.setEnabled(False)
        self._result_box.setPlainText("Testing connection...")
        self.setCursor(Qt.WaitCursor)
        self._test_handle = keyring_worker.start_task(
            lambda: run_connection_test(profile),
            on_done=self._on_test_finished,
            on_error=self._on_test_failed,
        )

    def _end_connection_test(self) -> None:
        self._test_handle = None
        self.unsetCursor()
        self.test_btn.setText(getattr(self, "_test_button_label", "Test Connection"))
        self.test_btn.setEnabled(self.profile_list.count() > 0 and not self._busy_keyring)

    def _on_test_failed(self, message: str) -> None:
        self._end_connection_test()
        self._result_box.setPlainText(f"Connection test could not run. {message}")

    def _on_test_finished(self, result: object) -> None:
        self._end_connection_test()
        self._show_test_result(result)

    def _show_test_result(self, result) -> None:  # noqa: ANN001
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
        # Ensure active edits are applied (and any keyring write has finished) before saving.
        self._on_apply_profile(then=self._finish_save)

    def _finish_save(self) -> None:
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

    def done(self, result: int) -> None:  # noqa: N802
        """Detach running tasks without waiting for them, then clean up keyring entries for this outcome."""
        self._closed = True
        for handle in (self._availability_handle, self._test_handle):
            if handle is not None:
                handle.cancel()
        refs = self._pending_deletes if result == QDialog.Accepted else self._created_refs
        for ref_id in list(refs):
            keyring_worker.start_task(keyring_worker.delete_key_task(ref_id))
        self._pending_deletes.clear()
        self._created_refs.clear()
        super().done(result)

    def keyPressEvent(self, event) -> None:  # noqa: ANN001
        if event.key() == Qt.Key_Escape:
            event.ignore()
            return
        super().keyPressEvent(event)
