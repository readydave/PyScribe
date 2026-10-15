"""Qt tests: the LLM Connections form shows only the rows that apply to the chosen provider and scope."""

from __future__ import annotations

from qt_close import close_and_drain
from services import AppConfig
from test_llm_keyring_ui import FakeStore, _QtCase
from ui_qt.llm_connection_dialog import LLMConnectionsDialog

ROWS = {
    "base_url": "base_url_input",
    "api_key": "api_key_input",
    "keyring": "keyring_check",
    "temperature": "temperature_input",
    "cidrs": "allowed_cidrs_input",
    "ack": "cloud_ack_check",
    "tls": "verify_tls_check",
    "concurrent": "concurrent_check",
}
DISCOVERY = (
    "include_non_private_check",
    "include_loopback_check",
    "refresh_networks_btn",
    "network_combo",
    "scan_btn",
    "discovery_label",
    "scan_results_combo",
    "apply_scan_btn",
)
ALWAYS_VISIBLE_OUTSIDE_FORM = ("default_profile_combo", "_result_box", "add_btn", "apply_btn", "test_btn", "save_btn")
ALWAYS = ("default_model_input", "timeout_input", "context_tokens_input", "max_output_input", "scope_combo")

HTTP = {"base_url", "api_key", "keyring", "temperature"}
# (provider, scope) -> rows that must be visible (of the ROWS above)
EXPECTED = {
    ("ollama", "local"): HTTP | {"tls", "concurrent"},
    ("ollama", "lan"): HTTP | {"tls", "concurrent", "cidrs"},
    ("ollama", "cloud"): HTTP | {"ack"},
    ("lm_studio", "local"): HTTP | {"tls", "concurrent"},
    ("lm_studio", "lan"): HTTP | {"tls", "concurrent", "cidrs"},
    ("lm_studio", "cloud"): HTTP | {"ack"},
    ("openai_compatible", "local"): HTTP | {"tls", "concurrent"},
    ("openai_compatible", "lan"): HTTP | {"tls", "concurrent", "cidrs"},
    ("openai_compatible", "cloud"): HTTP | {"ack"},
    ("anthropic", "cloud"): HTTP | {"ack"},
    ("anthropic", "local"): HTTP | {"tls", "concurrent"},
    ("anthropic", "lan"): HTTP | {"tls", "concurrent", "cidrs"},
    ("claude_cli", "cloud"): {"ack"},
}


def _profile(provider: str, scope: str, **extra: object) -> dict:
    url = "" if provider == "claude_cli" else ("https://api.example.com" if scope == "cloud" else "http://192.168.10.50:1234")
    return {"name": "p", "provider": provider, "scope": scope, "base_url": url, "api_key": "", "enabled": True, **extra}


class ProviderFormTests(_QtCase):
    def _dialog(self, profiles: list[dict]) -> LLMConnectionsDialog:
        self.use_store(FakeStore(available=True))
        config = AppConfig()
        config.llm_profiles = profiles
        dialog = LLMConnectionsDialog(config)
        self.addCleanup(self.drain_tasks)
        self.addCleanup(close_and_drain, dialog)
        self.assertTrue(self.pump(lambda: dialog._keyring_available))
        return dialog

    def _visible(self, dialog: LLMConnectionsDialog) -> set[str]:
        return {key for key, attr in ROWS.items() if not getattr(dialog, attr).isHidden()}

    def test_rows_per_provider_and_scope(self) -> None:
        for (provider, scope), expected in EXPECTED.items():
            with self.subTest(provider=provider, scope=scope):
                dialog = self._dialog([_profile(provider, scope)])
                self.assertEqual(self._visible(dialog), expected)
                for attr in ALWAYS:
                    self.assertFalse(getattr(dialog, attr).isHidden(), attr)
                self.assertEqual(dialog.cli_note_label.isHidden(), provider != "claude_cli")

    def test_network_discovery_block_follows_provider_and_scope(self) -> None:
        for (provider, scope), expected in EXPECTED.items():
            shown = provider != "claude_cli" and scope in ("local", "lan")
            with self.subTest(provider=provider, scope=scope):
                dialog = self._dialog([_profile(provider, scope)])
                for attr in DISCOVERY:
                    self.assertEqual(not getattr(dialog, attr).isHidden(), shown, attr)
                for attr in ALWAYS_VISIBLE_OUTSIDE_FORM:
                    self.assertFalse(getattr(dialog, attr).isHidden(), attr)

    def test_switching_provider_toggles_the_discovery_block(self) -> None:
        dialog = self._dialog([_profile("ollama", "local")])
        dialog.provider_combo.setCurrentText("anthropic")
        self.assertTrue(dialog.scan_btn.isHidden())
        dialog.provider_combo.setCurrentText("lm_studio")
        self.assertFalse(dialog.scan_btn.isHidden())
        dialog.provider_combo.setCurrentText("claude_cli")
        self.assertTrue(dialog.apply_scan_btn.isHidden())
        dialog.provider_combo.setCurrentText("openai_compatible")
        dialog.scope_combo.setCurrentText("cloud")
        self.assertTrue(dialog.network_combo.isHidden())
        dialog.scope_combo.setCurrentText("lan")
        self.assertFalse(dialog.network_combo.isHidden())

    def test_hidden_rows_are_disabled_and_fixed_scope_is_not_editable(self) -> None:
        dialog = self._dialog([_profile("anthropic", "cloud")])
        self.assertFalse(dialog.allowed_cidrs_input.isEnabled())
        self.assertFalse(dialog.scope_combo.isEnabled())
        self.assertEqual(dialog.scope_combo.currentText(), "cloud")
        dialog = self._dialog([_profile("ollama", "local")])
        self.assertTrue(dialog.scope_combo.isEnabled())

    def test_switching_to_anthropic_replaces_a_lan_url_and_fixes_scope(self) -> None:
        dialog = self._dialog([_profile("ollama", "lan")])
        dialog.provider_combo.setCurrentText("anthropic")
        self.assertEqual(dialog.base_url_input.text(), "https://api.anthropic.com")
        self.assertEqual(dialog.scope_combo.currentText(), "cloud")
        self.assertEqual(self._visible(dialog), EXPECTED[("anthropic", "cloud")])

    def test_switching_to_anthropic_keeps_a_custom_https_url(self) -> None:
        dialog = self._dialog([_profile("openai_compatible", "cloud")])
        dialog.base_url_input.setText("https://llm-gateway.example.com")
        dialog.provider_combo.setCurrentText("anthropic")
        self.assertEqual(dialog.base_url_input.text(), "https://llm-gateway.example.com")

    def test_switching_away_from_anthropic_replaces_its_url(self) -> None:
        dialog = self._dialog([_profile("anthropic", "cloud", base_url="https://api.anthropic.com")])
        dialog.provider_combo.setCurrentText("ollama")
        self.assertEqual(dialog.base_url_input.text(), "http://127.0.0.1:11434")
        self.assertEqual(dialog.scope_combo.currentText(), "local")

    def test_ollama_and_lm_studio_offer_no_cloud_and_reset_a_cloud_scope(self) -> None:
        for provider in ("ollama", "lm_studio"):
            with self.subTest(provider=provider):
                dialog = self._dialog([_profile("openai_compatible", "cloud")])
                dialog.provider_combo.setCurrentText(provider)
                items = [dialog.scope_combo.itemText(i) for i in range(dialog.scope_combo.count())]
                self.assertEqual(items, ["local", "lan"])
                self.assertEqual(dialog.scope_combo.currentText(), "local")

    def test_stale_saved_profile_is_shown_as_saved_with_the_right_rows(self) -> None:
        saved = _profile("ollama", "cloud", allowed_cidrs=["10.0.0.0/8"], api_key="env:SOME_KEY")
        dialog = self._dialog([saved])
        self.assertEqual(dialog.profiles()[0]["scope"], "cloud")  # never changed on load
        self.assertEqual(dialog.profiles()[0]["provider"], "ollama")
        self.assertEqual(dialog.scope_combo.currentText(), "cloud")
        self.assertEqual(self._visible(dialog), EXPECTED[("ollama", "cloud")])
        self.assertIn("unusual", dialog._result_box.toPlainText())
        self.assertEqual(dialog.api_key_input.text(), "env:SOME_KEY")

    def test_save_drops_cidrs_unless_lan_and_keeps_optional_keys(self) -> None:
        dialog = self._dialog([_profile("ollama", "lan", allowed_cidrs=["10.0.0.0/8"], api_key="env:OPT_KEY")])
        dialog._on_apply_profile()
        self.assertEqual(dialog.profiles()[0]["allowed_cidrs"], ["10.0.0.0/8"])
        dialog.scope_combo.setCurrentText("local")
        dialog._on_apply_profile()
        saved = dialog.profiles()[0]
        self.assertEqual(saved["allowed_cidrs"], [])
        self.assertEqual(saved["api_key"], "env:OPT_KEY")  # an optional key is never dropped

    def test_lan_scope_prefills_default_cidrs_when_empty(self) -> None:
        dialog = self._dialog([_profile("ollama", "local")])
        dialog.scope_combo.setCurrentText("lan")
        self.assertIn("192.168.0.0/16", dialog.allowed_cidrs_input.text())

    def test_cli_profile_saves_no_temperature(self) -> None:
        dialog = self._dialog([_profile("claude_cli", "cloud", temperature=0.7)])
        dialog.cloud_ack_check.setChecked(True)
        dialog._on_apply_profile()
        saved = dialog.profiles()[0]
        self.assertIsNone(saved["temperature"])
        self.assertEqual((saved["base_url"], saved["api_key"]), ("", ""))
