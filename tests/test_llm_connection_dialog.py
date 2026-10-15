from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from PySide6.QtWidgets import QApplication

from services import AppConfig
from services.llm_connection_service import evaluate_profile_scope_policy, load_llm_profiles
from ui_qt.llm_connection_dialog import CLOUD_PRESETS, LLMConnectionsDialog


class ConnectionDialogCloudTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls._app = QApplication.instance() or QApplication([])

    def _dialog(self) -> LLMConnectionsDialog:
        probe = patch("ui_qt.keyring_worker.secret_store.is_available", return_value=False)
        probe.start()
        self.addCleanup(probe.stop)
        dialog = LLMConnectionsDialog(AppConfig())
        self.addCleanup(dialog.deleteLater)
        return dialog

    def _add(self, dialog: LLMConnectionsDialog, label: str) -> None:
        dialog.preset_combo.setCurrentIndex(dialog.preset_combo.findText(label))
        dialog._on_add_cloud_preset()

    def test_presets_cover_the_main_hosted_providers(self) -> None:
        labels = [str(p["label"]) for p in CLOUD_PRESETS]
        self.assertEqual(labels, ["Anthropic (Claude)", "OpenAI", "Google Gemini", "OpenRouter (many models)"])
        self.assertTrue(all(str(p["api_key"]).startswith("env:") for p in CLOUD_PRESETS))
        self.assertTrue(all(str(p["base_url"]).startswith("https://") for p in CLOUD_PRESETS))

    def test_adding_a_preset_creates_an_unacknowledged_cloud_profile(self) -> None:
        dialog = self._dialog()
        self._add(dialog, "Anthropic (Claude)")
        profile = dialog.profiles()[-1]
        self.assertEqual((profile["provider"], profile["scope"]), ("anthropic", "cloud"))
        self.assertEqual(profile["api_key"], "env:ANTHROPIC_API_KEY")
        self.assertFalse(profile["cloud_acknowledged"])
        self.assertTrue(profile["verify_tls"])
        self._add(dialog, "Anthropic (Claude)")
        self.assertEqual([p["name"] for p in dialog.profiles()], ["anthropic", "anthropic-2"])

    def test_cloud_scope_locks_tls_and_hides_lan_options(self) -> None:
        dialog = self._dialog()
        self._add(dialog, "OpenAI")
        self.assertTrue(dialog.verify_tls_check.isChecked())
        self.assertFalse(dialog.verify_tls_check.isEnabled())
        self.assertFalse(dialog.allowed_cidrs_input.isEnabled())
        self.assertFalse(dialog.cloud_ack_check.isHidden())

    def test_apply_requires_the_confirmation_for_cloud(self) -> None:
        dialog = self._dialog()
        self._add(dialog, "OpenAI")
        with patch("ui_qt.llm_connection_dialog.QMessageBox.warning") as warn:
            dialog._on_apply_profile()
        self.assertEqual(warn.call_count, 1)
        self.assertFalse(dialog.profiles()[0]["cloud_acknowledged"])

        dialog.cloud_ack_check.setChecked(True)
        dialog.api_key_input.setText("sk-session-key-123456")
        dialog._on_apply_profile()
        saved = dialog.profiles()[0]
        self.assertTrue(saved["cloud_acknowledged"])
        self.assertEqual(saved["api_key"], "")  # a typed key is never persisted
        parsed = load_llm_profiles(dialog.profiles())[0]
        self.assertEqual(evaluate_profile_scope_policy(parsed), (True, None, None))

    def test_cloud_requires_https(self) -> None:
        dialog = self._dialog()
        self._add(dialog, "OpenAI")
        dialog.cloud_ack_check.setChecked(True)
        dialog.base_url_input.setText("http://api.openai.com")
        with patch("ui_qt.llm_connection_dialog.QMessageBox.warning") as warn:
            dialog._on_apply_profile()
        self.assertEqual(warn.call_count, 1)

    def test_token_and_temperature_fields_round_trip_and_validate(self) -> None:
        dialog = self._dialog()
        self._add(dialog, "Anthropic (Claude)")
        dialog.cloud_ack_check.setChecked(True)
        dialog.context_tokens_input.setText("150,000")
        dialog.max_output_input.setText("6000")
        dialog.temperature_input.setText("0.4")
        dialog._on_apply_profile()
        saved = dialog.profiles()[0]
        self.assertEqual((saved["context_tokens"], saved["max_output_tokens"], saved["temperature"]), (150000, 6000, 0.4))
        dialog._on_profile_selected(0)
        self.assertEqual(dialog.context_tokens_input.text(), "150000")

        for field, bad in ((dialog.context_tokens_input, "lots"), (dialog.max_output_input, "5"),
                           (dialog.temperature_input, "9")):
            dialog._on_profile_selected(0)
            field.setText(bad)
            with patch("ui_qt.llm_connection_dialog.QMessageBox.warning") as warn:
                dialog._on_apply_profile()
            self.assertEqual(warn.call_count, 1, bad)
        self.assertEqual(dialog.profiles()[0]["context_tokens"], 150000)

    def test_empty_temperature_means_provider_default(self) -> None:
        dialog = self._dialog()
        self._add(dialog, "Anthropic (Claude)")
        dialog.cloud_ack_check.setChecked(True)
        dialog.temperature_input.setText("")
        dialog._on_apply_profile()
        self.assertIsNone(dialog.profiles()[0]["temperature"])
        self.assertIsNone(load_llm_profiles(dialog.profiles())[0].temperature)

    def test_choosing_the_anthropic_provider_switches_to_cloud(self) -> None:
        dialog = self._dialog()
        dialog._on_add_profile()
        dialog.provider_combo.setCurrentText("anthropic")
        self.assertEqual(dialog.scope_combo.currentText(), "cloud")
        self.assertEqual(dialog.base_url_input.text(), "https://api.anthropic.com")


if __name__ == "__main__":
    unittest.main()
