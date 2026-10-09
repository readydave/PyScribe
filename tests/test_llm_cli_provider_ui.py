"""Qt tests for the CLI provider (claude_cli) in the LLM Connections dialog; fake secret store only."""

from __future__ import annotations

import threading
from unittest.mock import patch

from PySide6.QtWidgets import QMessageBox

from qt_close import close_and_drain
from services import AppConfig
from ui_qt import llm_connection_dialog
from ui_qt.llm_connection_dialog import CLI_PROVIDERS, LLMConnectionsDialog
from test_llm_keyring_ui import FakeStore, _QtCase, _result


class CliProviderDialogTests(_QtCase):
    def _dialog(self, store: FakeStore | None = None, profiles: list[dict] | None = None) -> LLMConnectionsDialog:
        self.use_store(store or FakeStore(available=True))
        config = AppConfig()
        if profiles is not None:
            config.llm_profiles = profiles
        dialog = LLMConnectionsDialog(config)
        self.addCleanup(self.drain_tasks)
        self.addCleanup(close_and_drain, dialog)
        return dialog

    def test_provider_table_is_data_driven_and_claude_only(self) -> None:
        self.assertEqual([item["id"] for item in CLI_PROVIDERS], ["claude_cli"])
        dialog = self._dialog()
        items = [dialog.provider_combo.itemText(i) for i in range(dialog.provider_combo.count())]
        self.assertIn("claude_cli", items)
        self.assertNotIn("codex_cli", items)
        self.assertFalse(dialog.add_cli_btn.isHidden())

    def test_empty_table_hides_the_button_and_providers(self) -> None:
        with patch.object(llm_connection_dialog, "CLI_PROVIDERS", ()):
            dialog = self._dialog()
        self.assertTrue(dialog.add_cli_btn.isHidden())
        self.assertNotIn("claude_cli", [dialog.provider_combo.itemText(i) for i in range(dialog.provider_combo.count())])

    def test_add_cli_profile_disables_url_key_and_keyring_fields(self) -> None:
        dialog = self._dialog()
        self.assertTrue(self.pump(lambda: not dialog.keyring_check.isHidden()))
        dialog._on_add_cli_profile()
        profile = dialog.profiles()[0]
        self.assertEqual(profile["provider"], "claude_cli")
        self.assertEqual(profile["scope"], "cloud")
        self.assertEqual((profile["base_url"], profile["api_key"]), ("", ""))
        self.assertFalse(profile["cloud_acknowledged"])
        for widget in (dialog.base_url_input, dialog.api_key_input, dialog.keyring_check, dialog.scope_combo):
            self.assertFalse(widget.isEnabled())
        self.assertFalse(dialog.cli_note_label.isHidden())
        self.assertIn("personal use only", dialog.cli_note_label.text())
        self.assertIn("signed-in CLI", dialog.cloud_ack_check.text())

    def test_switching_back_reenables_fields_without_restoring_values(self) -> None:
        profiles = [{"name": "o", "provider": "ollama", "scope": "local", "base_url": "http://10.1.2.3:11434",
                     "api_key": "env:SOME_KEY", "enabled": True}]
        dialog = self._dialog(profiles=profiles)
        dialog.provider_combo.setCurrentText("claude_cli")
        self.assertEqual((dialog.base_url_input.text(), dialog.api_key_input.text()), ("", ""))
        dialog.provider_combo.setCurrentText("ollama")
        for widget in (dialog.base_url_input, dialog.api_key_input, dialog.scope_combo):
            self.assertTrue(widget.isEnabled())
        self.assertNotEqual(dialog.base_url_input.text(), "http://10.1.2.3:11434")
        self.assertEqual(dialog.api_key_input.text(), "")
        self.assertTrue(dialog.cli_note_label.isHidden())

    def test_apply_needs_the_confirmation_then_persists_no_key(self) -> None:
        dialog = self._dialog()
        dialog._on_add_cli_profile()
        with patch("ui_qt.llm_connection_dialog.QMessageBox.warning") as warn:
            dialog._on_apply_profile()
        self.assertEqual(warn.call_count, 1)
        self.assertIn("signed-in CLI", warn.call_args.args[2])
        dialog.cloud_ack_check.setChecked(True)
        dialog._on_apply_profile()
        saved = dialog.profiles()[0]
        self.assertTrue(saved["cloud_acknowledged"])
        self.assertEqual((saved["base_url"], saved["api_key"], saved["api_key_runtime"]), ("", "", ""))
        self.assertEqual(saved["scope"], "cloud")

    def test_switching_a_keyring_profile_to_cli_deletes_the_entry_only_on_save(self) -> None:
        for finish, expect_deleted in (("save", True), ("cancel", False)):
            with self.subTest(finish=finish):
                store = FakeStore()
                store.entries["refid"] = "stored-key"
                profile = {"name": "c", "provider": "anthropic", "scope": "cloud", "base_url": "https://api.anthropic.com",
                           "api_key": "keyring:refid", "cloud_acknowledged": True, "enabled": True}
                dialog = self._dialog(store, [profile])
                dialog.provider_combo.setCurrentText("claude_cli")
                dialog.cloud_ack_check.setChecked(True)
                dialog._on_apply_profile()
                self.assertEqual(dialog.profiles()[0]["api_key"], "")
                self.assertIn("refid", store.entries)  # nothing is removed before the dialog is saved
                if finish == "save":
                    dialog._on_save_and_close()
                else:
                    dialog.reject()
                self.drain_tasks()
                self.assertEqual("refid" not in store.entries, expect_deleted)

    def test_missing_binary_error_is_shown_after_background_test(self) -> None:
        gate = threading.Event()
        self.addCleanup(gate.set)
        failed = _result("fail")
        failed.failure_code = "cli_not_found"
        failed.failure_detail = "The 'claude' program was not found. Install Claude Code and sign in."
        failed.provider = "claude_cli"

        def fake_test(profile):  # noqa: ANN001
            gate.wait(10)
            return failed

        dialog = self._dialog()
        dialog._on_add_cli_profile()
        dialog.cloud_ack_check.setChecked(True)
        with patch("ui_qt.llm_connection_dialog.run_connection_test", fake_test):
            dialog._on_test_connection()
            self.assertFalse(dialog.test_btn.isEnabled())
            gate.set()
            self.assertTrue(self.pump(lambda: "was not found" in dialog._result_box.toPlainText()))
        self.assertTrue(dialog.test_btn.isEnabled())
