"""Qt tests for the keyring option and the background Test Connection in the LLM dialogs (fake secret store only)."""

from __future__ import annotations

import os
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PySide6.QtWidgets import QApplication, QMessageBox

from qt_close import close_and_drain
from services import AppConfig
from services import secret_store as real_secret_store
from services.secret_store import SecretStoreError
from ui_qt import keyring_worker
from ui_qt.llm_connection_dialog import KEYRING_PLACEHOLDER, LLMConnectionsDialog
from ui_qt.llm_postprocess_dialog import LLMPostprocessDialog

SECRET = "sk-test-secret-value-123456"


class FakeStore:
    """Stands in for services.secret_store; never touches a real keyring."""

    SecretStoreError = SecretStoreError

    def __init__(self, available: bool = True, fail_set: bool = False) -> None:
        self.available = available
        self.fail_set = fail_set
        self.entries: dict[str, str] = {}
        self.deleted: list[str] = []
        self._counter = 0
        self._lock = threading.Lock()

    def is_available(self) -> bool:
        return self.available

    def new_ref(self) -> str:
        with self._lock:
            self._counter += 1
            return f"keyring:fake{self._counter:04d}"

    def ref_id_from(self, value: object) -> str | None:
        return real_secret_store.ref_id_from(value)

    def set_key(self, ref_id: str, key: str) -> None:
        if self.fail_set:
            raise SecretStoreError("The system keyring could not be used (is it unlocked and running?).")
        with self._lock:
            self.entries[ref_id] = key

    def delete_key(self, ref_id: str) -> None:
        with self._lock:
            self.entries.pop(ref_id, None)
            self.deleted.append(ref_id)


def _result(status: str = "pass") -> SimpleNamespace:
    return SimpleNamespace(
        status=status,
        provider="ollama",
        base_url="http://127.0.0.1:11434",
        selected_model="m1",
        loaded_model="",
        detected_models=["m1"],
        stages=[],
        failure_code="",
        failure_detail="",
    )


class _QtCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls._app = QApplication.instance() or QApplication([])

    def pump(self, condition, timeout: float = 5.0) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            QApplication.processEvents()
            if condition():
                return True
            time.sleep(0.01)
        QApplication.processEvents()
        return bool(condition())

    def drain_tasks(self) -> None:
        self.assertTrue(self.pump(lambda: keyring_worker.live_task_count() == 0), "background tasks did not finish")

    def use_store(self, store: FakeStore) -> None:
        for target in ("ui_qt.keyring_worker.secret_store", "ui_qt.llm_connection_dialog.secret_store"):
            patcher = patch(target, store)
            patcher.start()
            self.addCleanup(patcher.stop)


class KeyringDialogTests(_QtCase):
    def _dialog(self, store: FakeStore, profiles: list[dict] | None = None) -> LLMConnectionsDialog:
        self.use_store(store)
        config = AppConfig()
        if profiles is not None:
            config.llm_profiles = profiles
        dialog = LLMConnectionsDialog(config)
        self.addCleanup(self.drain_tasks)
        self.addCleanup(close_and_drain, dialog)
        return dialog

    @staticmethod
    def _profile(api_key: str = "") -> dict:
        return {
            "name": "cloudish",
            "provider": "ollama",
            "scope": "local",
            "base_url": "http://127.0.0.1:11434",
            "api_key": api_key,
            "api_key_runtime": "",
            "enabled": True,
        }

    def test_checkbox_is_hidden_when_keyring_unavailable(self) -> None:
        dialog = self._dialog(FakeStore(available=False))
        self.drain_tasks()
        self.assertTrue(dialog.keyring_check.isHidden())

    def test_checkbox_appears_when_keyring_is_available(self) -> None:
        dialog = self._dialog(FakeStore(available=True))
        self.assertTrue(self.pump(lambda: not dialog.keyring_check.isHidden()))

    def test_ticked_key_is_stored_and_profile_gets_a_ref_only(self) -> None:
        store = FakeStore()
        dialog = self._dialog(store)
        self.assertTrue(self.pump(lambda: not dialog.keyring_check.isHidden()))
        dialog._on_add_profile()
        dialog.api_key_input.setText(SECRET)
        dialog.keyring_check.setChecked(True)
        dialog._on_apply_profile()
        self.assertTrue(self.pump(lambda: str(dialog.profiles()[0]["api_key"]).startswith("keyring:")))
        profile = dialog.profiles()[0]
        self.assertEqual(profile["api_key_runtime"], "")
        self.assertNotIn(SECRET, str(profile))
        self.assertEqual(list(store.entries.values()), [SECRET])
        self.assertEqual(dialog.api_key_input.text(), KEYRING_PLACEHOLDER)
        self.assertNotIn(SECRET, dialog._result_box.toPlainText())

    def test_failed_store_leaves_profile_without_a_ref(self) -> None:
        store = FakeStore(fail_set=True)
        dialog = self._dialog(store)
        self.assertTrue(self.pump(lambda: not dialog.keyring_check.isHidden()))
        dialog._on_add_profile()
        dialog.api_key_input.setText(SECRET)
        dialog.keyring_check.setChecked(True)
        dialog._on_apply_profile()
        self.assertTrue(self.pump(lambda: "not stored" in dialog._result_box.toPlainText()))
        self.assertEqual(dialog.profiles()[0]["api_key"], "")
        self.assertNotIn(SECRET, dialog._result_box.toPlainText())
        self.assertTrue(dialog.apply_btn.isEnabled())

    def test_replacing_a_key_deletes_the_old_entry_only_after_save(self) -> None:
        store = FakeStore()
        store.entries["oldid"] = "old-key"
        dialog = self._dialog(store, [self._profile("keyring:oldid")])
        self.assertTrue(self.pump(lambda: not dialog.keyring_check.isHidden()))
        self.assertEqual(dialog.api_key_input.text(), KEYRING_PLACEHOLDER)
        dialog.api_key_input.setText(SECRET)
        dialog._on_apply_profile()
        self.assertTrue(self.pump(lambda: dialog.profiles()[0]["api_key"] != "keyring:oldid"))
        self.assertIn("oldid", store.entries)  # still there until the dialog is saved
        dialog._on_save_and_close()
        self.drain_tasks()
        self.assertNotIn("oldid", store.entries)
        self.assertEqual(len(store.entries), 1)

    def test_unchanged_placeholder_keeps_the_ref(self) -> None:
        store = FakeStore()
        store.entries["keepid"] = "kept"
        dialog = self._dialog(store, [self._profile("keyring:keepid")])
        dialog._on_apply_profile()
        self.assertEqual(dialog.profiles()[0]["api_key"], "keyring:keepid")
        dialog._on_save_and_close()
        self.drain_tasks()
        self.assertEqual(store.deleted, [])

    def test_cancel_removes_entries_made_in_this_session(self) -> None:
        store = FakeStore()
        dialog = self._dialog(store)
        self.assertTrue(self.pump(lambda: not dialog.keyring_check.isHidden()))
        dialog._on_add_profile()
        dialog.api_key_input.setText(SECRET)
        dialog.keyring_check.setChecked(True)
        dialog._on_apply_profile()
        self.assertTrue(self.pump(lambda: bool(store.entries)))
        dialog.reject()
        self.drain_tasks()
        self.assertEqual(store.entries, {})

    def test_deleting_a_profile_and_clearing_a_key_remove_entries_on_save(self) -> None:
        store = FakeStore()
        store.entries.update({"aaa": "k1", "bbb": "k2"})
        profiles = [self._profile("keyring:aaa"), {**self._profile("keyring:bbb"), "name": "second"}]
        dialog = self._dialog(store, profiles)
        dialog.profile_list.setCurrentRow(0)
        with patch("ui_qt.llm_connection_dialog.QMessageBox.question", return_value=QMessageBox.Yes):
            dialog._on_delete_profile()
        dialog.profile_list.setCurrentRow(0)
        dialog.api_key_input.setText("env:MY_KEY")
        dialog._on_apply_profile()
        self.assertEqual(store.entries, {"aaa": "k1", "bbb": "k2"})
        dialog._on_save_and_close()
        self.drain_tasks()
        self.assertEqual(store.entries, {})

    def test_test_connection_runs_in_background_and_shows_result(self) -> None:
        gate = threading.Event()
        self.addCleanup(gate.set)

        def slow_test(profile, cancel_event=None):  # noqa: ANN001
            gate.wait(10)
            return _result()

        dialog = self._dialog(FakeStore(available=False))
        dialog._on_add_profile()
        with patch("ui_qt.llm_connection_dialog.run_connection_test", slow_test):
            dialog._on_test_connection()
            self.assertFalse(dialog.test_btn.isEnabled())
            QApplication.processEvents()
            self.assertIn("Testing", dialog._result_box.toPlainText())
            gate.set()
            self.assertTrue(self.pump(lambda: "Overall: PASS" in dialog._result_box.toPlainText()))
        self.assertTrue(dialog.test_btn.isEnabled())

    def test_closing_mid_test_is_clean_and_drops_the_result(self) -> None:
        gate = threading.Event()
        self.addCleanup(gate.set)

        def slow_test(profile, cancel_event=None):  # noqa: ANN001
            gate.wait(10)
            return _result()

        dialog = self._dialog(FakeStore(available=False))
        dialog._on_add_profile()
        with patch("ui_qt.llm_connection_dialog.run_connection_test", slow_test):
            dialog._on_test_connection()
            start = time.monotonic()
            dialog.reject()
            self.assertLess(time.monotonic() - start, 1.0)  # closing never waits for the test
            self.assertGreaterEqual(keyring_worker.live_task_count(), 1)  # thread and worker outlive the dialog
            gate.set()
            self.drain_tasks()
        self.assertNotIn("Overall", dialog._result_box.toPlainText())


class DrainLiveTasksTests(_QtCase):
    def test_drain_returns_within_budget_and_cancels_callbacks(self) -> None:
        gate = threading.Event()
        self.addCleanup(gate.set)
        results: list[object] = []
        keyring_worker.start_task(lambda: gate.wait(10), on_done=results.append)
        self.assertTrue(keyring_worker._QUIT_HOOKED)  # aboutToQuit is connected on the first task
        start = time.monotonic()
        with self.assertLogs(keyring_worker.LOGGER, level="WARNING"):
            keyring_worker.drain_live_tasks(timeout_ms=300)
        self.assertLess(time.monotonic() - start, 1.5)
        gate.set()
        self.drain_tasks()
        self.assertEqual(results, [])  # cancelled before the result arrived

    def test_drain_waits_for_a_task_that_finishes_in_time(self) -> None:
        keyring_worker.start_task(lambda: time.sleep(0.1))
        keyring_worker.drain_live_tasks(timeout_ms=2000)
        self.drain_tasks()


class CooperativeCancelTests(_QtCase):
    def test_cancel_sets_the_event_the_task_received(self) -> None:
        seen: list[threading.Event] = []
        started = threading.Event()

        def task(event: threading.Event) -> None:
            seen.append(event)
            started.set()
            event.wait(10)

        handle = keyring_worker.start_task(task, cancel_event_arg=True)
        self.assertTrue(started.wait(5))
        self.assertFalse(seen[0].is_set())
        handle.cancel()
        self.assertTrue(seen[0].is_set())
        self.assertIs(seen[0], handle.cancel_event)
        self.drain_tasks()

    def test_cancel_aware_slow_task_ends_within_budget_and_drain_returns_true(self) -> None:
        keyring_worker.start_task(lambda event: event.wait(30), cancel_event_arg=True)
        start = time.monotonic()
        self.assertTrue(keyring_worker.drain_live_tasks(timeout_ms=2000))
        self.assertLess(time.monotonic() - start, 2.0)
        self.drain_tasks()

    def test_drain_returns_false_for_a_task_that_ignores_cancel(self) -> None:
        gate = threading.Event()
        self.addCleanup(gate.set)
        keyring_worker.start_task(lambda: gate.wait(10))
        with self.assertLogs(keyring_worker.LOGGER, level="WARNING"):
            self.assertFalse(keyring_worker.drain_live_tasks(timeout_ms=200))
        self.assertEqual(keyring_worker.running_task_count(), 1)
        gate.set()
        self.drain_tasks()


class ExitFallbackTests(_QtCase):
    def test_finished_but_unreleased_task_does_not_trigger_exit(self) -> None:
        from ui_qt import main_window

        handle = keyring_worker.start_task(lambda: None)
        handle._thread.wait(5000)  # finished, but no event loop runs here, so it is still in _LIVE
        self.assertGreaterEqual(keyring_worker.live_task_count(), 1)
        self.assertEqual(keyring_worker.running_task_count(), 0)
        exits: list[int] = []
        main_window._exit_if_tasks_running(0, exits.append)
        self.assertEqual(exits, [])
        self.drain_tasks()

    def test_running_task_triggers_exit_with_the_exit_code(self) -> None:
        from ui_qt import main_window

        gate = threading.Event()
        self.addCleanup(gate.set)
        keyring_worker.start_task(lambda: gate.wait(10))  # ignores cancel
        exits: list[int] = []
        with patch.object(main_window.logging, "shutdown") as shutdown:
            main_window._exit_if_tasks_running(3, exits.append, recheck_ms=50)
        self.assertEqual(exits, [3])
        shutdown.assert_called_once()
        gate.set()
        self.drain_tasks()

    def test_cancel_aware_task_stops_in_the_recheck_so_no_exit(self) -> None:
        from ui_qt import main_window

        keyring_worker.start_task(lambda event: event.wait(30), cancel_event_arg=True)
        exits: list[int] = []
        main_window._exit_if_tasks_running(0, exits.append, recheck_ms=1000)
        self.assertEqual(exits, [])
        self.drain_tasks()

    def test_run_qt_app_passes_the_exit_code_through(self) -> None:
        from ui_qt import main_window

        fake_app = SimpleNamespace(exec=lambda: 5)
        exits: list[int] = []
        gate = threading.Event()
        self.addCleanup(gate.set)
        keyring_worker.start_task(lambda: gate.wait(10))
        with patch.object(main_window, "QApplication") as qapp, patch.object(main_window, "MainWindow"), patch.object(
            main_window.logging, "shutdown"
        ):
            qapp.instance.return_value = fake_app
            main_window.run_qt_app(exit_fn=exits.append)
        self.assertEqual(exits, [5])
        gate.set()
        self.drain_tasks()


class PostprocessRefreshTests(_QtCase):
    def _dialog(self) -> LLMPostprocessDialog:
        config = AppConfig()
        config.llm_profiles = [
            {
                "name": "local-llm",
                "provider": "ollama",
                "scope": "local",
                "base_url": "http://127.0.0.1:11434",
                "enabled": True,
            }
        ]
        dialog = LLMPostprocessDialog(
            config=config,
            current_transcript_text="hello",
            current_ocr_text="",
            is_transcription_running=lambda: False,
        )
        self.addCleanup(self.drain_tasks)
        self.addCleanup(close_and_drain, dialog)
        return dialog

    def test_refresh_runs_in_background(self) -> None:
        gate = threading.Event()
        self.addCleanup(gate.set)

        def slow_test(profile, cancel_event=None):  # noqa: ANN001
            gate.wait(10)
            return _result()

        dialog = self._dialog()
        with patch("ui_qt.llm_postprocess_dialog.run_connection_test", slow_test):
            dialog._on_refresh_connection()
            self.assertFalse(dialog.refresh_btn.isEnabled())
            gate.set()
            self.assertTrue(self.pump(lambda: "PASS" in dialog.connection_status.text()))
        self.assertTrue(dialog.refresh_btn.isEnabled())

    def test_closing_mid_refresh_is_clean(self) -> None:
        gate = threading.Event()
        self.addCleanup(gate.set)

        def slow_test(profile, cancel_event=None):  # noqa: ANN001
            gate.wait(10)
            return _result()

        dialog = self._dialog()
        with patch("ui_qt.llm_postprocess_dialog.run_connection_test", slow_test):
            dialog._on_refresh_connection()
            start = time.monotonic()
            dialog.reject()
            self.assertLess(time.monotonic() - start, 1.0)
            self.assertGreaterEqual(keyring_worker.live_task_count(), 1)
            gate.set()
            self.drain_tasks()
        self.assertNotIn("PASS", dialog.connection_status.text())


if __name__ == "__main__":
    unittest.main()
