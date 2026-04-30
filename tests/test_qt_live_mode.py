"""Qt live mode UI tests."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PySide6.QtWidgets import QApplication, QMessageBox

from services import AppConfig
from services.live_vram_service import LiveVramPreflight
from services.model_service import RuntimeInfo
from ui_qt.main_window import MainWindow


class _FakeLiveSession:
    def __init__(self) -> None:
        self.session_dir = Path("/tmp/pyscribe-live/session-1")
        self.capture_path = self.session_dir / "2026-04-29_120000-live-capture.wav"
        self.options = SimpleNamespace(
            model_name="deepdml/faster-whisper-large-v3-turbo-ct2",
            use_diarization=False,
            diar_backend="off",
            max_speakers=None,
            language=None,
        )
        self.close_capture_calls = 0
        self.request_final_decode_calls = 0
        self.shutdown_calls = 0
        self.mark_finalizing_calls = 0
        self.finalize_cancelled_calls = 0
        self.finalize_failed_calls: list[str] = []
        self.finalize_success_calls: list[str] = []

    def poll_events(self) -> list[dict]:
        return []

    def is_idle(self) -> bool:
        return True

    def close_capture(self) -> None:
        self.close_capture_calls += 1

    def request_final_decode(self) -> None:
        self.request_final_decode_calls += 1

    def shutdown(self, preserve_error: bool = False) -> None:
        del preserve_error
        self.shutdown_calls += 1

    def mark_finalizing(self) -> None:
        self.mark_finalizing_calls += 1

    def finalize_cancelled(self) -> None:
        self.finalize_cancelled_calls += 1

    def finalize_failed(self, error_text: str) -> None:
        self.finalize_failed_calls.append(error_text)

    def finalize_success(self, transcript: str) -> None:
        self.finalize_success_calls.append(transcript)


class _FakeThread:
    def __init__(self, running: bool = True) -> None:
        self._running = running
        self.quit_calls = 0

    def isRunning(self) -> bool:
        return self._running

    def quit(self) -> None:
        self.quit_calls += 1
        self._running = False

    def wait(self, timeout: int) -> bool:
        del timeout
        self._running = False
        return True


class _FakeAudioSource:
    def __init__(self) -> None:
        self.suspend_calls = 0
        self.resume_calls = 0
        self.stop_calls = 0
        self.deleted = False

    def suspend(self) -> None:
        self.suspend_calls += 1

    def resume(self) -> None:
        self.resume_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1

    def deleteLater(self) -> None:
        self.deleted = True


class QtLiveModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls._app = QApplication.instance() or QApplication([])

    @staticmethod
    def _runtime() -> RuntimeInfo:
        return RuntimeInfo(
            device="cpu",
            compute_type="int8",
            gpu_name="N/A",
            vram_gb=0.0,
            cpu_count=8,
        )

    def _build_window(self, devices: list[object]) -> MainWindow:
        patches = [
            patch("ui_qt.main_window.detect_runtime", return_value=self._runtime()),
            patch("ui_qt.main_window.load_config", return_value=AppConfig()),
            patch("ui_qt.main_window.save_config", return_value=None),
            patch("ui_qt.main_window.list_live_audio_inputs", return_value=devices),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        window = MainWindow()
        window.show()
        QApplication.processEvents()
        self.addCleanup(window.close)
        return window

    def test_live_mode_toggle_updates_visibility(self) -> None:
        win = self._build_window(
            [SimpleNamespace(id="mic-1", name="Microphone", kind="microphone", available=True)]
        )

        self.assertTrue(win.drop_card.isVisible())
        self.assertFalse(win.live_card.isVisible())
        self.assertFalse(win.pause_live_btn.isVisible())

        win.input_mode_combo.setCurrentIndex(win.input_mode_combo.findData("live"))
        QApplication.processEvents()

        self.assertFalse(win.drop_card.isVisible())
        self.assertTrue(win.live_card.isVisible())
        self.assertTrue(win.stop_live_btn.isVisible())
        self.assertTrue(win.pause_live_btn.isVisible())
        self.assertFalse(win.pause_live_btn.isEnabled())
        self.assertEqual(win.transcribe_btn.text(), "Start Live")

    def test_loopback_mode_disables_start_without_loopback_device(self) -> None:
        win = self._build_window(
            [SimpleNamespace(id="mic-1", name="Microphone", kind="microphone", available=True)]
        )

        win.input_mode_combo.setCurrentIndex(win.input_mode_combo.findData("live"))
        win.live_source_combo.setCurrentIndex(win.live_source_combo.findData("loopback"))
        QApplication.processEvents()

        self.assertFalse(win.transcribe_btn.isEnabled())
        self.assertIn("No loopback input", win.live_guidance_label.text())

    def test_stop_live_capture_starts_final_post_pass(self) -> None:
        win = self._build_window(
            [SimpleNamespace(id="mic-1", name="Microphone", kind="microphone", available=True)]
        )
        win.input_mode_combo.setCurrentIndex(win.input_mode_combo.findData("live"))
        session = _FakeLiveSession()
        win._live_session = session
        win._live_capture_active = True
        win._live_finalizing = False

        with patch.object(win, "_launch_transcription_worker") as launch_worker:
            win.stop_live_capture()
            win._poll_live_session_events()

        self.assertEqual(session.close_capture_calls, 1)
        self.assertEqual(session.request_final_decode_calls, 1)
        self.assertEqual(session.shutdown_calls, 1)
        self.assertEqual(session.mark_finalizing_calls, 1)
        launch_worker.assert_called_once()
        win._live_finalizing = False
        win._live_session = None

    def test_pause_resume_live_capture_updates_button_state_and_timer(self) -> None:
        win = self._build_window(
            [SimpleNamespace(id="mic-1", name="Microphone", kind="microphone", available=True)]
        )
        win.input_mode_combo.setCurrentIndex(win.input_mode_combo.findData("live"))
        win._live_session = _FakeLiveSession()
        win._live_capture_active = True
        win._live_audio_source = _FakeAudioSource()
        win._live_started_at = 100.0
        win.cancel_btn.setEnabled(True)
        win.force_stop_btn.setEnabled(True)
        win._update_live_mode_ui()

        with patch("ui_qt.main_window.time.perf_counter", return_value=107.0):
            win.toggle_live_pause()

        self.assertTrue(win._live_paused)
        self.assertEqual(win._live_audio_source.suspend_calls, 1)
        self.assertEqual(win.pause_live_btn.text(), "Resume")
        self.assertEqual(win.status_label.text(), "Live capture paused.")
        self.assertTrue(win.stop_live_btn.isEnabled())
        self.assertTrue(win.force_stop_btn.isEnabled())

        with patch("ui_qt.main_window.time.perf_counter", return_value=113.0):
            win._update_live_elapsed_label()
        self.assertEqual(win.live_timer_label.text(), "00:00:07")

        with patch("ui_qt.main_window.time.perf_counter", return_value=113.0):
            win.toggle_live_pause()

        self.assertFalse(win._live_paused)
        self.assertEqual(win._live_audio_source.resume_calls, 1)
        self.assertEqual(win.pause_live_btn.text(), "Pause")
        self.assertEqual(win.status_label.text(), "Recording live audio...")

        with patch("ui_qt.main_window.time.perf_counter", return_value=116.0):
            win._update_live_elapsed_label()
        self.assertEqual(win.live_timer_label.text(), "00:00:10")

        win._teardown_live_session(shutdown=False)
        win._live_session = None

    def test_cancel_live_capture_confirmation_decline_preserves_session(self) -> None:
        win = self._build_window(
            [SimpleNamespace(id="mic-1", name="Microphone", kind="microphone", available=True)]
        )
        win.input_mode_combo.setCurrentIndex(win.input_mode_combo.findData("live"))

        cancel_session = _FakeLiveSession()
        win._live_session = cancel_session
        win._live_capture_active = True

        with patch("ui_qt.main_window.QMessageBox.question", return_value=QMessageBox.No) as question:
            win.cancel_transcription()

        question.assert_called_once()
        self.assertIs(win._live_session, cancel_session)
        self.assertTrue(win._live_capture_active)
        self.assertEqual(cancel_session.finalize_cancelled_calls, 0)
        self.assertNotIn("Cancellation requested.", win.terminal_log.toPlainText())

        win._live_capture_active = False
        win._live_session = None

    def test_cancel_and_force_stop_reset_live_ui_and_log_session_path(self) -> None:
        win = self._build_window(
            [SimpleNamespace(id="mic-1", name="Microphone", kind="microphone", available=True)]
        )
        win.input_mode_combo.setCurrentIndex(win.input_mode_combo.findData("live"))

        cancel_session = _FakeLiveSession()
        win._live_session = cancel_session
        win._live_capture_active = True
        win.transcript_text = "partial transcript"
        with patch("ui_qt.main_window.QMessageBox.question", return_value=QMessageBox.Yes):
            win.cancel_transcription()

        self.assertIsNone(win._live_session)
        self.assertEqual(cancel_session.finalize_cancelled_calls, 1)
        self.assertIn(str(cancel_session.session_dir), win.terminal_log.toPlainText())
        self.assertEqual(win.status_label.text(), "Cancelled.")

        force_session = _FakeLiveSession()
        win._live_session = force_session
        win._live_capture_active = True
        win.force_stop_transcription()

        self.assertIsNone(win._live_session)
        self.assertEqual(force_session.finalize_failed_calls, ["Live session force-stopped."])
        self.assertIn(str(force_session.session_dir), win.terminal_log.toPlainText())
        self.assertEqual(win.status_label.text(), "Force-stopped.")

    def test_live_final_pass_completion_reenables_controls(self) -> None:
        win = self._build_window(
            [SimpleNamespace(id="mic-1", name="Microphone", kind="microphone", available=True)]
        )
        win.input_mode_combo.setCurrentIndex(win.input_mode_combo.findData("live"))
        session = _FakeLiveSession()
        win._live_session = session
        win._live_finalizing = True
        win.worker_thread = _FakeThread(running=True)
        win.worker = object()
        win.transcribe_btn.setEnabled(False)
        win.input_mode_combo.setEnabled(False)

        win._on_worker_finished(False, "final text", "final text", "", 4.0, 0.0, 0.0)

        self.assertIsNone(win._live_session)
        self.assertFalse(win._live_finalizing)
        self.assertEqual(session.finalize_success_calls, ["final text"])
        self.assertTrue(win.transcribe_btn.isEnabled())
        self.assertTrue(win.input_mode_combo.isEnabled())
        self.assertEqual(win.status_label.text(), "Live transcription complete.")
        self.assertTrue(win.live_card.isVisible())
        self.assertTrue(win.stop_live_btn.isVisible())
        self.assertFalse(win.stop_live_btn.isEnabled())
        self.assertTrue(win.live_title_input.isEnabled())

        next_session = _FakeLiveSession()
        next_session.session_dir = Path("/tmp/pyscribe-live/session-2")
        win._live_session = next_session
        win._live_capture_active = True
        win._live_finalizing = False
        win.transcribe_btn.setEnabled(False)
        win.cancel_btn.setEnabled(True)
        win.force_stop_btn.setEnabled(True)
        win._update_live_mode_ui()

        try:
            self.assertTrue(win.live_card.isVisible())
            self.assertTrue(win.stop_live_btn.isVisible())
            self.assertTrue(win.stop_live_btn.isEnabled())
            self.assertTrue(win.pause_live_btn.isVisible())
            self.assertTrue(win.pause_live_btn.isEnabled())
            self.assertFalse(win.live_title_input.isEnabled())
            self.assertTrue(win.cancel_btn.isEnabled())
            self.assertEqual(win.transcribe_btn.text(), "Start Live")
        finally:
            win._live_capture_active = False
            win._live_session = None
            win._update_live_mode_ui()

    def test_live_vram_preflight_decline_cancels_start(self) -> None:
        win = self._build_window(
            [SimpleNamespace(id="mic-1", name="Microphone", kind="microphone", available=True)]
        )
        win.runtime = RuntimeInfo(
            device="cuda",
            compute_type="float16",
            gpu_name="Test GPU",
            vram_gb=12.0,
            cpu_count=8,
        )
        result = LiveVramPreflight(
            status="low",
            model_name="large-v3",
            estimated_required_gb=7.5,
            model_estimate_gb=6.5,
            safety_buffer_gb=1.0,
            free_gb=2.0,
            total_gb=12.0,
            used_gb=10.0,
            message="low vram",
        )

        with (
            patch("ui_qt.main_window.assess_live_vram_preflight", return_value=result),
            patch("ui_qt.main_window.QMessageBox.question", return_value=QMessageBox.No) as question,
        ):
            self.assertFalse(win._confirm_live_vram_preflight("large-v3"))

        question.assert_called_once()
        self.assertEqual(win.status_label.text(), "Live transcription canceled: not enough free GPU memory.")
        self.assertIn("low VRAM warning", win.terminal_log.toPlainText())

    def test_live_vram_preflight_continue_allows_start(self) -> None:
        win = self._build_window(
            [SimpleNamespace(id="mic-1", name="Microphone", kind="microphone", available=True)]
        )
        result = LiveVramPreflight(
            status="low",
            model_name="large-v3",
            estimated_required_gb=7.5,
            model_estimate_gb=6.5,
            safety_buffer_gb=1.0,
            free_gb=2.0,
            total_gb=12.0,
            used_gb=10.0,
            message="low vram",
        )

        with (
            patch("ui_qt.main_window.assess_live_vram_preflight", return_value=result),
            patch("ui_qt.main_window.QMessageBox.question", return_value=QMessageBox.Yes) as question,
        ):
            self.assertTrue(win._confirm_live_vram_preflight("large-v3"))

        question.assert_called_once()
        self.assertIn("continued live transcription", win.terminal_log.toPlainText())

    def test_live_vram_preflight_unavailable_does_not_prompt(self) -> None:
        win = self._build_window(
            [SimpleNamespace(id="mic-1", name="Microphone", kind="microphone", available=True)]
        )
        result = LiveVramPreflight(
            status="unavailable",
            model_name="large-v3",
            estimated_required_gb=7.5,
            model_estimate_gb=6.5,
            safety_buffer_gb=1.0,
            free_gb=None,
            total_gb=None,
            used_gb=None,
            message="Current GPU memory could not be read.",
        )

        with (
            patch("ui_qt.main_window.assess_live_vram_preflight", return_value=result),
            patch("ui_qt.main_window.QMessageBox.question") as question,
        ):
            self.assertTrue(win._confirm_live_vram_preflight("large-v3"))

        question.assert_not_called()


if __name__ == "__main__":
    unittest.main()
