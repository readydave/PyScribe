from __future__ import annotations

import os
import threading
import time
import unittest
from unittest.mock import patch

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from services.config_service import AppConfig
from services.model_service import RuntimeInfo
from ui_qt import main_window as main_window_module
from ui_qt.main_window import MainWindow


class CloseDuringProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls._app = QApplication.instance() or QApplication([])

    def _build_window(self) -> tuple[MainWindow, threading.Event]:
        gate = threading.Event()
        self.addCleanup(gate.set)  # never leave a probe thread blocked after the test
        runtime = RuntimeInfo(device="cpu", compute_type="int8", gpu_name="N/A", vram_gb=0.0, cpu_count=8)
        patches = [
            patch("ui_qt.main_window.detect_runtime", return_value=runtime),
            patch("ui_qt.main_window.load_config", return_value=AppConfig()),
            patch("ui_qt.main_window.save_config"),
            patch("ui_qt.main_window.list_live_audio_inputs", return_value=[]),
            patch("ui_qt.main_window.get_diarization_backend_availability", side_effect=lambda **_: (gate.wait(20), {})[1]),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        window = MainWindow()
        window.show()
        QApplication.processEvents()
        self.assertTrue(window._diar_probe_running(), "the fake probe should be running")
        self.addCleanup(self._drain, window, gate)
        return window, gate

    def _drain(self, window: MainWindow, gate: threading.Event) -> None:
        gate.set()
        self._pump(lambda: window._diar_probe_thread is None)
        window.close()

    def _pump(self, done, timeout: float = 5.0) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            QApplication.processEvents()
            if done():
                return True
            time.sleep(0.01)
        return False

    def test_close_returns_quickly_and_hides_while_probe_runs(self) -> None:
        win, _gate = self._build_window()
        win.setup_dock.setFloating(True)
        win.setup_dock.show()
        started = time.monotonic()
        closed = win.close()
        elapsed = time.monotonic() - started
        self.assertFalse(closed)
        self.assertLess(elapsed, 0.5)
        self.assertFalse(win.isVisible())
        self.assertFalse(win.setup_dock.isVisible())
        self.assertTrue(win._diar_probe_running())
        self.assertFalse(win.close())  # a second close while pending is a no-op
        self.assertTrue(win._close_pending)

    def test_close_completes_after_probe_ends_with_worker_already_released(self) -> None:
        win, gate = self._build_window()
        QApplication.setQuitOnLastWindowClosed(True)
        seen: dict[str, object] = {}
        original = MainWindow._finish_deferred_close

        def spy(self_: MainWindow) -> None:
            seen["worker_was_none"] = self_._diar_probe_worker is None
            original(self_)

        with patch.object(MainWindow, "_finish_deferred_close", spy):
            win.close()
            gate.set()
            self.assertTrue(self._pump(lambda: win._close_finalized))
        self.assertEqual(seen, {"worker_was_none": True})
        self.assertFalse(win._close_pending)
        self.assertFalse(win._close_timer.isActive())
        self.assertTrue(QApplication.quitOnLastWindowClosed())

    def test_app_event_loop_quits_when_the_deferred_close_finishes(self) -> None:
        win, gate = self._build_window()
        QApplication.setQuitOnLastWindowClosed(True)
        win.close()
        QTimer.singleShot(200, gate.set)
        QTimer.singleShot(8000, lambda: self._app.exit(99))  # failsafe: the loop must end before this
        self.assertEqual(self._app.exec(), 0)
        self.assertTrue(win._close_finalized)

    def test_probe_finishing_during_close_event_does_not_strand_the_window(self) -> None:
        win, gate = self._build_window()

        def finish_probe_mid_close() -> None:
            gate.set()  # the probe ends while closeEvent is still saving state
            self.assertTrue(self._pump(lambda: win._diar_probe_thread is None))

        with patch.object(win, "_save_window_geometry", side_effect=finish_probe_mid_close), \
                patch.object(main_window_module, "DIAR_PROBE_CLOSE_TIMEOUT_MS", 3000), \
                patch.object(main_window_module.os, "_exit") as fake_exit:
            closed = win.close()
            self._pump(lambda: False, timeout=0.3)
        self.assertTrue(closed)  # nothing left to wait for, so the close is accepted at once
        self.assertFalse(win._close_pending)
        fake_exit.assert_not_called()

    def test_timeout_flushes_logs_and_exits_without_waiting_for_the_probe(self) -> None:
        win, _gate = self._build_window()
        with patch.object(main_window_module, "DIAR_PROBE_CLOSE_TIMEOUT_MS", 50), \
                patch.object(main_window_module.os, "_exit") as fake_exit, \
                patch.object(main_window_module.logging, "shutdown") as fake_shutdown:
            win.close()
            self.assertTrue(self._pump(lambda: fake_exit.called, timeout=3.0))
        fake_shutdown.assert_called_once()
        fake_exit.assert_called_once_with(0)

    def test_timeout_timer_is_stopped_after_a_clean_finish(self) -> None:
        win, gate = self._build_window()
        with patch.object(main_window_module, "DIAR_PROBE_CLOSE_TIMEOUT_MS", 300), \
                patch.object(main_window_module.os, "_exit") as fake_exit:
            win.close()
            gate.set()
            self.assertTrue(self._pump(lambda: win._close_finalized))
            self._pump(lambda: False, timeout=0.6)
        fake_exit.assert_not_called()

    def test_close_without_a_probe_still_closes_at_once(self) -> None:
        win, gate = self._build_window()
        gate.set()
        self.assertTrue(self._pump(lambda: win._diar_probe_thread is None))
        self.assertTrue(win.close())
        self.assertFalse(win._close_pending)


if __name__ == "__main__":
    unittest.main()
