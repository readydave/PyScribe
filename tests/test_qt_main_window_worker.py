"""Regression tests for Qt transcription worker process control."""

from __future__ import annotations

import os
import queue
import unittest
from unittest.mock import patch

from PySide6.QtWidgets import QApplication

from services.model_service import RuntimeInfo
from ui_qt.main_window import TranscriptionWorker


class _FakeQueue:
    def get_nowait(self) -> object:
        raise queue.Empty

    def close(self) -> None:
        return None

    def join_thread(self) -> None:
        return None


class _FakeEvent:
    def __init__(self) -> None:
        self._set = False

    def is_set(self) -> bool:
        return self._set

    def set(self) -> None:
        self._set = True


class _FakeProcess:
    def __init__(
        self,
        *,
        alive_after_start: bool,
        exitcode: int | None,
        sticky_terminate: bool = False,
    ) -> None:
        self.pid = 43210
        self.exitcode = exitcode
        self._alive_after_start = alive_after_start
        self._alive = False
        self._sticky_terminate = sticky_terminate
        self.terminate_calls = 0
        self.kill_calls = 0

    def start(self) -> None:
        self._alive = self._alive_after_start

    def is_alive(self) -> bool:
        return self._alive

    def terminate(self) -> None:
        self.terminate_calls += 1
        if not self._sticky_terminate:
            self._alive = False
            if self.exitcode is None:
                self.exitcode = -15

    def kill(self) -> None:
        self.kill_calls += 1
        self._alive = False
        self.exitcode = -9

    def join(self, timeout: float | None = None) -> None:
        return None


class _FakeContext:
    def __init__(self, process: _FakeProcess) -> None:
        self._process = process

    def Queue(self) -> _FakeQueue:
        return _FakeQueue()

    def Event(self) -> _FakeEvent:
        return _FakeEvent()

    def Process(self, target: object, args: tuple[object, ...]) -> _FakeProcess:
        return self._process


class QtMainWindowWorkerTests(unittest.TestCase):
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

    def _build_worker(self) -> TranscriptionWorker:
        with patch("ui_qt.main_window.detect_runtime", return_value=self._runtime()):
            return TranscriptionWorker(
                media_path="/tmp/example.wav",
                model_name="base",
                run_mode="transcribe_only",
                use_diarization=False,
                diar_backend="off",
                max_speakers=None,
                use_visual_analysis=False,
                visual_profile="balanced",
                visual_ocr_backend="auto",
                visual_sample_seconds=1.0,
                language=None,
            )

    def test_stop_child_process_escalates_to_kill_when_terminate_is_sticky(self) -> None:
        worker = self._build_worker()
        process = _FakeProcess(alive_after_start=True, exitcode=None, sticky_terminate=True)
        process.start()

        stopped = worker._stop_child_process(process, reason="unit-test", wait_timeout=0.0)

        self.assertTrue(stopped)
        self.assertEqual(process.terminate_calls, 1)
        self.assertEqual(process.kill_calls, 1)
        self.assertFalse(process.is_alive())

    def test_run_emits_failure_when_child_exits_without_terminal_event(self) -> None:
        worker = self._build_worker()
        process = _FakeProcess(alive_after_start=False, exitcode=1)
        errors: list[str] = []
        finished: list[tuple[object, ...]] = []
        worker.failed.connect(errors.append)
        worker.finished.connect(lambda *args: finished.append(args))

        with patch("ui_qt.main_window.mp.get_context", return_value=_FakeContext(process)):
            worker.run()

        self.assertEqual(finished, [])
        self.assertEqual(errors, ["Worker process exited unexpectedly (exit code 1)."])


if __name__ == "__main__":
    unittest.main()
