from __future__ import annotations

import os
import threading
import time
import unittest

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import QApplication

from ui_qt.thread_lifecycle import release_worker

_destroyed_on: list[int] = []


class _Worker(QObject):
    finished = Signal()

    @Slot()
    def run(self) -> None:
        self.finished.emit()

    def __del__(self) -> None:
        _destroyed_on.append(threading.get_ident())


class ThreadLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls._app = QApplication.instance() or QApplication([])

    def test_release_handles_none_and_unstarted_thread(self) -> None:
        self.assertTrue(release_worker(None))
        self.assertTrue(release_worker(QThread()))

    def test_release_is_safe_to_call_twice(self) -> None:
        thread = QThread()
        thread.start()
        self.assertFalse(release_worker(thread, wait_ms=20))  # still running: bounded wait gives up
        thread.quit()
        thread.wait(2000)
        self.assertTrue(release_worker(thread))
        self.assertTrue(release_worker(thread))

    def test_release_never_waits_from_inside_the_thread(self) -> None:
        thread = QThread()
        results: list[bool] = []
        thread.started.connect(lambda: results.append(release_worker(thread, wait_ms=5000)), type=Qt.DirectConnection)
        thread.started.connect(thread.quit, type=Qt.DirectConnection)
        thread.start()
        self.assertTrue(thread.wait(3000))
        self.assertEqual(results, [False])

    def test_worker_is_destroyed_on_the_main_thread(self) -> None:
        _destroyed_on.clear()
        holder = QObject()
        thread = QThread(holder)
        worker: _Worker | None = _Worker()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        state = {"worker": worker}
        worker = None

        def on_finished() -> None:
            release_worker(thread)
            state["worker"] = None  # dropped here, on the main thread

        thread.finished.connect(on_finished)
        thread.start()
        end = time.monotonic() + 5.0  # pump events directly: a stray quit() from another test must not end this wait
        while state["worker"] is not None and time.monotonic() < end:
            QApplication.processEvents()
            time.sleep(0.01)
        thread.wait(2000)
        self.assertIsNone(state["worker"])
        self.assertEqual(_destroyed_on, [threading.get_ident()])


if __name__ == "__main__":
    unittest.main()
