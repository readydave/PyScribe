"""Run blocking LLM-profile work (keyring calls, connection tests) off the GUI thread.

Each task owns a QThread (no event loop, so it ends when the task returns). It stays referenced in a module-level set until the thread
has finished, so closing a dialog mid-task never destroys a running QThread; the dialog just calls
``TaskHandle.cancel()`` so a late result is dropped. Handles live on the GUI thread, so results and the
release step are delivered there (never through ``deleteLater`` connected to the worker's own signal).
Keys are captured in the task callable only; they never travel in a signal.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QCoreApplication, QObject, QThread, QTimer, Signal, Slot

from services import secret_store
from services.secret_store import SecretStoreError
from ui_qt.thread_lifecycle import release_worker

LOGGER = logging.getLogger(__name__)

_LIVE: set["TaskHandle"] = set()
_QUIT_HOOKED = False

_GENERIC_ERROR = "The operation failed. See the application log for details."


class _Job(QThread):
    """Runs one callable in ``run()``; there is no event loop, so the thread ends as soon as the callable returns."""

    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, fn: Callable[[], Any]) -> None:
        super().__init__()
        self._fn: Callable[[], Any] | None = fn

    def run(self) -> None:  # noqa: D102 - QThread entry point
        fn, self._fn = self._fn, None  # drop the reference (it may hold a key) once taken
        try:
            result = fn() if fn is not None else None
        except SecretStoreError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            LOGGER.warning("Background task failed: %s", type(exc).__name__)
            self.failed.emit(_GENERIC_ERROR)
        else:
            self.succeeded.emit(result)
        finally:
            fn = None


class TaskHandle(QObject):
    """GUI-thread side of one background task."""

    def __init__(
        self,
        fn: Callable[[], Any],
        on_done: Callable[[Any], None] | None,
        on_error: Callable[[str], None] | None,
        pass_cancel_event: bool = False,
    ) -> None:
        super().__init__()
        self._on_done = on_done
        self._on_error = on_error
        self.cancel_event = threading.Event()  # set by cancel(); a cooperative task polls it
        call = (lambda: fn(self.cancel_event)) if pass_cancel_event else fn  # type: ignore[call-arg]
        self._thread: _Job | None = _Job(call)
        self._thread.succeeded.connect(self._handle_done)
        self._thread.failed.connect(self._handle_error)
        self._thread.finished.connect(self._release)

    def start(self) -> "TaskHandle":
        _LIVE.add(self)
        assert self._thread is not None
        self._thread.start()
        return self

    def cancel(self) -> None:
        """Drop the callbacks and ask the task to stop. The thread is not joined; it is released once it ends."""
        self.cancel_event.set()
        self._on_done = None
        self._on_error = None

    @Slot(object)
    def _handle_done(self, result: object) -> None:
        callback, self._on_done, self._on_error = self._on_done, None, None
        if callback is not None:
            callback(result)

    @Slot(str)
    def _handle_error(self, message: str) -> None:
        callback = self._on_error
        self._on_done = self._on_error = None
        if callback is not None:
            callback(message)

    @Slot()
    def _release(self) -> None:
        thread = self._thread
        if release_worker(thread):
            _LIVE.discard(self)
            self._thread = None
        else:  # thread still winding down; try again shortly
            QTimer.singleShot(100, self._release)


def start_task(
    fn: Callable[[], Any],
    on_done: Callable[[Any], None] | None = None,
    on_error: Callable[[str], None] | None = None,
    cancel_event_arg: bool = False,
) -> TaskHandle:
    """Run ``fn`` on a worker thread; callbacks run on the GUI thread unless the handle is cancelled.

    With ``cancel_event_arg`` the task is called as ``fn(event)``; the event is set when the handle is cancelled.
    """
    _hook_app_quit()
    return TaskHandle(fn, on_done, on_error, pass_cancel_event=cancel_event_arg).start()


def _hook_app_quit() -> None:
    """Connect drain_live_tasks to aboutToQuit once, so quitting the app mid-task never leaves a running QThread."""
    global _QUIT_HOOKED
    app = QCoreApplication.instance()
    if _QUIT_HOOKED or app is None:
        return
    app.aboutToQuit.connect(drain_live_tasks)
    _QUIT_HOOKED = True


def drain_live_tasks(timeout_ms: int = 2000) -> bool:
    """Cancel every live task and wait for its thread, sharing one time budget.

    Returns True when every thread has stopped; otherwise logs a warning and returns False.
    """
    deadline = time.monotonic() + max(0, timeout_ms) / 1000.0
    stopped = True
    for handle in list(_LIVE):
        handle.cancel()
        thread = handle._thread
        if thread is None or not thread.isRunning():
            continue
        remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
        if not thread.wait(remaining_ms):
            stopped = False
    if not stopped:
        LOGGER.warning("A background task was still running at shutdown (waited %d ms).", timeout_ms)
    return stopped


def running_task_count() -> int:
    """Threads that are really still running (finished ones may wait in _LIVE for a release that needs an event loop)."""
    return sum(1 for handle in list(_LIVE) if handle._thread is not None and handle._thread.isRunning())


def live_task_count() -> int:
    return len(_LIVE)


# --- task bodies (run on the worker thread) ---------------------------------------------------------

def check_available() -> Callable[[], bool]:
    return lambda: bool(secret_store.is_available())


def store_key_task(key: str) -> Callable[[], str]:
    """Store ``key`` under a fresh ref and return its id. Old refs are the caller's business (see the dialogs)."""

    def run() -> str:
        new = secret_store.new_ref()
        ref_id = secret_store.ref_id_from(new) or str(new)
        secret_store.set_key(ref_id, key)
        return ref_id

    return run


def delete_key_task(ref_id: str) -> Callable[[], None]:
    return lambda: secret_store.delete_key(ref_id)
