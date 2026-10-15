"""Thread-lifecycle helper: Python-derived QObjects must be destroyed on the main thread.

Destroying one on a worker thread (``deleteLater`` connected to the worker's own signal) takes a
Qt signal/slot mutex and then needs the GIL, which the GUI thread can hold while waiting on the
same mutex pool, so the app can deadlock. Workers are instead released from a slot on the owner's
(main) thread once the QThread has finished.
"""

from __future__ import annotations

from PySide6.QtCore import QThread

RELEASE_WAIT_MS = 1000


def release_worker(thread: QThread | None, wait_ms: int = RELEASE_WAIT_MS) -> bool:
    """Make sure ``thread`` has stopped before the caller drops its worker reference on the main thread.

    Safe to call twice, with ``None`` or a thread that never started. It never waits from inside the
    thread itself. Returns True when the thread is not running afterwards (so the worker may be dropped).
    """
    if thread is None:
        return True
    if not thread.isRunning():
        return True
    if QThread.currentThread() is thread:
        return False
    return bool(thread.wait(wait_ms))
