"""Shared Qt test helper: close a MainWindow and let a deferred close finish.

MainWindow.closeEvent no longer blocks on the diarization probe thread; it hides the window and
finishes when the thread ends. Tests must wait for that, or a still-running QThread is destroyed
at interpreter exit ("QThread: Destroyed while thread is still running").
"""

from __future__ import annotations

import time

from PySide6.QtWidgets import QApplication


def close_and_drain(window, timeout: float = 30.0) -> None:
    window.close()
    end = time.monotonic() + timeout
    while getattr(window, "_close_pending", False) and time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.01)
