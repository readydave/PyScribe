"""Compact job status: current stage, elapsed time and cancel, shown only while work is active."""

from __future__ import annotations

import time

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QProgressBar, QWidget

from ui_qt.components import AppButton, tokens
from ui_qt.views.studio.transcript_view import format_clock

# What each stage is doing, in words (the timeline uses the short stage names).
ACTIVE_PHRASES = {
    "load": "Loading model",
    "transcribe": "Transcribing",
    "speakers": "Identifying speakers",
    "visuals": "Analyzing visuals",
    "save": "Saving",
}


class JobStatusBar(QFrame):
    details_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("JobStatusBar")
        self.setAccessibleName("Job status")
        self._started_at: float | None = None
        row = QHBoxLayout(self)
        row.setContentsMargins(16, 6, 16, 6)
        row.setSpacing(12)
        self.stage_label = QLabel("")
        self.stage_label.setProperty("type", "label")
        self.stage_label.setAccessibleName("Current stage")
        self.elapsed_label = QLabel("0:00")
        self.elapsed_label.setProperty("type", "caption")
        self.elapsed_label.setAccessibleName("Elapsed time")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(8)
        self.progress.setAccessibleName("Progress")
        self.details_button = AppButton("Details", "ghost")
        self.details_button.clicked.connect(self.details_requested)
        self.cancel_button = AppButton("Cancel", "secondary")
        self.cancel_button.setAccessibleName("Cancel job")
        row.addWidget(self.stage_label)
        row.addWidget(self.elapsed_label)
        row.addWidget(self.progress, 1)
        row.addWidget(self.details_button)
        row.addWidget(self.cancel_button)
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)
        self.setVisible(False)

    def show_job(self, stage_text: str, percent: int | None) -> None:
        """Show (or update) the bar. ``percent=None`` means the amount of work is unknown."""
        if self._started_at is None:
            self._started_at = time.monotonic()
            self._timer.start()
        self.stage_label.setText(stage_text)
        if percent is None:
            # Under reduced motion a busy bar would animate forever, so show it full and still instead.
            self.progress.setRange(0, 100)
            self.progress.setValue(100 if tokens.reduced_motion() else 0)
            if not tokens.reduced_motion():
                self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(max(0, min(100, int(percent))))
        self._tick()
        self.setVisible(True)

    def hide_job(self) -> None:
        self._timer.stop()
        self._started_at = None
        self.setVisible(False)

    def elapsed_seconds(self) -> float:
        return 0.0 if self._started_at is None else time.monotonic() - self._started_at

    def _tick(self) -> None:
        self.elapsed_label.setText(format_clock(self.elapsed_seconds()))
