"""Waveform with a playhead: click or drag to seek, Left/Right keys step by 5 seconds."""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QKeyEvent, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from ui_qt import theme

STEP_MS = 5000


class WaveformWidget(QWidget):
    seek_requested = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._peaks: tuple[float, ...] = ()
        self._duration_ms = 0
        self._position_ms = 0
        self.setMinimumHeight(44)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAccessibleName("Waveform")
        self.setAccessibleDescription("Click to seek. Left and right arrow keys move five seconds.")
        self.setCursor(Qt.PointingHandCursor)

    # --- state -----------------------------------------------------------------------------------

    def set_peaks(self, peaks: Sequence[float]) -> None:
        self._peaks = tuple(peaks)
        self.update()

    def set_duration_ms(self, duration_ms: int) -> None:
        self._duration_ms = max(0, int(duration_ms))
        self.update()

    def set_position_ms(self, position_ms: int) -> None:
        self._position_ms = max(0, int(position_ms))
        self.update()

    def position_ms(self) -> int:
        return self._position_ms

    def duration_ms(self) -> int:
        return self._duration_ms

    def has_peaks(self) -> bool:
        return bool(self._peaks)

    def fraction(self) -> float:
        return min(1.0, self._position_ms / self._duration_ms) if self._duration_ms else 0.0

    # --- input -----------------------------------------------------------------------------------

    def _seek_to_x(self, x: float) -> None:
        if self._duration_ms <= 0 or self.width() <= 0:
            return
        self.seek_requested.emit(int(max(0.0, min(1.0, x / self.width())) * self._duration_ms))

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._seek_to_x(event.position().x())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.buttons() & Qt.LeftButton:
            self._seek_to_x(event.position().x())

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if self._duration_ms > 0 and event.key() in (Qt.Key_Left, Qt.Key_Right):
            delta = STEP_MS if event.key() == Qt.Key_Right else -STEP_MS
            self.seek_requested.emit(max(0, min(self._duration_ms, self._position_ms + delta)))
            return
        super().keyPressEvent(event)

    # --- paint -----------------------------------------------------------------------------------

    def paintEvent(self, _event: object) -> None:  # noqa: N802
        palette = theme.active_palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        bar, gap = 2.0, 1.5
        count = max(1, int(self.width() // (bar + gap)))
        mid = self.height() / 2
        played = self.fraction() * self.width()
        painter.setPen(Qt.NoPen)
        for index in range(count):
            x = index * (bar + gap)
            value = self._peaks[min(len(self._peaks) - 1, index * len(self._peaks) // count)] if self._peaks else 0.06
            half = max(1.0, value * (self.height() / 2 - 3))
            painter.setBrush(QColor(palette.accent if x <= played and self._peaks else palette.muted if self._peaks else palette.rule))
            painter.drawRoundedRect(QRectF(x, mid - half, bar, half * 2), 1, 1)
        if self._duration_ms > 0:
            painter.setPen(QPen(QColor(palette.ink), 2))
            painter.drawLine(int(played), 2, int(played), self.height() - 2)
        if self.hasFocus():
            painter.setPen(QPen(QColor(palette.ink), 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 4, 4)
        painter.end()
