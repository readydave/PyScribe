"""Compact waveform strip for cards: draws normalised peaks, or a quiet baseline when there are none."""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QSizePolicy, QWidget

from ui_qt import theme


class MiniWaveform(QWidget):
    def __init__(self, peaks: Sequence[float] = (), parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._peaks: tuple[float, ...] = tuple(peaks)
        self.setFixedHeight(28)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

    def set_peaks(self, peaks: Sequence[float]) -> None:
        self._peaks = tuple(peaks)
        self.update()

    def paintEvent(self, _event: object) -> None:
        palette = theme.active_palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        bar, gap = 3.0, 2.0
        count = max(1, int(self.width() // (bar + gap)))
        mid = self.height() / 2
        color = QColor(palette.muted if self._peaks else palette.rule)
        painter.setPen(Qt.NoPen)
        painter.setBrush(color)
        for index in range(count):
            if self._peaks:
                value = self._peaks[min(len(self._peaks) - 1, index * len(self._peaks) // count)]
            else:
                value = 0.08
            half = max(1.0, value * (self.height() / 2 - 1))
            painter.drawRoundedRect(QRectF(index * (bar + gap), mid - half, bar, half * 2), 1.5, 1.5)
        painter.end()
