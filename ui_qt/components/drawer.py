"""Side drawer: a panel that slides open next to the main content (instant when reduced motion is on)."""

from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ui_qt.components.buttons import IconButton
from ui_qt.components.tokens import motion_ms


class Drawer(QFrame):
    visibility_changed = Signal(bool)

    def __init__(self, title: str, width: int = 360, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Drawer")
        self._width = width
        self._open = False
        self._animation = QPropertyAnimation(self, b"maximumWidth", self)
        self._animation.setEasingCurve(QEasingCurve.OutCubic)
        self._animation.finished.connect(self._on_animation_finished)
        self.setMinimumWidth(0)
        self.setMaximumWidth(0)
        self.setVisible(False)
        self.setAccessibleName(title)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 12, 16, 12)
        header = QHBoxLayout()
        self.title_label = QLabel(title)
        self.title_label.setProperty("type", "heading")
        header.addWidget(self.title_label, 1)
        self.close_button = IconButton("✕", "Close panel")
        self.close_button.clicked.connect(self.close_drawer)
        header.addWidget(self.close_button)
        outer.addLayout(header)
        self.body = QVBoxLayout()
        outer.addLayout(self.body, 1)

    def is_open(self) -> bool:
        return self._open

    def toggle(self) -> None:
        self.close_drawer() if self._open else self.open_drawer()

    def open_drawer(self) -> None:
        if self._open:
            return
        self._open = True
        self.setVisible(True)
        self._slide_to(self._width)
        self.visibility_changed.emit(True)

    def close_drawer(self) -> None:
        if not self._open:
            return
        self._open = False
        self._slide_to(0)
        self.visibility_changed.emit(False)

    def _slide_to(self, target: int) -> None:
        self._animation.stop()
        duration = motion_ms("base")
        if duration == 0:
            self.setMaximumWidth(target)
            self.setVisible(target > 0)
            return
        self._animation.setDuration(duration)
        self._animation.setStartValue(self.maximumWidth())
        self._animation.setEndValue(target)
        self._animation.start()

    def _on_animation_finished(self) -> None:
        if not self._open:
            self.setVisible(False)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Escape:
            self.close_drawer()
            return
        super().keyPressEvent(event)
