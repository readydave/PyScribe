"""Card surface; optionally interactive (click, Enter or Space activates it) and selectable."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QFrame, QWidget

from ui_qt.components._util import set_prop


class Card(QFrame):
    clicked = Signal()

    def __init__(self, parent: QWidget | None = None, *, interactive: bool = False) -> None:
        super().__init__(parent)
        self.setProperty("studioCard", True)
        self.setProperty("interactive", interactive)
        self.setProperty("selected", False)
        self._interactive = interactive
        self._pressed = False
        if interactive:
            self.setFocusPolicy(Qt.StrongFocus)
            self.setCursor(Qt.PointingHandCursor)

    def is_interactive(self) -> bool:
        return self._interactive

    def set_selected(self, selected: bool) -> None:
        set_prop(self, "selected", bool(selected))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._pressed = self._interactive and event.button() == Qt.LeftButton
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._pressed and event.button() == Qt.LeftButton and self.rect().contains(event.position().toPoint()):
            self.clicked.emit()
        self._pressed = False
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if self._interactive and event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self.clicked.emit()
            return
        super().keyPressEvent(event)
