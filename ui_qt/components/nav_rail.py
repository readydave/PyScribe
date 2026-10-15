"""Navigation rail: a small, stable list of destinations (arrow keys move the selection)."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QButtonGroup, QFrame, QLabel, QPushButton, QVBoxLayout, QWidget

RAIL_WIDTH = 188


class NavRail(QFrame):
    selected = Signal(str)

    def __init__(self, items: list[tuple[str, str]], brand: str = "PyScribe", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("NavRail")
        self.setFixedWidth(RAIL_WIDTH)
        self._keys = [key for key, _label in items]
        self._buttons: dict[str, QPushButton] = {}
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 16, 12, 12)
        layout.setSpacing(4)
        brand_label = QLabel(brand)
        brand_label.setProperty("type", "heading")
        layout.addWidget(brand_label)
        layout.addSpacing(12)
        for key, label in items:
            button = QPushButton(label)
            button.setCheckable(True)
            button.setProperty("nav", True)
            button.setFocusPolicy(Qt.StrongFocus)
            button.setCursor(Qt.PointingHandCursor)
            button.setAccessibleName(label)
            button.clicked.connect(lambda _checked=False, k=key: self.set_current(k))
            self._group.addButton(button)
            self._buttons[key] = button
            layout.addWidget(button)
        layout.addStretch(1)
        self.footer = QVBoxLayout()
        layout.addLayout(self.footer)

    def set_current(self, key: str) -> None:
        button = self._buttons.get(key)
        if button is None:
            return
        button.setChecked(True)
        self.selected.emit(key)

    def current(self) -> str:
        for key, button in self._buttons.items():
            if button.isChecked():
                return key
        return ""

    def button(self, key: str) -> QPushButton:
        return self._buttons[key]

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key_Up, Qt.Key_Down) and self._keys:
            current = self.current()
            index = self._keys.index(current) if current in self._keys else 0
            index = (index + (1 if event.key() == Qt.Key_Down else -1)) % len(self._keys)
            self.set_current(self._keys[index])
            self._buttons[self._keys[index]].setFocus()
            return
        super().keyPressEvent(event)
