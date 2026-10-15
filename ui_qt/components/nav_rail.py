"""Navigation rail: a small, stable list of destinations (arrow keys move the selection)."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QButtonGroup, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

RAIL_WIDTH = 200
COLLAPSED_WIDTH = 52


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
        self._brand = brand
        self._collapsed = False
        self.header = QHBoxLayout()
        self.brand_label = QLabel(brand)
        self.brand_label.setProperty("type", "heading")
        self.header.addWidget(self.brand_label)
        self.header.addStretch(1)
        layout.addLayout(self.header)
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

    def set_current(self, key: str, *, emit: bool = True) -> None:
        """Select ``key``; with ``emit=False`` only the highlight changes (the caller shows the page itself)."""
        button = self._buttons.get(key)
        if button is None:
            return
        button.setChecked(True)
        if emit:
            self.selected.emit(key)

    def current(self) -> str:
        for key, button in self._buttons.items():
            if button.isChecked():
                return key
        return ""

    def button(self, key: str) -> QPushButton:
        return self._buttons[key]

    def buttons(self) -> list[QPushButton]:
        return [self._buttons[key] for key in self._keys]

    def is_collapsed(self) -> bool:
        return self._collapsed

    def set_collapsed(self, collapsed: bool) -> None:
        """Shrink to a narrow strip (brand initials only) or restore the full rail."""
        self._collapsed = collapsed
        self.setFixedWidth(COLLAPSED_WIDTH if collapsed else RAIL_WIDTH)
        self.brand_label.setText(self._brand[:2] if collapsed else self._brand)
        for button in self._buttons.values():
            button.setVisible(not collapsed)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key_Up, Qt.Key_Down) and self._keys:
            current = self.current()
            index = self._keys.index(current) if current in self._keys else 0
            index = (index + (1 if event.key() == Qt.Key_Down else -1)) % len(self._keys)
            self.set_current(self._keys[index])
            self._buttons[self._keys[index]].setFocus()
            return
        super().keyPressEvent(event)
