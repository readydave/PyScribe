"""Inspector tabs: one view open at a time, collapsible to give the main content more width."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QPushButton, QStackedWidget, QVBoxLayout, QWidget

from ui_qt.components.buttons import IconButton

COLLAPSED_WIDTH = 44
EXPANDED_MIN_WIDTH = 280


class InspectorTabs(QWidget):
    current_changed = Signal(str)
    collapsed_changed = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._keys: list[str] = []
        self._buttons: dict[str, QPushButton] = {}
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._collapsed = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)
        header = QHBoxLayout()
        header.setSpacing(4)
        self._tab_row = QHBoxLayout()
        self._tab_row.setSpacing(4)
        header.addLayout(self._tab_row, 1)
        self.collapse_button = IconButton("»", "Collapse inspector")
        self.collapse_button.clicked.connect(lambda: self.set_collapsed(not self._collapsed))
        header.addWidget(self.collapse_button)
        outer.addLayout(header)
        self.stack = QStackedWidget()
        outer.addWidget(self.stack, 1)
        self.setMinimumWidth(EXPANDED_MIN_WIDTH)

    def add_tab(self, key: str, label: str, widget: QWidget) -> None:
        button = QPushButton(label)
        button.setCheckable(True)
        button.setProperty("tab", True)
        button.clicked.connect(lambda _checked=False, k=key: self.set_current(k))
        self._group.addButton(button)
        self._tab_row.addWidget(button)
        self._buttons[key] = button
        self._keys.append(key)
        self.stack.addWidget(widget)
        if len(self._keys) == 1:
            self.set_current(key)

    def set_current(self, key: str) -> None:
        if key not in self._buttons:
            return
        self._buttons[key].setChecked(True)
        self.stack.setCurrentIndex(self._keys.index(key))
        if self._collapsed:
            self.set_collapsed(False)
        self.current_changed.emit(key)

    def current(self) -> str:
        index = self.stack.currentIndex()
        return self._keys[index] if 0 <= index < len(self._keys) else ""

    def is_collapsed(self) -> bool:
        return self._collapsed

    def set_collapsed(self, collapsed: bool) -> None:
        if collapsed == self._collapsed:
            return
        self._collapsed = collapsed
        self.stack.setVisible(not collapsed)
        for button in self._buttons.values():
            button.setVisible(not collapsed)
        self.collapse_button.setText("«" if collapsed else "»")
        self.collapse_button.setAccessibleName("Expand inspector" if collapsed else "Collapse inspector")
        self.collapse_button.setToolTip(self.collapse_button.accessibleName())
        self.setMinimumWidth(COLLAPSED_WIDTH if collapsed else EXPANDED_MIN_WIDTH)
        self.setMaximumWidth(COLLAPSED_WIDTH if collapsed else 16777215)
        self.collapsed_changed.emit(collapsed)
