"""Segmented control: a small exclusive set of choices rendered as joined buttons."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QPushButton, QWidget


class SegmentedControl(QWidget):
    changed = Signal(str)

    def __init__(self, options: list[tuple[str, str]], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._buttons: dict[str, QPushButton] = {}
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        last = len(options) - 1
        for index, (key, label) in enumerate(options):
            button = QPushButton(label)
            button.setCheckable(True)
            button.setProperty("segment", "left" if index == 0 else "right" if index == last else "mid")
            button.clicked.connect(lambda _checked=False, k=key: self.changed.emit(k))
            self._group.addButton(button)
            self._buttons[key] = button
            layout.addWidget(button)
        if options:
            self.set_current(options[0][0])

    def set_current(self, key: str) -> None:
        button = self._buttons.get(key)
        if button is not None:
            button.setChecked(True)

    def current(self) -> str:
        for key, button in self._buttons.items():
            if button.isChecked():
                return key
        return ""
