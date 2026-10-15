"""List row: title, optional subtitle and an optional trailing widget; activates on click, Enter or Space."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ui_qt.components.card import Card


class ListRow(Card):
    activated = Signal()

    def __init__(self, title: str, subtitle: str = "", trailing: QWidget | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent, interactive=True)
        self.setAccessibleName(title)
        row = QHBoxLayout(self)
        row.setContentsMargins(12, 8, 12, 8)
        text = QVBoxLayout()
        text.setSpacing(2)
        self.title_label = QLabel(title)
        self.title_label.setProperty("type", "label")
        text.addWidget(self.title_label)
        self.subtitle_label = QLabel(subtitle)
        self.subtitle_label.setProperty("type", "caption")
        self.subtitle_label.setVisible(bool(subtitle))
        text.addWidget(self.subtitle_label)
        row.addLayout(text, 1)
        if trailing is not None:
            row.addWidget(trailing)
        self.clicked.connect(self.activated)
