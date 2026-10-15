"""Empty-state block: a title, a short explanation and one primary action."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ui_qt.components.buttons import AppButton


class EmptyState(QWidget):
    action_clicked = Signal()

    def __init__(self, title: str, message: str, action_text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(8)
        self.title_label = QLabel(title)
        self.title_label.setProperty("type", "heading")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.message_label = QLabel(message)
        self.message_label.setProperty("type", "caption")
        self.message_label.setAlignment(Qt.AlignCenter)
        self.message_label.setWordWrap(True)
        layout.addWidget(self.title_label)
        layout.addWidget(self.message_label)
        self.action_button: AppButton | None = None
        if action_text:
            self.action_button = AppButton(action_text, "primary")
            self.action_button.clicked.connect(self.action_clicked)
            layout.addWidget(self.action_button, alignment=Qt.AlignCenter)
