"""Buttons: primary, secondary, ghost and icon variants (styled by ``[variant]`` in the theme QSS)."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton, QWidget

from ui_qt.components._util import set_prop

VARIANTS = ("primary", "secondary", "ghost", "icon")


class AppButton(QPushButton):
    def __init__(self, text: str = "", variant: str = "secondary", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setCursor(Qt.PointingHandCursor)
        self.setProperty("variant", variant if variant in VARIANTS else "secondary")

    def set_variant(self, variant: str) -> None:
        set_prop(self, "variant", variant if variant in VARIANTS else "secondary")


class IconButton(AppButton):
    """A compact glyph button. ``accessible_name`` is required because there is no visible text label."""

    def __init__(self, glyph: str, accessible_name: str, parent: QWidget | None = None) -> None:
        super().__init__(glyph, "icon", parent)
        self.setAccessibleName(accessible_name)
        self.setToolTip(accessible_name)
        self.setFixedSize(32, 32)
