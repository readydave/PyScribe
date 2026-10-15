"""Chip: a small pill label for status and tags."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QWidget

from ui_qt.components._util import set_prop

CHIP_VARIANTS = ("neutral", "accent", "success", "danger")


class Chip(QLabel):
    def __init__(self, text: str = "", variant: str = "neutral", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setProperty("chip", variant if variant in CHIP_VARIANTS else "neutral")

    def set_variant(self, variant: str) -> None:
        set_prop(self, "chip", variant if variant in CHIP_VARIANTS else "neutral")
