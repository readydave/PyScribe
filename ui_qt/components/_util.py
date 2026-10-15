"""Small helpers shared by the component library."""

from __future__ import annotations

from PySide6.QtWidgets import QWidget


def set_prop(widget: QWidget, name: str, value: object) -> None:
    """Set a dynamic property and re-polish the widget so the stylesheet picks it up."""
    if widget.property(name) == value:
        return
    widget.setProperty(name, value)
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()
