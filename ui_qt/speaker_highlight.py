"""Colours the speaker label at the start of each transcript line (``[S1] text``)."""

from __future__ import annotations

import re

from PySide6.QtGui import QColor, QFont, QSyntaxHighlighter, QTextCharFormat, QTextDocument

from ui_qt import theme

_LABEL = re.compile(r"^\[(S(?:\d+|\?))\]")


class SpeakerHighlighter(QSyntaxHighlighter):
    """Bold, colour-coded speaker labels; the colour follows the active light/dark theme."""

    def __init__(self, document: QTextDocument) -> None:
        super().__init__(document)

    def highlightBlock(self, text: str) -> None:  # noqa: N802
        match = _LABEL.match(text)
        if not match:
            return
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(theme.speaker_color(match.group(1))))
        fmt.setFontWeight(QFont.DemiBold)
        self.setFormat(0, match.end(), fmt)
