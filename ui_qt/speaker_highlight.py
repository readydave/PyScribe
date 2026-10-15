"""Transcript decoration: coloured speaker labels, monospace timestamps, per-speaker margin rules.

Lines look like ``[S1] text`` or ``[S1] [00:01:02] text``; timestamps may also lead the line.
Lines without a label (wrapped or continuation lines) inherit the previous speaker's rule.
"""

from __future__ import annotations

import re

from PySide6.QtCore import QEvent, QObject, QPoint, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QPainter,
    QSyntaxHighlighter,
    QTextCharFormat,
    QTextCursor,
    QTextDocument,
)
from PySide6.QtWidgets import QPlainTextEdit, QWidget

from ui_qt import theme

_LABEL = re.compile(r"^\[(S(?:\d+|\?))\]")
_CLOCK = r"\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?"
_TIMESTAMP = re.compile(rf"\[{_CLOCK}(?:\s*[-–>]+\s*{_CLOCK})?\]")
_RULE_WIDTH = 3
_RULE_GUTTER = 10  # document margin so the rule sits left of the text


def _speaker_state(label: str) -> int:
    """Block state for a label: 1 for ``S?``, otherwise the speaker number + 1 (0 means none)."""
    digits = "".join(ch for ch in label if ch.isdigit())
    return int(digits) + 1 if digits else 1


class SpeakerHighlighter(QSyntaxHighlighter):
    """Colour-coded speaker labels and monospace timestamps; also records each block's speaker."""

    def __init__(self, document: QTextDocument) -> None:
        super().__init__(document)
        self._rules: _RuleOverlay | None = None
        editor = _owning_editor(document)
        if editor is not None:
            self.install_rules(editor)

    def install_rules(self, editor: QPlainTextEdit) -> None:
        """Draw per-speaker margin rules beside the text of ``editor`` (idempotent)."""
        if self._rules is None:
            editor.document().setDocumentMargin(_RULE_GUTTER)
            self._rules = _RuleOverlay(editor)

    def highlightBlock(self, text: str) -> None:  # noqa: N802
        match = _LABEL.match(text)
        if match:
            self.setCurrentBlockState(_speaker_state(match.group(1)))
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(theme.speaker_color(match.group(1))))
            fmt.setFontWeight(QFont.DemiBold)
            self.setFormat(0, match.end(), fmt)
        else:
            self.setCurrentBlockState(max(self.previousBlockState(), 0))
        mono = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        for stamp in _TIMESTAMP.finditer(text):
            tfmt = QTextCharFormat()
            tfmt.setFontFamilies([mono.family()])
            tfmt.setFontFixedPitch(True)
            tfmt.setForeground(QColor(theme.active_palette().muted))
            self.setFormat(stamp.start(), stamp.end() - stamp.start(), tfmt)


def _owning_editor(document: QTextDocument) -> QPlainTextEdit | None:
    node: QObject | None = document.parent()
    while node is not None:
        if isinstance(node, QPlainTextEdit):
            return node
        node = node.parent()
    return None


class _RuleOverlay(QWidget):
    """Transparent child of the editor viewport that paints a vertical rule per speaker block."""

    def __init__(self, editor: QPlainTextEdit) -> None:
        super().__init__(editor.viewport())
        self._editor = editor
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setGeometry(editor.viewport().rect())
        editor.viewport().installEventFilter(self)
        editor.updateRequest.connect(lambda *_: self.update())
        self.show()

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() == QEvent.Resize and isinstance(obj, QWidget):
            self.setGeometry(obj.rect())
        return False

    def paintEvent(self, event: object) -> None:  # noqa: N802
        editor = self._editor
        doc = editor.document()
        layout = doc.documentLayout()
        height = self.height()
        block = editor.cursorForPosition(QPoint(0, 0)).block()
        painter = QPainter(self)
        try:
            while block.isValid():
                top = editor.cursorRect(QTextCursor(block)).top()
                if top > height:
                    break
                state = block.userState()
                if state > 0 and block.isVisible():
                    label = "S?" if state == 1 else f"S{state - 1}"
                    rect = layout.blockBoundingRect(block)
                    color = QColor(theme.speaker_color(label))
                    painter.fillRect(
                        QRectF(_RULE_GUTTER / 2 - _RULE_WIDTH / 2, top, _RULE_WIDTH, rect.height()),
                        color,
                    )
                block = block.next()
        finally:
            painter.end()
