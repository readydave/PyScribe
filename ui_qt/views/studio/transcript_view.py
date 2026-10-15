"""Transcript list: timestamp, speaker badge and name, readable text; active-line highlight, search, inline edit."""

from __future__ import annotations

import html
import time

from PySide6.QtCore import QModelIndex, QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QKeyEvent, QPainter, QPen, QTextDocument, QWheelEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QListView,
    QPlainTextEdit,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QWidget,
)

from services.ui_themes import contrast
from ui_qt import theme
from ui_qt.components import tokens
from ui_qt.views.studio.segment_model import NameRole, SegmentModel, SpeakerRole, StartRole

PAD_X = 20
PAD_Y = 10
TIME_W = 58
BADGE = 26
GAP = 12
FOLLOW_PAUSE_S = 4.0


def format_clock(seconds: float) -> str:
    total = int(max(0.0, seconds))
    h, rest = divmod(total, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class SegmentEditor(QPlainTextEdit):
    """Inline editor: Enter saves, Escape cancels (Shift+Enter is not needed, segments are single paragraphs)."""

    commit_requested = Signal()
    cancel_requested = Signal()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.commit_requested.emit()
            return
        if event.key() == Qt.Key_Escape:
            self.cancel_requested.emit()
            return
        super().keyPressEvent(event)


class SegmentDelegate(QStyledItemDelegate):
    def __init__(self, view: TranscriptView) -> None:
        super().__init__(view)
        self._view = view
        self._heights: dict[tuple[int, int, int, str], int] = {}

    def clear_cache(self) -> None:
        self._heights.clear()

    def _text_x(self) -> int:
        return PAD_X + TIME_W + GAP + BADGE + GAP

    def _document(self, text: str, width: int, query: str) -> QTextDocument:
        palette = theme.active_palette()
        doc = QTextDocument()
        font = QFont(self._view.font())
        font.setPixelSize(tokens.transcript_px())
        doc.setDefaultFont(font)
        doc.setDocumentMargin(0)
        doc.setTextWidth(max(80, width))
        doc.setHtml(f'<span style="color:{palette.ink}">{self._highlight(text, query, palette.accent)}</span>')
        return doc

    @staticmethod
    def _highlight(text: str, query: str, color: str) -> str:
        if not query:
            return html.escape(text)
        out, lower, needle, start = [], text.casefold(), query.casefold(), 0
        while needle and (found := lower.find(needle, start)) != -1:
            out.append(html.escape(text[start:found]))
            out.append(f'<span style="background-color:{color}; color:#ffffff">{html.escape(text[found:found + len(needle)])}</span>')
            start = found + len(needle)
        out.append(html.escape(text[start:]))
        return "".join(out)

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:  # noqa: N802
        width = self._view.viewport().width() - self._text_x() - PAD_X
        key = (index.row(), width, tokens.transcript_px(), self._view.query)
        height = self._heights.get(key)
        if height is None:
            doc = self._document(str(index.data(Qt.DisplayRole) or ""), width, self._view.query)
            name_h = QFontMetrics(self._view.font()).height() + 4
            height = max(BADGE, name_h + int(doc.size().height())) + PAD_Y * 2
            self._heights[key] = height
        return QSize(self._view.viewport().width(), height)

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        palette = theme.active_palette()
        row, rect = index.row(), option.rect
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        if row == self._view.active_row:
            fill = QColor(palette.accent)
            fill.setAlpha(46)
            painter.fillRect(rect, fill)
        elif option.state & option.state.State_Selected:
            painter.fillRect(rect, QColor(palette.card))
        if row == self._view.current_match_row:
            painter.setPen(QPen(QColor(palette.accent), 2))
            painter.drawRect(rect.adjusted(1, 1, -1, -1))

        label = str(index.data(SpeakerRole) or "")
        name = str(index.data(NameRole) or "")
        color = QColor(theme.speaker_color(label) if label else palette.muted)
        small = QFont(self._view.font())
        small.setPixelSize(tokens.TYPE_PX["caption"] + 1)
        painter.setFont(small)
        painter.setPen(QColor(palette.muted))
        time_rect = QRect(rect.left() + PAD_X, rect.top() + PAD_Y, TIME_W, BADGE)
        painter.drawText(time_rect, Qt.AlignLeft | Qt.AlignVCenter, format_clock(float(index.data(StartRole) or 0)))

        badge_rect = QRect(time_rect.right() + GAP, rect.top() + PAD_Y, BADGE, BADGE)
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(badge_rect)
        initial = (name or label or "?")[:1].upper()
        on_color = "#FFFFFF" if contrast("#FFFFFF", color.name().upper()) >= contrast("#000000", color.name().upper()) else "#000000"
        bold = QFont(small)
        bold.setBold(True)
        painter.setFont(bold)
        painter.setPen(QColor(on_color))
        painter.drawText(badge_rect, Qt.AlignCenter, initial)

        text_x = rect.left() + self._text_x()
        text_width = rect.width() - self._text_x() - PAD_X
        name_h = QFontMetrics(self._view.font()).height() + 4
        if name:
            name_font = QFont(self._view.font())
            name_font.setPixelSize(tokens.TYPE_PX["label"])
            name_font.setBold(True)
            painter.setFont(name_font)
            painter.setPen(color)
            painter.drawText(QRect(text_x, rect.top() + PAD_Y, text_width, name_h), Qt.AlignLeft | Qt.AlignVCenter, name)
        doc = self._document(str(index.data(Qt.DisplayRole) or ""), text_width, self._view.query)
        painter.translate(QPoint(text_x, rect.top() + PAD_Y + (name_h if name else 0)))
        doc.drawContents(painter)
        painter.restore()

    def createEditor(self, parent: QWidget, option: QStyleOptionViewItem, index: QModelIndex) -> QWidget:  # noqa: N802
        editor = SegmentEditor(parent)
        editor.setFrameShape(SegmentEditor.NoFrame)
        font = QFont(self._view.font())
        font.setPixelSize(tokens.transcript_px())
        editor.setFont(font)
        editor.commit_requested.connect(lambda: self._finish(editor, commit=True))
        editor.cancel_requested.connect(lambda: self._finish(editor, commit=False))
        return editor

    def _finish(self, editor: QWidget, *, commit: bool) -> None:
        if commit:
            self.commitData.emit(editor)
        self.closeEditor.emit(editor, QStyledItemDelegate.NoHint)

    def setEditorData(self, editor: QWidget, index: QModelIndex) -> None:  # noqa: N802
        if isinstance(editor, QPlainTextEdit):
            editor.setPlainText(str(index.data(Qt.EditRole) or ""))
            editor.selectAll()

    def setModelData(self, editor: QWidget, model: SegmentModel, index: QModelIndex) -> None:  # noqa: N802
        if isinstance(editor, QPlainTextEdit):
            model.setData(index, editor.toPlainText(), Qt.EditRole)

    def updateEditorGeometry(self, editor: QWidget, option: QStyleOptionViewItem, index: QModelIndex) -> None:  # noqa: N802
        name_h = QFontMetrics(self._view.font()).height() + 4 if index.data(NameRole) else 0
        rect = option.rect
        editor.setGeometry(rect.left() + self._text_x() - 6, rect.top() + PAD_Y + name_h - 4, rect.width() - self._text_x() - PAD_X + 6, rect.height() - PAD_Y - name_h)


class TranscriptView(QListView):
    seek_requested = Signal(float)
    toggle_play_requested = Signal()
    matches_changed = Signal(int, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("TranscriptView")
        self.setAccessibleName("Transcript")
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.setUniformItemSizes(False)
        self.setLayoutMode(QListView.Batched)
        self.setBatchSize(200)
        self.setFrameShape(QListView.NoFrame)
        self.setMouseTracking(False)
        self._model = SegmentModel(self)
        self.setModel(self._model)
        self._delegate = SegmentDelegate(self)
        self.setItemDelegate(self._delegate)
        self.active_row = -1
        self.query = ""
        self.current_match_row = -1
        self._matches: list[int] = []
        self._match_pos = -1
        self._follow_resume_at = 0.0
        self.clicked.connect(self._on_clicked)
        self._model.text_edited.connect(lambda *_: self._relayout())

    # --- data ----------------------------------------------------------------------------------

    @property
    def segment_model(self) -> SegmentModel:
        return self._model

    def load(self, segments: list[dict], speaker_names: dict[str, str]) -> None:
        self.active_row = -1
        self._delegate.clear_cache()
        self._model.set_segments(segments, speaker_names)
        self._recompute_matches(select=False)

    def set_speaker_names(self, names: dict[str, str]) -> None:
        self._model.set_speaker_names(names)
        self._relayout()

    def refresh_style(self) -> None:
        """Re-measure rows after a text-size or theme change."""
        self._relayout()

    def _relayout(self) -> None:
        self._delegate.clear_cache()
        self.scheduleDelayedItemsLayout()
        self.viewport().update()

    def resizeEvent(self, event: object) -> None:  # noqa: N802
        self._delegate.clear_cache()
        super().resizeEvent(event)
        self.scheduleDelayedItemsLayout()

    # --- playback sync ---------------------------------------------------------------------------

    def set_active_time(self, seconds: float, *, follow: bool = True) -> None:
        row = self._model.row_for_time(seconds)
        if row == self.active_row:
            return
        previous, self.active_row = self.active_row, row
        for changed in (previous, row):
            if changed >= 0:
                self.viewport().update(self.visualRect(self._model.index(changed)))
        if follow and row >= 0 and time.monotonic() >= self._follow_resume_at:
            self.scrollTo(self._model.index(row), QAbstractItemView.PositionAtCenter)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        self._follow_resume_at = time.monotonic() + FOLLOW_PAUSE_S  # the reader is scrolling; stop following briefly
        super().wheelEvent(event)

    def _on_clicked(self, index: QModelIndex) -> None:
        if index.isValid():
            self.seek_requested.emit(float(index.data(StartRole) or 0.0))

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and self.state() != QAbstractItemView.EditingState:
            index = self.currentIndex()
            if index.isValid():
                self.seek_requested.emit(float(index.data(StartRole) or 0.0))
            return
        if event.key() == Qt.Key_Space and self.state() != QAbstractItemView.EditingState:
            self.toggle_play_requested.emit()
            return
        super().keyPressEvent(event)

    # --- search ----------------------------------------------------------------------------------

    def set_query(self, query: str) -> None:
        self.query = query.strip()
        self._delegate.clear_cache()
        self._recompute_matches(select=True)
        self.scheduleDelayedItemsLayout()
        self.viewport().update()

    def _recompute_matches(self, *, select: bool) -> None:
        needle = self.query.casefold()
        self._matches = (
            [r for r in range(self._model.rowCount()) if needle in str(self._model.index(r).data(Qt.DisplayRole)).casefold()]
            if needle
            else []
        )
        self._match_pos = 0 if self._matches else -1
        self._show_match(select)

    def _show_match(self, select: bool) -> None:
        self.current_match_row = self._matches[self._match_pos] if self._match_pos >= 0 else -1
        if select and self.current_match_row >= 0:
            index = self._model.index(self.current_match_row)
            self.setCurrentIndex(index)
            self.scrollTo(index, QAbstractItemView.PositionAtCenter)
        self.viewport().update()
        self.matches_changed.emit(self._match_pos + 1, len(self._matches))

    def next_match(self) -> None:
        if self._matches:
            self._match_pos = (self._match_pos + 1) % len(self._matches)
            self._show_match(True)

    def previous_match(self) -> None:
        if self._matches:
            self._match_pos = (self._match_pos - 1) % len(self._matches)
            self._show_match(True)

    def match_count(self) -> int:
        return len(self._matches)

