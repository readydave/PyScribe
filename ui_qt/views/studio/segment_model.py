"""List model over a project's timed segments."""

from __future__ import annotations

import bisect
from typing import Any

from PySide6.QtCore import QAbstractListModel, QModelIndex, QPersistentModelIndex, Qt, Signal

SegmentRole = Qt.UserRole + 1
StartRole = Qt.UserRole + 2
SpeakerRole = Qt.UserRole + 3
NameRole = Qt.UserRole + 4
TextRole = Qt.UserRole + 5

_UNKNOWN = ("", "S?", "?")


class SegmentModel(QAbstractListModel):
    text_edited = Signal(int, str)

    def __init__(self, parent: object | None = None) -> None:
        super().__init__(parent)
        self._segments: list[dict[str, Any]] = []
        self._starts: list[float] = []
        self._names: dict[str, str] = {}

    def set_segments(self, segments: list[dict[str, Any]], speaker_names: dict[str, str] | None = None) -> None:
        self.beginResetModel()
        self._segments = sorted((dict(s) for s in segments), key=lambda s: s["start"])
        self._starts = [s["start"] for s in self._segments]
        self._names = dict(speaker_names or {})
        self.endResetModel()

    def set_speaker_names(self, names: dict[str, str]) -> None:
        self._names = dict(names)
        if self._segments:
            self.dataChanged.emit(self.index(0), self.index(len(self._segments) - 1), [NameRole])

    def segments(self) -> list[dict[str, Any]]:
        return [dict(s) for s in self._segments]

    def segment(self, row: int) -> dict[str, Any]:
        return self._segments[row]

    def display_name(self, row: int) -> str:
        label = self._segments[row].get("speaker") or ""
        return "" if label in _UNKNOWN else self._names.get(label, label)

    def rowCount(self, parent: QModelIndex | QPersistentModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._segments)

    def data(self, index: QModelIndex | QPersistentModelIndex, role: int = Qt.DisplayRole) -> object:
        if not index.isValid() or not 0 <= index.row() < len(self._segments):
            return None
        segment = self._segments[index.row()]
        if role in (Qt.DisplayRole, Qt.EditRole, TextRole):
            return segment["text"]
        if role == SegmentRole:
            return segment
        if role == StartRole:
            return segment["start"]
        if role == SpeakerRole:
            return segment.get("speaker") or ""
        if role == NameRole:
            return self.display_name(index.row())
        return None

    def flags(self, index: QModelIndex | QPersistentModelIndex) -> Qt.ItemFlag:
        base = super().flags(index)
        return base | Qt.ItemIsEditable if index.isValid() else base

    def setData(self, index: QModelIndex | QPersistentModelIndex, value: object, role: int = Qt.EditRole) -> bool:  # noqa: N802
        if role != Qt.EditRole or not index.isValid():
            return False
        text = " ".join(str(value or "").split())
        if not text or text == self._segments[index.row()]["text"]:
            return False
        self._segments[index.row()]["text"] = text
        self.dataChanged.emit(index, index, [Qt.DisplayRole, TextRole])
        self.text_edited.emit(index.row(), text)
        return True

    def row_for_time(self, seconds: float) -> int:
        """Index of the segment playing at ``seconds`` (the last one that has started); -1 before the first."""
        if not self._starts:
            return -1
        return bisect.bisect_right(self._starts, seconds) - 1
