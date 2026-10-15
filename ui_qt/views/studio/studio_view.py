"""Transcript Studio: header (title, search, export), transcript, inspector and the playback bar."""

from __future__ import annotations

import datetime

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from services.export_service import FORMATS
from services.project_store import Project
from ui_qt.components import AppButton, Chip, IconButton
from ui_qt.views.studio.inspector import StudioInspector
from ui_qt.views.studio.transcript_view import TranscriptView, format_clock
from ui_qt.views.studio.transport_bar import TransportBar

INSPECTOR_WIDTH = 400
COLLAPSED_WIDTH = 68


def subtitle_for(project: Project) -> str:
    parts = []
    try:
        when = datetime.datetime.fromisoformat(project.created)
        parts.append(f"{when.strftime('%a, %b')} {when.day}, {when.year}")
    except ValueError:
        pass
    if project.duration_s:
        parts.append(format_clock(project.duration_s))
    if project.speaker_count:
        parts.append(f"{project.speaker_count} speaker{'s' if project.speaker_count != 1 else ''}")
    return "  •  ".join(parts)


class StudioView(QWidget):
    back_requested = Signal()
    export_requested = Signal(str)
    rename_requested = Signal()
    reveal_requested = Signal()
    copy_requested = Signal()
    generate_requested = Signal(str)
    speaker_renamed = Signal(str, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.project_id: str | None = None
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = QHBoxLayout()
        header.setContentsMargins(20, 12, 20, 8)
        header.setSpacing(10)
        self.back_button = AppButton("← Projects", "ghost")
        self.back_button.clicked.connect(self.back_requested)
        header.addWidget(self.back_button)
        titles = QVBoxLayout()
        titles.setSpacing(0)
        self.title_label = QLabel("")
        self.title_label.setProperty("type", "heading")
        self.subtitle_label = QLabel("")
        self.subtitle_label.setProperty("type", "caption")
        titles.addWidget(self.title_label)
        titles.addWidget(self.subtitle_label)
        header.addLayout(titles, 1)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search transcript…  Ctrl+F")
        self.search.setClearButtonEnabled(True)
        self.search.setAccessibleName("Search transcript")
        self.search.setMinimumWidth(220)
        self.match_label = QLabel("")
        self.match_label.setProperty("type", "caption")
        self.match_label.setMinimumWidth(64)
        self.prev_button = IconButton("‹", "Previous match")
        self.next_button = IconButton("›", "Next match")
        for widget in (self.search, self.match_label, self.prev_button, self.next_button):
            header.addWidget(widget)
        self.local_chip = Chip("Local only", "success")
        self.local_chip.setToolTip("Transcription ran on this device. AI actions only send text where your connection profile points.")
        header.addWidget(self.local_chip)
        self.export_button = AppButton("Export", "primary")
        self.export_menu = QMenu(self.export_button)
        for fmt in FORMATS:
            self.export_menu.addAction(fmt.label, lambda key=fmt.key: self.export_requested.emit(key))
        self.export_button.setMenu(self.export_menu)
        header.addWidget(self.export_button)
        self.more_button = IconButton("⋮", "More actions")
        more = QMenu(self.more_button)
        more.addAction("Rename project…", self.rename_requested)
        more.addAction("Copy transcript", self.copy_requested)
        more.addAction("Show project folder", self.reveal_requested)
        self.more_button.setMenu(more)
        header.addWidget(self.more_button)
        root.addLayout(header)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.view = TranscriptView()
        self.empty_label = QLabel("This project has no transcript text.\nVisual analysis results, if any, are in the Visuals tab.")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setProperty("type", "caption")
        self.body = QStackedWidget()
        self.body.addWidget(self.view)
        self.body.addWidget(self.empty_label)
        self.inspector = StudioInspector()
        self.splitter.addWidget(self.body)
        self.splitter.addWidget(self.inspector)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)
        self.splitter.setSizes([700, INSPECTOR_WIDTH])
        root.addWidget(self.splitter, 1)

        self.transport = TransportBar()
        root.addWidget(self.transport)

        self.search.textChanged.connect(self.view.set_query)
        self.search.returnPressed.connect(self.view.next_match)
        self.next_button.clicked.connect(self.view.next_match)
        self.prev_button.clicked.connect(self.view.previous_match)
        self.view.matches_changed.connect(self._on_matches)
        self.view.seek_requested.connect(self._on_seek_from_text)
        self.view.toggle_play_requested.connect(self.transport.toggle)
        self.transport.position_changed.connect(self._on_position)
        self.inspector.generate_requested.connect(self.generate_requested)
        self.inspector.speaker_renamed.connect(self.speaker_renamed)
        self.inspector.collapsed_changed.connect(self._on_inspector_collapsed)
        QShortcut(QKeySequence.Find, self, activated=self.focus_search)
        QShortcut(QKeySequence(Qt.Key_Escape), self.search, activated=self._clear_search, context=Qt.WidgetShortcut)
        self._on_matches(0, 0)

    # --- loading ---------------------------------------------------------------------------------

    def load_project(self, project: Project, peaks: list[float] | None = None) -> None:
        self.project_id = project.id
        self.title_label.setText(project.title)
        self.subtitle_label.setText(subtitle_for(project))
        self.search.clear()
        self.view.load(project.segments, project.speaker_names)
        self.body.setCurrentIndex(0 if project.segments else 1)
        self.inspector.set_project(project)
        if not project.segments and project.visual_report:
            self.inspector.tabs.set_current("visuals")
        self.transport.load(project.media_path, int(project.duration_s * 1000))
        self.transport.set_peaks(peaks or [])
        self.export_button.setEnabled(bool(project.segments))

    def clear(self) -> None:
        self.project_id = None
        self.transport.unload("No recording is available for this project.")
        self.view.load([], {})

    def set_title(self, title: str) -> None:
        self.title_label.setText(title)

    def set_peaks(self, peaks: list[float]) -> None:
        self.transport.set_peaks(peaks)

    def refresh_speakers(self, project: Project) -> None:
        self.view.set_speaker_names(project.speaker_names)
        self.inspector.speakers.set_project(project)
        self.subtitle_label.setText(subtitle_for(project))

    def refresh_outputs(self, project: Project) -> None:
        self.inspector.update_outputs(project)

    def refresh_text_size(self) -> None:
        self.view.refresh_style()

    # --- playback sync ---------------------------------------------------------------------------

    def _on_seek_from_text(self, seconds: float) -> None:
        self.view.set_active_time(seconds, follow=False)
        self.transport.seek_ms(int(seconds * 1000))

    def _on_position(self, position_ms: int) -> None:
        self.view.set_active_time(position_ms / 1000.0, follow=self.transport.is_playing())

    def pause(self) -> None:
        self.transport.pause()

    def shutdown(self) -> None:
        self.transport.shutdown()

    # --- search ----------------------------------------------------------------------------------

    def focus_search(self) -> None:
        self.search.setFocus()
        self.search.selectAll()

    def _clear_search(self) -> None:
        self.search.clear()
        self.view.setFocus()

    def _on_matches(self, current: int, total: int) -> None:
        if not self.search.text().strip():
            self.match_label.setText("")
        else:
            self.match_label.setText(f"{current} / {total}" if total else "No matches")
        has_matches = total > 0
        self.prev_button.setEnabled(has_matches)
        self.next_button.setEnabled(has_matches)

    def _on_inspector_collapsed(self, collapsed: bool) -> None:
        total = sum(self.splitter.sizes()) or 1000
        side = COLLAPSED_WIDTH if collapsed else INSPECTOR_WIDTH
        self.splitter.setSizes([max(100, total - side), side])
