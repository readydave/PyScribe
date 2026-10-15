"""Projects Home: start new work, jump back into recent projects, with the local-processing promise in the footer."""

from __future__ import annotations

import datetime
import getpass

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from services.project_store import ProjectSummary
from ui_qt.components import AppButton, Card, Chip, EmptyState, IconButton
from ui_qt.components.mini_waveform import MiniWaveform

RECENT_LIMIT = 6
COLUMNS = 3
STATUS_CHIPS = {
    "completed": ("Completed", "success"),
    "transcribing": ("Transcribing…", "accent"),
    "failed": ("Failed", "danger"),
    "cancelled": ("Cancelled", "neutral"),
}
KIND_GLYPH = {"file": "▤", "batch": "☰", "live": "●"}


def greeting(now: datetime.datetime | None = None, user: str | None = None) -> str:
    """'Good afternoon, Dave' style greeting; the name is dropped when the login name is not a person's."""
    hour = (now or datetime.datetime.now()).hour
    part = "morning" if hour < 12 else "afternoon" if hour < 18 else "evening"
    if user is None:
        try:
            user = getpass.getuser()
        except Exception:
            user = ""
    name = user.strip().split(" ")[0].title() if user and user.lower() not in {"root", "user", "admin"} else ""
    return f"Good {part}, {name}" if name else f"Good {part}"


def format_duration(seconds: float) -> str:
    total = int(max(0, seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def format_activity(iso: str, now: datetime.datetime | None = None) -> str:
    """'Today 2:14 PM', 'Yesterday 11:03 AM' or 'Jun 12, 2:14 PM' from an ISO timestamp."""
    try:
        when = datetime.datetime.fromisoformat(iso)
    except ValueError:
        return ""
    today = (now or datetime.datetime.now()).date()
    clock = when.strftime("%I:%M %p").lstrip("0")
    if when.date() == today:
        return f"Today {clock}"
    if (today - when.date()).days == 1:
        return f"Yesterday {clock}"
    return f"{when.strftime('%b')} {when.day}, {clock}"


def describe(summary: ProjectSummary) -> str:
    parts = [format_duration(summary.duration_s)] if summary.duration_s else []
    if summary.speaker_count:
        parts.append(f"{summary.speaker_count} speaker{'s' if summary.speaker_count != 1 else ''}")
    return "  •  ".join(parts)


class DropCard(Card):
    """The large primary action: click to choose a file, or drop audio/video onto it."""

    file_dropped = Signal(str)
    browse_requested = Signal()

    def __init__(self) -> None:
        super().__init__(interactive=True)
        self.setAcceptDrops(True)
        self.setObjectName("homeDropCard")
        self.setProperty("primaryAction", True)
        self.setMinimumHeight(170)
        self.setAccessibleName("New transcription")
        self.setAccessibleDescription("Drop an audio or video file here, or press Enter to choose one.")
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(6)
        title = QLabel("New transcription")
        title.setProperty("type", "heading")
        title.setAlignment(Qt.AlignCenter)
        hint = QLabel("Drag and drop audio or video files here\nor click to choose")
        hint.setProperty("type", "caption")
        hint.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        layout.addWidget(hint)
        self.clicked.connect(self.browse_requested)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls() and any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()
            self.set_selected(True)
            return
        event.ignore()

    def dragLeaveEvent(self, event: object) -> None:  # noqa: N802
        self.set_selected(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        self.set_selected(False)
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            self.file_dropped.emit(paths[0])
        event.acceptProposedAction()


class ProjectCard(Card):
    """One recent project: title, facts, waveform cue, status and a menu."""

    open_requested = Signal(str)
    rename_requested = Signal(str)
    reveal_requested = Signal(str)
    delete_requested = Signal(str)

    def __init__(self, summary: ProjectSummary) -> None:
        super().__init__(interactive=True)
        self.project_id = summary.id
        self.setAccessibleName(summary.title)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(8)

        top = QHBoxLayout()
        glyph = QLabel(KIND_GLYPH.get(summary.kind, "▤"))
        glyph.setProperty("type", "heading")
        top.addWidget(glyph)
        text = QVBoxLayout()
        text.setSpacing(1)
        self.title_label = QLabel(summary.title)
        self.title_label.setProperty("type", "label")
        self.title_label.setToolTip(summary.title)
        self.facts_label = QLabel(describe(summary))
        self.facts_label.setProperty("type", "caption")
        text.addWidget(self.title_label)
        text.addWidget(self.facts_label)
        top.addLayout(text, 1)
        self.menu_button = IconButton("⋯", f"Actions for {summary.title}")
        self.menu_button.clicked.connect(self._show_menu)
        top.addWidget(self.menu_button, alignment=Qt.AlignTop)
        outer.addLayout(top)

        outer.addWidget(MiniWaveform(summary.preview_peaks))

        bottom = QHBoxLayout()
        label, variant = STATUS_CHIPS.get(summary.status, STATUS_CHIPS["completed"])
        self.status_chip = Chip(label, variant)
        bottom.addWidget(self.status_chip)
        bottom.addStretch(1)
        self.activity_label = QLabel(format_activity(summary.updated))
        self.activity_label.setProperty("type", "caption")
        bottom.addWidget(self.activity_label)
        outer.addLayout(bottom)

        self.clicked.connect(lambda: self.open_requested.emit(self.project_id))

    def set_progress(self, percent: int) -> None:
        self.status_chip.setText(f"Transcribing… {max(0, min(100, percent))}%")

    def _show_menu(self) -> None:
        menu = QMenu(self)
        menu.addAction("Open", lambda: self.open_requested.emit(self.project_id))
        menu.addAction("Rename…", lambda: self.rename_requested.emit(self.project_id))
        menu.addAction("Show in folder", lambda: self.reveal_requested.emit(self.project_id))
        menu.addSeparator()
        menu.addAction("Delete…", lambda: self.delete_requested.emit(self.project_id))
        menu.exec(self.menu_button.mapToGlobal(self.menu_button.rect().bottomLeft()))


class ProjectsHome(QWidget):
    file_chosen = Signal(str)
    browse_requested = Signal()
    live_requested = Signal()
    import_folder_requested = Signal()
    open_project_requested = Signal(str)
    rename_project_requested = Signal(str)
    reveal_project_requested = Signal(str)
    delete_project_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._show_all = False
        self._summaries: list[ProjectSummary] = []
        self._cards: dict[str, ProjectCard] = {}

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        content = QWidget()
        scroll.setWidget(content)
        page = QVBoxLayout(self)
        page.setContentsMargins(0, 0, 0, 0)
        page.addWidget(scroll)

        body = QVBoxLayout(content)
        body.setContentsMargins(28, 24, 28, 16)
        body.setSpacing(16)

        self.greeting_label = QLabel(greeting())
        self.greeting_label.setProperty("type", "title")
        subtitle = QLabel("Your recordings stay on this device.")
        subtitle.setProperty("type", "caption")
        body.addWidget(self.greeting_label)
        body.addWidget(subtitle)

        actions = QHBoxLayout()
        actions.setSpacing(14)
        self.drop_card = DropCard()
        self.drop_card.file_dropped.connect(self.file_chosen)
        self.drop_card.browse_requested.connect(self.browse_requested)
        actions.addWidget(self.drop_card, 3)
        self.live_card = self._secondary_card("◉", "Live capture", "Record and transcribe\nin real time", self.live_requested)
        self.import_card = self._secondary_card("▣", "Import folder", "Transcribe multiple\nfiles at once", self.import_folder_requested)
        actions.addWidget(self.live_card, 1)
        actions.addWidget(self.import_card, 1)
        body.addLayout(actions)

        header = QHBoxLayout()
        self.recent_title = QLabel("Recent projects")
        self.recent_title.setProperty("type", "heading")
        header.addWidget(self.recent_title)
        header.addStretch(1)
        self.view_all_button = AppButton("View all →", "ghost")
        self.view_all_button.clicked.connect(self._toggle_view_all)
        header.addWidget(self.view_all_button)
        body.addLayout(header)

        self.grid_host = QWidget()
        self.grid = QGridLayout(self.grid_host)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(14)
        body.addWidget(self.grid_host)
        self.empty_state = EmptyState(
            "No projects yet",
            "Drop an audio or video file above, or start a live capture. "
            "Finished transcripts are saved here so you can reopen them.",
            "New transcription",
        )
        self.empty_state.action_clicked.connect(self.browse_requested)
        body.addWidget(self.empty_state)
        body.addStretch(1)

        self.footer = QLabel("🛡  All processing is local by default.")
        self.footer.setProperty("type", "caption")
        self.footer.setAccessibleName("All processing is local by default")
        body.addWidget(self.footer)
        self.set_projects([])

    @staticmethod
    def _secondary_card(glyph: str, title: str, hint: str, signal: Signal) -> Card:
        card = Card(interactive=True)
        card.setMinimumHeight(170)
        card.setAccessibleName(title)
        layout = QVBoxLayout(card)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(6)
        icon = QLabel(glyph)
        icon.setProperty("type", "title")
        icon.setAlignment(Qt.AlignCenter)
        name = QLabel(title)
        name.setProperty("type", "label")
        name.setAlignment(Qt.AlignCenter)
        caption = QLabel(hint)
        caption.setProperty("type", "caption")
        caption.setAlignment(Qt.AlignCenter)
        for widget in (icon, name, caption):
            layout.addWidget(widget)
        card.clicked.connect(signal)
        return card

    def set_projects(self, summaries: list[ProjectSummary]) -> None:
        self._summaries = list(summaries)
        self._rebuild()

    def set_progress(self, project_id: str, percent: int) -> None:
        card = self._cards.get(project_id)
        if card is not None:
            card.set_progress(percent)

    def _toggle_view_all(self) -> None:
        self._show_all = not self._show_all
        self._rebuild()

    def _rebuild(self) -> None:
        while self.grid.count():
            item = self.grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._cards.clear()
        shown = self._summaries if self._show_all else self._summaries[:RECENT_LIMIT]
        for index, summary in enumerate(shown):
            card = ProjectCard(summary)
            card.open_requested.connect(self.open_project_requested)
            card.rename_requested.connect(self.rename_project_requested)
            card.reveal_requested.connect(self.reveal_project_requested)
            card.delete_requested.connect(self.delete_project_requested)
            self._cards[summary.id] = card
            self.grid.addWidget(card, index // COLUMNS, index % COLUMNS)
        for column in range(COLUMNS):
            self.grid.setColumnStretch(column, 1)
        has_projects = bool(self._summaries)
        self.grid_host.setVisible(has_projects)
        self.empty_state.setVisible(not has_projects)
        self.view_all_button.setVisible(len(self._summaries) > RECENT_LIMIT)
        self.view_all_button.setText("Show recent →" if self._show_all else "View all →")
        self.recent_title.setText("All projects" if self._show_all else "Recent projects")
