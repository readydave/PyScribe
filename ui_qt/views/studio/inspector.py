"""Studio inspector: Summary, Action items, Speakers and Visuals (one view open at a time, collapsible)."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QScrollArea,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from services.project_store import Project, speaker_labels
from services.prompt_template_service import get_prompt_template
from ui_qt import theme
from ui_qt.components import AppButton, InspectorTabs
from ui_qt.views.studio.transcript_view import format_clock

SUMMARY_TEMPLATE = "meeting-summary"
ACTIONS_TEMPLATE = "action-items"


def speaker_stats(segments: list[dict]) -> dict[str, tuple[int, float]]:
    """``{label: (turns, seconds spoken)}``; consecutive segments by one speaker count as a single turn."""
    stats: dict[str, tuple[int, float]] = {}
    previous = None
    for segment in segments:
        label = segment.get("speaker")
        if not label or label in ("S?", "?"):
            previous = None
            continue
        turns, seconds = stats.get(label, (0, 0.0))
        stats[label] = (turns + (0 if label == previous else 1), seconds + max(0.0, segment["end"] - segment["start"]))
        previous = label
    return stats


class OutputPage(QWidget):
    """A stored AI output with a button to generate (or regenerate) it."""

    generate_requested = Signal(str)

    def __init__(self, template_id: str, generate_label: str, empty_text: str) -> None:
        super().__init__()
        self._template_id = template_id
        self._empty_text = empty_text
        self._generate_label = generate_label
        self._regenerate_label = generate_label.replace("Generate", "Regenerate", 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(8)
        self.chooser = QComboBox()
        self.chooser.setAccessibleName("Stored output")
        self.chooser.setVisible(False)
        self.chooser.currentIndexChanged.connect(self._on_choice)
        layout.addWidget(self.chooser)
        self.browser = QTextBrowser()
        self.browser.setOpenLinks(False)
        self.browser.setFrameShape(QFrame.NoFrame)
        self.browser.setAccessibleName(generate_label)
        layout.addWidget(self.browser, 1)
        self.generate_button = AppButton(generate_label, "secondary")
        self.generate_button.clicked.connect(lambda: self.generate_requested.emit(self._template_id))
        layout.addWidget(self.generate_button)
        self._outputs: dict[str, dict[str, str]] = {}
        self.set_outputs({})

    def set_outputs(self, outputs: dict[str, dict[str, str]]) -> None:
        self._outputs = outputs
        keys = self._keys()
        self.chooser.blockSignals(True)
        self.chooser.clear()
        for key in keys:
            self.chooser.addItem(self._name_for(key), key)
        self.chooser.setVisible(len(keys) > 1)
        self.chooser.blockSignals(False)
        self._show(keys[0] if keys else "")

    def _keys(self) -> list[str]:
        return [self._template_id] if self._template_id in self._outputs else []

    @staticmethod
    def _name_for(key: str) -> str:
        template = get_prompt_template(key)
        return template.name if template is not None else key

    def _show(self, key: str) -> None:
        text = self._outputs.get(key, {}).get("text", "").strip()
        if text:
            self.browser.setMarkdown(text)
        else:
            self.browser.setPlainText(self._empty_text)
        self.generate_button.setText(self._regenerate_label if text else self._generate_label)

    def _on_choice(self) -> None:
        self._show(str(self.chooser.currentData() or ""))


class SummaryPage(OutputPage):
    """The summary, plus any other stored AI output (for example from a custom template)."""

    def _keys(self) -> list[str]:
        others = [k for k in self._outputs if k not in (SUMMARY_TEMPLATE, ACTIONS_TEMPLATE)]
        first = [SUMMARY_TEMPLATE] if SUMMARY_TEMPLATE in self._outputs else []
        return first + sorted(others)


class SpeakerBadge(QWidget):
    def __init__(self, label: str) -> None:
        super().__init__()
        self._label = label
        self.setFixedSize(24, 24)

    def paintEvent(self, _event: object) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(theme.speaker_color(self._label)))
        painter.drawEllipse(self.rect())
        painter.end()


class SpeakersPage(QWidget):
    renamed = Signal(str, str)

    def __init__(self) -> None:
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 8, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setFocusPolicy(Qt.NoFocus)
        host = QWidget()
        self._rows = QVBoxLayout(host)
        self._rows.setContentsMargins(0, 0, 0, 0)
        self._rows.setSpacing(10)
        self._rows.addStretch(1)
        scroll.setWidget(host)
        outer.addWidget(scroll, 1)
        self.empty = QLabel("No speaker labels in this transcript. Turn on speaker identification before transcribing to get them.")
        self.empty.setWordWrap(True)
        self.empty.setProperty("type", "caption")
        outer.addWidget(self.empty)
        self.edits: dict[str, QLineEdit] = {}

    def set_project(self, project: Project) -> None:
        while self._rows.count() > 1:
            item = self._rows.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.edits.clear()
        stats = speaker_stats(project.segments)
        labels = speaker_labels(project.segments)
        self.empty.setVisible(not labels)
        for label in labels:
            turns, seconds = stats.get(label, (0, 0.0))
            row = QHBoxLayout()
            row.setSpacing(10)
            row.addWidget(SpeakerBadge(label))
            column = QVBoxLayout()
            column.setSpacing(2)
            edit = QLineEdit(project.speaker_names.get(label, ""))
            edit.setPlaceholderText(f"{label} — add a name")
            edit.setAccessibleName(f"Name for {label}")
            edit.setMaxLength(40)
            edit.editingFinished.connect(lambda lbl=label, widget=edit: self._emit_if_changed(lbl, widget))
            edit.setProperty("saved", edit.text())
            facts = QLabel(f"{turns} turn{'s' if turns != 1 else ''} · {format_clock(seconds)} speaking")
            facts.setProperty("type", "caption")
            column.addWidget(edit)
            column.addWidget(facts)
            row.addLayout(column, 1)
            wrapper = QWidget()
            wrapper.setLayout(row)
            self._rows.insertWidget(self._rows.count() - 1, wrapper)
            self.edits[label] = edit

    def _emit_if_changed(self, label: str, edit: QLineEdit) -> None:
        text = edit.text().strip()
        if text != edit.property("saved"):
            edit.setProperty("saved", text)
            self.renamed.emit(label, text)


class VisualsPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 0)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setFrameShape(QFrame.NoFrame)
        self.text.setAccessibleName("Visual analysis")
        layout.addWidget(self.text)

    def set_report(self, report: str) -> None:
        self.text.setPlainText(report.strip() or "No visual analysis for this project. Turn on visual analysis (OCR) before transcribing a video.")


class StudioInspector(QWidget):
    generate_requested = Signal(str)
    speaker_renamed = Signal(str, str)
    collapsed_changed = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        self.tabs = InspectorTabs()
        self.summary = SummaryPage(
            SUMMARY_TEMPLATE, "Generate summary…",
            "No summary yet. Nothing is sent anywhere until you generate one with your AI connection.",
        )
        self.actions = OutputPage(
            ACTIONS_TEMPLATE, "Generate action items…",
            "No action items yet. Nothing is sent anywhere until you generate them.",
        )
        self.speakers = SpeakersPage()
        self.visuals = VisualsPage()
        self.tabs.add_tab("summary", "Summary", self.summary)
        self.tabs.add_tab("actions", "Action items", self.actions)
        self.tabs.add_tab("speakers", "Speakers", self.speakers)
        self.tabs.add_tab("visuals", "Visuals", self.visuals)
        layout.addWidget(self.tabs)
        for page in (self.summary, self.actions):
            page.generate_requested.connect(self.generate_requested)
        self.speakers.renamed.connect(self.speaker_renamed)
        self.tabs.collapsed_changed.connect(self.collapsed_changed)

    def set_project(self, project: Project) -> None:
        self.update_outputs(project)
        self.speakers.set_project(project)
        self.visuals.set_report(project.visual_report)

    def update_outputs(self, project: Project) -> None:
        self.summary.set_outputs(project.ai_outputs)
        self.actions.set_outputs(project.ai_outputs)
