"""Job timeline widget: one row per pipeline stage plus a collapsible event log."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ui_qt.job_stages import STAGE_LABELS, STAGE_ORDER, Stage


def set_state(widget: QWidget, state: str) -> None:
    """Set the QSS ``state`` property on ``widget`` and re-polish it."""
    if widget.property("state") == state:
        return
    widget.setProperty("state", state)
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


class JobTimeline(QFrame):
    """Stage rows (name, progress bar, detail text) with a collapsible Details log."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        title = QLabel("Progress")
        title.setObjectName("PageSubtitle")
        layout.addWidget(title)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)

        self._names: dict[Stage, QLabel] = {}
        self._bars: dict[Stage, QProgressBar] = {}
        self._details: dict[Stage, QLabel] = {}
        for row, stage in enumerate(STAGE_ORDER):
            name = QLabel(STAGE_LABELS[stage])
            name.setObjectName("StageName")
            name.setMinimumWidth(84)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setMinimumHeight(24)
            detail = QLabel("--")
            detail.setObjectName("StageDetail")
            detail.setMinimumWidth(190)
            detail.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            grid.addWidget(name, row, 0)
            grid.addWidget(bar, row, 1)
            grid.addWidget(detail, row, 2)
            self._names[stage] = name
            self._bars[stage] = bar
            self._details[stage] = detail
            self.set_stage_state(stage, "pending")

        self.details_toggle = QToolButton()
        self.details_toggle.setObjectName("detailsToggle")
        self.details_toggle.setCheckable(True)
        self.details_toggle.setChecked(False)
        self.details_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.details_toggle.setArrowType(Qt.RightArrow)
        self.details_toggle.setText("Details")
        self.details_toggle.toggled.connect(self._on_details_toggled)
        layout.addWidget(self.details_toggle, 0, Qt.AlignLeft)

        self.log = QPlainTextEdit()
        self.log.setObjectName("TerminalLog")
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Pipeline events appear here...")
        self.log.setMinimumHeight(96)
        self.log.setVisible(False)
        layout.addWidget(self.log)

    def bar(self, stage: Stage) -> QProgressBar:
        return self._bars[stage]

    def detail_label(self, stage: Stage) -> QLabel:
        return self._details[stage]

    def set_stage_state(self, stage: Stage, state: str) -> None:
        set_state(self._names[stage], state)
        set_state(self._bars[stage], state)

    def set_stage_visible(self, stage: Stage, visible: bool) -> None:
        """Show or hide a stage's whole row (name, bar and detail)."""
        self._names[stage].setVisible(visible)
        self._bars[stage].setVisible(visible)
        self._details[stage].setVisible(visible)

    def _on_details_toggled(self, checked: bool) -> None:
        self.log.setVisible(checked)
        self.details_toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
