"""Hardware dock: rolling 60-second traces for CPU, RAM, GPU and VRAM."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ui_qt import theme

HISTORY_SECONDS = 60
COMPACT_WIDTH = 230
STAGE_NAMES = {"load": "Loading model", "save": "Saving", "transcribe": "Transcribing", "speakers": "Identifying speakers", "visuals": "Analyzing visuals"}


@dataclass
class _Sample:
    fraction: float
    stage: str | None


class _TraceRow(QWidget):
    """One metric: name, current value, and a sparkline (or a mini bar when narrow)."""

    def __init__(self, name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._name = name
        self._samples: deque[_Sample] = deque(maxlen=HISTORY_SECONDS)
        self._value_text = "--"
        self._ceiling_text: str | None = None
        self._idle = True
        self.setMinimumHeight(24)
        self.setMaximumHeight(110)

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    def push(self, fraction: float, value_text: str, stage: str | None, ceiling_text: str | None = None) -> None:
        self._samples.append(_Sample(max(0.0, min(1.0, fraction)), stage))
        self._value_text = value_text
        self._ceiling_text = ceiling_text
        self._idle = False
        self.update()

    def set_idle(self) -> None:
        self._idle = True
        self._value_text = "--"
        self.update()

    def sizeHint(self):  # noqa: N802
        hint = super().sizeHint()
        hint.setHeight(52)
        return hint

    def minimumSizeHint(self):  # noqa: N802
        hint = super().minimumSizeHint()
        hint.setHeight(24 if self.width() < COMPACT_WIDTH else 46)
        return hint

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802
        if event.type() in (QEvent.StyleChange, QEvent.PaletteChange):
            self.update()
        super().changeEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        palette = theme.active_palette()
        if self.width() < COMPACT_WIDTH:
            self._paint_compact(painter, palette)
        else:
            self._paint_full(painter, palette)
        painter.end()

    def _stage_color(self, stage: str | None) -> QColor:
        return QColor(theme.stage_color(stage))

    def _paint_compact(self, painter: QPainter, palette: theme.Palette) -> None:
        w, h = self.width(), self.height()
        small = QFont(self.font())
        small.setPointSize(9)
        painter.setFont(small)
        painter.setPen(QColor(palette.muted))
        painter.drawText(QRectF(0, 0, 44, h), Qt.AlignVCenter | Qt.AlignLeft, self._name)
        bar = QRectF(48, h / 2 - 5, max(10.0, w - 48 - 70), 10)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(palette.rule))
        painter.drawRoundedRect(bar, 5, 5)
        if self._samples and not self._idle:
            last = self._samples[-1]
            fill = QRectF(bar.x(), bar.y(), bar.width() * last.fraction, bar.height())
            painter.setBrush(self._stage_color(last.stage))
            painter.drawRoundedRect(fill, 5, 5)
        painter.setPen(QColor(palette.ink))
        painter.drawText(QRectF(w - 66, 0, 66, h), Qt.AlignVCenter | Qt.AlignRight, self._value_text)

    def _paint_full(self, painter: QPainter, palette: theme.Palette) -> None:
        w, h = self.width(), self.height()
        name_font = QFont(self.font())
        name_font.setPointSize(9)
        painter.setFont(name_font)
        painter.setPen(QColor(palette.muted))
        painter.drawText(QRectF(4, 4, 90, 16), Qt.AlignLeft | Qt.AlignVCenter, self._name)
        value_font = QFont(self.font())
        value_font.setPointSize(15)
        value_font.setWeight(QFont.DemiBold)
        painter.setFont(value_font)
        painter.setPen(QColor(palette.ink if not self._idle else palette.muted))
        painter.drawText(QRectF(4, 20, 96, h - 24), Qt.AlignLeft | Qt.AlignTop, self._value_text)

        area = QRectF(102, 6, max(20.0, w - 102 - 6), h - 12)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(palette.input_bg))
        painter.drawRoundedRect(area, 6, 6)
        if self._ceiling_text:
            pen = QPen(QColor(palette.muted), 1, Qt.DashLine)
            painter.setPen(pen)
            painter.drawLine(QPointF(area.left() + 2, area.top() + 2), QPointF(area.right() - 2, area.top() + 2))
            painter.setFont(name_font)
            painter.drawText(
                QRectF(area.left() + 6, area.top() + 3, area.width() - 12, 14),
                Qt.AlignRight | Qt.AlignTop,
                self._ceiling_text,
            )
        if len(self._samples) < 2:
            return
        self._paint_trace(painter, area.adjusted(2, 4, -2, -2), palette)

    def _paint_trace(self, painter: QPainter, area: QRectF, palette: theme.Palette) -> None:
        samples = list(self._samples)
        step = area.width() / (HISTORY_SECONDS - 1)
        base_y = area.bottom()

        def point(index: int, sample: _Sample) -> QPointF:
            x = area.right() - (len(samples) - 1 - index) * step
            return QPointF(x, base_y - sample.fraction * area.height())

        for i in range(1, len(samples)):
            a, b = point(i - 1, samples[i - 1]), point(i, samples[i])
            color = QColor(palette.muted) if self._idle else self._stage_color(samples[i].stage)
            fill = QColor(color)
            fill.setAlpha(55)
            painter.setPen(Qt.NoPen)
            painter.setBrush(fill)
            painter.drawPolygon(QPolygonF([a, b, QPointF(b.x(), base_y), QPointF(a.x(), base_y)]))
            painter.setPen(QPen(color, 1.6))
            painter.drawLine(a, b)
        peak_index = max(range(len(samples)), key=lambda idx: samples[idx].fraction)
        peak = point(peak_index, samples[peak_index])
        painter.setPen(QPen(QColor(palette.ink), 1))
        painter.setBrush(QColor(palette.card))
        painter.drawEllipse(peak, 2.5, 2.5)


class HardwarePanel(QWidget):
    """Instrument panel fed by one sample per second while a job runs."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        legend = QHBoxLayout()
        legend.setContentsMargins(4, 0, 4, 0)
        self.stage_label = QLabel("Idle")
        self.stage_label.setObjectName("hint")
        legend.addWidget(self.stage_label)
        legend.addStretch(1)
        layout.addLayout(legend)
        self._rows = {
            "cpu": _TraceRow("CPU"),
            "ram": _TraceRow("Memory"),
            "gpu": _TraceRow("GPU"),
            "vram": _TraceRow("VRAM"),
        }
        for row in self._rows.values():
            layout.addWidget(row)
        self._rows["gpu"].setVisible(False)
        self._rows["vram"].setVisible(False)
        layout.addStretch(1)

        self._stage: str | None = None

    def row(self, key: str) -> _TraceRow:
        return self._rows[key]

    def set_stage(self, stage: str | None) -> None:
        self._stage = stage
        self.stage_label.setText(STAGE_NAMES.get(stage or "", "Idle"))

    def add_sample(self, sample: dict[str, float]) -> None:
        """Add one reading: cpu/ram/gpu are percents; vram_used/vram_total are GB."""
        if not sample:
            self.set_idle()
            return
        stage = self._stage
        cpu, ram = sample.get("cpu"), sample.get("ram")
        if cpu is not None:
            self._rows["cpu"].push(cpu / 100.0, f"{cpu:.0f}%", stage)
        if ram is not None:
            self._rows["ram"].push(ram / 100.0, f"{ram:.0f}%", stage)
        gpu = sample.get("gpu")
        used, total = sample.get("vram_used"), sample.get("vram_total")
        has_gpu = gpu is not None and used is not None and total
        self._rows["gpu"].setVisible(bool(has_gpu))
        self._rows["vram"].setVisible(bool(has_gpu))
        if has_gpu:
            self._rows["gpu"].push(gpu / 100.0, f"{gpu:.0f}%", stage)
            self._rows["vram"].push(used / total, f"{used:.1f} GB", stage, ceiling_text=f"{total:.1f} GB")

    def set_idle(self) -> None:
        for row in self._rows.values():
            row.set_idle()
        self.set_stage(None)
