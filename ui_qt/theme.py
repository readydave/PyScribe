"""Shared theme for the PyScribe Qt UI: palette tokens, bundled font, and the QSS template."""

from __future__ import annotations

import logging

from pathlib import Path

from PySide6.QtCore import QPointF, QStandardPaths, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QImage, QPainter, QPalette, QPen, QPolygonF
from PySide6.QtWidgets import QApplication

from services.ui_themes import DEFAULT_THEME_ID, Palette, ResolvedTheme, resolve
from services.ui_tokens import FONT_DIR, FONT_FAMILY, PALETTES

LOGGER = logging.getLogger(__name__)

FONT_FALLBACKS = '"Segoe UI", "Roboto", "Helvetica", sans-serif'
THEME_MODES = ("system", "light", "dark")

_active_mode = "light"
_resolved: ResolvedTheme | None = None

# Progress/stage states understood by the `state` dynamic property in the QSS.
STAGE_STATES = ("pending", "active", "done", "failed", "disabled")


def active_palette() -> Palette:
    """Palette of the theme most recently applied (used by custom-painted widgets)."""
    if _resolved is not None:
        return _resolved.palette
    return PALETTES[_active_mode]


def active_mode() -> str:
    """The effective mode (``light`` or ``dark``) most recently applied."""
    return _active_mode


def active_theme() -> ResolvedTheme | None:
    """The resolved theme most recently applied, or None before the first ``apply_theme``."""
    return _resolved


def stage_color(stage: str | None) -> str:
    """Trace colour for a job stage; the neutral bar colour when no stage is running."""
    palette = active_palette()
    if _resolved is None:
        return palette.bar_active
    return _resolved.stage_colors.get(stage or "", palette.bar_active)


def speaker_color(label: str) -> str:
    """Colour for a speaker label such as ``S1`` in the active theme; unknown speakers use muted text."""
    palette = active_palette()
    digits = "".join(ch for ch in str(label) if ch.isdigit())
    if not digits or _resolved is None or not _resolved.speaker_colors:
        return palette.muted
    return _resolved.speaker_colors[(int(digits) - 1) % len(_resolved.speaker_colors)]


def sanitize_mode(value: object) -> str:
    """Return a valid theme mode, defaulting to ``system``."""
    mode = str(value or "system").strip().lower()
    return mode if mode in THEME_MODES else "system"


def resolve_mode(mode: str) -> str:
    """Resolve ``system`` to ``light`` or ``dark`` using the OS colour scheme."""
    if mode in ("light", "dark"):
        return mode
    app = QApplication.instance()
    if app is None:
        return "light"
    scheme = app.styleHints().colorScheme()
    if scheme == Qt.ColorScheme.Dark:
        return "dark"
    if scheme == Qt.ColorScheme.Light:
        return "light"
    return "dark" if app.palette().color(QPalette.Window).lightness() < 128 else "light"


def load_fonts() -> str:
    """Register the bundled font files and return the family to use.

    Falls back to a system family name when the files are missing.
    """
    loaded = False
    if FONT_DIR.is_dir():
        for path in sorted(FONT_DIR.glob("*.ttf")):
            if QFontDatabase.addApplicationFont(str(path)) >= 0:
                loaded = True
            else:
                LOGGER.warning("Could not load bundled font: %s", path)
    if not loaded:
        LOGGER.warning("Bundled font not found in %s; using system fonts.", FONT_DIR)
        return "Segoe UI"
    return FONT_FAMILY


def _sync_color_scheme(app: QApplication, mode: str) -> None:
    """Make native pieces (title bars, message boxes, file dialogs) follow a forced light/dark choice.

    ``system`` hands control back to the OS. Needs Qt 6.8+; older Qt keeps the OS scheme.
    """
    hints = app.styleHints()
    if not hasattr(hints, "setColorScheme"):
        return
    schemes = {"light": Qt.ColorScheme.Light, "dark": Qt.ColorScheme.Dark}
    hints.setColorScheme(schemes.get(mode, Qt.ColorScheme.Unknown))


def apply_theme(
    app: QApplication,
    mode: str,
    theme_id: str = DEFAULT_THEME_ID,
    custom_themes: list[dict] | None = None,
) -> str:
    """Apply the Fusion style, bundled font, and the QSS for a theme. Returns the effective mode."""
    global _active_mode, _resolved
    _sync_color_scheme(app, mode)
    effective = resolve_mode(mode)
    _active_mode = effective
    _resolved = resolve(theme_id, custom_themes, effective)
    family = load_fonts()
    if app.style().objectName().lower() != "fusion":
        app.setStyle("Fusion")
    font = QFont(family, 10)
    if app.font() != font:
        app.setFont(font)
    qss = build_qss(effective, family, _resolved.palette)
    # Re-applying an identical stylesheet still re-polishes every widget, so skip no-ops.
    if app.styleSheet() != qss:
        app.setStyleSheet(qss)
    return effective


def _check_icon(color: str) -> str:
    """Render a tick mark in ``color`` to a cached PNG and return its path ("" if it can't be written).

    QSS cannot draw a check mark, and styling the indicator hides the native one.
    """
    try:
        directory = Path(QStandardPaths.writableLocation(QStandardPaths.CacheLocation)) / "pyscribe-theme"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"check-{color.lstrip('#').lower()}.png"
        if not path.exists():
            image = QImage(32, 32, QImage.Format_ARGB32)
            image.fill(Qt.transparent)
            painter = QPainter(image)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(QPen(QColor(color), 4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawPolyline(QPolygonF([QPointF(8, 17), QPointF(14, 23), QPointF(24, 9)]))
            painter.end()
            image.save(str(path), "PNG")
        return path.as_posix() if path.exists() else ""
    except OSError:
        LOGGER.warning("Could not write the check-mark icon.", exc_info=True)
        return ""


def build_qss(mode: str, family: str = FONT_FAMILY, palette: Palette | None = None) -> str:
    """Build the application stylesheet for ``mode`` (``light`` or ``dark``)."""
    p = palette or PALETTES["dark" if mode == "dark" else "light"]
    check = _check_icon(p.accent_text)
    check_image = f'image: url("{check}");' if check else ""
    return f"""
        QWidget {{
            background: {p.page};
            color: {p.ink};
            font-family: "{family}", {FONT_FALLBACKS};
        }}
        QPushButton, QFrame, QLineEdit {{
            border-radius: 8px;
        }}
        QLabel, QCheckBox, QRadioButton {{
            background: transparent;
        }}
        QCheckBox {{
            spacing: 8px;
        }}
        QCheckBox::indicator {{
            width: 16px;
            height: 16px;
            border: 2px solid {p.muted};
            border-radius: 4px;
            background: {p.input_bg};
        }}
        QCheckBox::indicator:hover {{
            border-color: {p.ink};
        }}
        QCheckBox::indicator:checked {{
            background: {p.accent};
            border-color: {p.accent};
            {check_image}
        }}
        QCheckBox::indicator:disabled {{
            border-color: {p.disabled_bg};
            background: {p.surface};
        }}
        #Sidebar {{
            background: {p.sidebar};
            border-right: 1px solid {p.rule};
            padding: 12px;
        }}
        #SidebarBrand {{
            font-weight: 700;
            padding: 6px 2px;
            color: {p.ink};
        }}
        #SidebarNav {{
            border: 1px solid {p.rule};
            background: {p.sidebar};
            outline: none;
            padding: 10px;
        }}
        #SidebarNav::item {{
            padding: 10px 12px;
            margin: 3px 0;
            border-radius: 8px;
        }}
        #SidebarNav::item:selected {{
            background: {p.accent};
            color: {p.accent_text};
            font-weight: 600;
        }}
        #MainStack {{
            background: {p.surface};
            padding: 12px;
        }}
        #MainSurface, #StatusPanel {{
            background: {p.surface};
            border: 1px solid {p.rule};
            padding: 12px;
        }}
        #Card {{
            background: {p.card};
            border: 1px solid {p.rule};
            padding: 12px;
        }}
        #PageTitle {{
            font-weight: 700;
            padding-bottom: 2px;
        }}
        #PageSubtitle {{
            color: {p.muted};
            padding-bottom: 6px;
        }}
        #pathLabel {{
            background: {p.input_bg};
            border: 1px solid {p.rule};
            padding: 10px 12px;
        }}
        #dropZone {{
            border: 2px dashed {p.muted};
            background: {p.surface};
            padding: 15px;
        }}
        #dropZone[activeDrop="true"] {{
            border-color: {p.ink};
            background: {p.card};
        }}
        #dropBrowseButton {{
            background: {p.accent};
            color: {p.accent_text};
            padding: 2px 18px;
            font-weight: 700;
            font-size: 11pt;
            min-width: 140px;
            min-height: 44px;
            border-radius: 22px;
            outline: none;
            border: none;
        }}
        #dropBrowseButton:hover {{
            background: {p.accent_hover};
        }}
        #dropTitle {{
            color: {p.ink};
            font-weight: 700;
            background: transparent;
        }}
        #dropSubtitle {{
            color: {p.muted};
            background: transparent;
        }}
        QPushButton, QToolButton {{
            background: {p.accent};
            color: {p.accent_text};
            border: 1px solid {p.accent};
            padding: 10px 14px;
            font-weight: 600;
        }}
        QToolButton#sidebarToggleButton, QToolButton#statusToggleButton {{
            min-width: 24px;
            max-width: 24px;
            min-height: 24px;
            max-height: 24px;
            padding: 2px;
            font-weight: 700;
        }}
        QToolButton#detailsToggle {{
            background: transparent;
            color: {p.muted};
            border: none;
            padding: 2px 4px;
            font-weight: 600;
        }}
        QToolButton#detailsToggle:hover {{
            color: {p.ink};
        }}
        QPushButton:hover, QToolButton:hover {{
            background: {p.accent_hover};
            border-color: {p.accent_hover};
        }}
        QToolButton#detailsToggle:hover {{
            background: transparent;
            border: none;
        }}
        QPushButton[role="primary"] {{
            background: {p.rubric};
            border-color: {p.rubric};
            color: {p.rubric_text};
        }}
        QPushButton[role="primary"]:hover {{
            background: {p.rubric_hover};
            border-color: {p.rubric_hover};
        }}
        QPushButton:disabled, QToolButton:disabled {{
            background: {p.disabled_bg};
            border-color: {p.disabled_bg};
            color: {p.disabled_text};
        }}
        QPushButton:focus, QToolButton:focus {{
            outline: none;
            border: 2px solid {p.ink};
            padding: 9px 13px;
        }}
        QToolButton#detailsToggle:focus {{
            border: 1px solid {p.ink};
            border-radius: 4px;
            padding: 1px 3px;
        }}
        QToolButton#sidebarToggleButton:focus, QToolButton#statusToggleButton:focus {{
            padding: 1px;
        }}
        #dropBrowseButton:focus {{
            border: 2px solid {p.ink};
            padding: 1px 17px;
        }}
        QComboBox:focus {{
            border: 1px solid {p.ink};
        }}
        QCheckBox:focus {{
            border: 1px solid {p.ink};
            border-radius: 4px;
        }}
        QPushButton#exitButton {{
            background: transparent;
            border-color: {p.rule};
            color: {p.muted};
        }}
        QPushButton#exitButton:hover {{
            background: {p.rubric};
            border-color: {p.rubric};
            color: {p.rubric_text};
        }}
        QLineEdit, QComboBox, QPlainTextEdit, QTextEdit {{
            background: {p.input_bg};
            border: 1px solid {p.rule};
            padding: 10px 12px;
            selection-background-color: {p.accent};
            selection-color: {p.accent_text};
        }}
        QComboBox QAbstractItemView {{
            background: {p.card};
            color: {p.ink};
            selection-background-color: {p.accent};
            selection-color: {p.accent_text};
        }}
        QComboBox::item:disabled {{
            color: {p.disabled_text};
            background: transparent;
        }}
        QGroupBox {{
            background: {p.card};
            border: 1px solid {p.rule};
            border-radius: 8px;
            margin-top: 8px;
            padding: 12px;
            font-weight: 600;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
            color: {p.muted};
        }}
        QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus {{
            border: 1px solid {p.ink};
        }}
        #metricsLabel[state="recording"] {{
            color: {p.failed_text};
            font-weight: 700;
        }}
        #hint, #metricsLabel {{
            color: {p.muted};
        }}
        #tokenLabel {{
            color: {p.done};
            font-weight: 600;
        }}
        QProgressBar {{
            border: 1px solid {p.rule};
            border-radius: 10px;
            background: {p.input_bg};
            color: {p.ink};
            text-align: center;
            min-height: 20px;
        }}
        QProgressBar::chunk {{
            background: {p.bar_active};
            border-radius: 8px;
            margin: 1px;
        }}
        QProgressBar[state="done"] {{
            color: {p.done_text};
        }}
        QProgressBar[state="done"]::chunk {{
            background: {p.done};
        }}
        QProgressBar[state="failed"]::chunk {{
            background: {p.rubric};
        }}
        QProgressBar[state="disabled"]::chunk {{
            background: {p.disabled_bg};
        }}
        #Transparent {{
            background: transparent;
        }}
        #StatusLine {{
            font-weight: 600;
            font-size: 11pt;
        }}
        QMainWindow::separator {{
            background: {p.page};
            width: 6px;
            height: 6px;
        }}
        QMainWindow::separator:hover {{
            background: {p.rule};
        }}
        QDockWidget {{
            font-weight: 600;
        }}
        QDockWidget::title {{
            background: {p.surface};
            border: 1px solid {p.rule};
            padding: 6px 10px;
            text-align: left;
        }}
        QTabBar::tab {{
            background: {p.surface};
            color: {p.muted};
            border: 1px solid {p.rule};
            padding: 6px 14px;
        }}
        QTabBar::tab:selected {{
            background: {p.card};
            color: {p.ink};
        }}
        QPushButton[segment] {{
            background: transparent;
            color: {p.ink};
            border: 1px solid {p.rule};
            border-radius: 0px;
            padding: 8px 22px;
        }}
        QPushButton[segment="left"] {{
            border-top-left-radius: 8px;
            border-bottom-left-radius: 8px;
        }}
        QPushButton[segment="right"] {{
            border-top-right-radius: 8px;
            border-bottom-right-radius: 8px;
        }}
        QPushButton[segment]:hover {{
            background: {p.surface};
            border-color: {p.rule};
        }}
        QPushButton[segment]:checked {{
            background: {p.accent};
            color: {p.accent_text};
            border-color: {p.accent};
        }}
        QListWidget::item:selected, QListView::item:selected {{
            background: {p.accent};
            color: {p.accent_text};
        }}
        QScrollBar:vertical {{
            background: transparent;
            width: 12px;
            margin: 0;
        }}
        QScrollBar:horizontal {{
            background: transparent;
            height: 12px;
            margin: 0;
        }}
        QScrollBar::handle:vertical {{
            background: {p.rule};
            border-radius: 5px;
            min-height: 28px;
            margin: 2px;
        }}
        QScrollBar::handle:horizontal {{
            background: {p.rule};
            border-radius: 5px;
            min-width: 28px;
            margin: 2px;
        }}
        QScrollBar::handle:hover {{
            background: {p.muted};
        }}
        QScrollBar::add-line, QScrollBar::sub-line {{
            width: 0;
            height: 0;
        }}
        QScrollBar::add-page, QScrollBar::sub-page {{
            background: transparent;
        }}
        #StageName {{
            font-weight: 600;
        }}
        #StageName[state="pending"], #StageName[state="disabled"] {{
            color: {p.muted};
            font-weight: 500;
        }}
        #StageName[state="failed"] {{
            color: {p.failed_text};
        }}
        #StageDetail {{
            color: {p.muted};
        }}
        QPlainTextEdit#TerminalLog {{
            background: {p.log_bg};
            color: {p.log_text};
            border: 1px solid {p.rule};
            font-family: "Consolas", "DejaVu Sans Mono", "Courier New", monospace;
            padding: 10px;
        }}
    """
