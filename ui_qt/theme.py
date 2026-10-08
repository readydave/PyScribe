"""Shared theme for the PyScribe Qt UI: palette tokens, bundled font, and the QSS template."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication

LOGGER = logging.getLogger(__name__)

FONT_FAMILY = "Atkinson Hyperlegible Next"
FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
FONT_FALLBACKS = '"Segoe UI", "Roboto", "Helvetica", sans-serif'
THEME_MODES = ("system", "light", "dark")

# Progress/stage states understood by the `state` dynamic property in the QSS.
STAGE_STATES = ("pending", "active", "done", "failed", "disabled")


@dataclass(frozen=True)
class Palette:
    """Colour tokens. Rubric red is reserved for the primary action and live recording."""

    page: str
    surface: str
    sidebar: str
    card: str
    input_bg: str
    rule: str
    ink: str
    muted: str
    accent: str
    accent_hover: str
    accent_text: str
    rubric: str
    rubric_hover: str
    done: str
    done_text: str
    bar_active: str
    disabled_bg: str
    disabled_text: str
    log_bg: str
    log_text: str


PALETTES: dict[str, Palette] = {
    "light": Palette(
        page="#EEF1F5",
        surface="#F6F8FB",
        sidebar="#E6EAF1",
        card="#FFFFFF",
        input_bg="#FFFFFF",
        rule="#D3D9E3",
        ink="#26304A",
        muted="#566079",
        accent="#26304A",
        accent_hover="#38456A",
        accent_text="#FFFFFF",
        rubric="#C2412D",
        rubric_hover="#A83624",
        done="#2F7D6B",
        done_text="#FFFFFF",
        bar_active="#9DB2E3",
        disabled_bg="#C5CCD9",
        disabled_text="#6B7488",
        log_bg="#1B2133",
        log_text="#C9D3EC",
    ),
    "dark": Palette(
        page="#141821",
        surface="#191E2A",
        sidebar="#10131A",
        card="#1D2230",
        input_bg="#171B26",
        rule="#2E3546",
        ink="#E3E7F0",
        muted="#98A2B8",
        accent="#3B4A70",
        accent_hover="#4B5D8C",
        accent_text="#F2F5FA",
        rubric="#E2614C",
        rubric_hover="#EE7561",
        done="#4FB39B",
        done_text="#0F1A17",
        bar_active="#4A6AB5",
        disabled_bg="#2A3144",
        disabled_text="#7B859C",
        log_bg="#0E1119",
        log_text="#B9C5E3",
    ),
}


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


def apply_theme(app: QApplication, mode: str) -> str:
    """Apply the Fusion style, bundled font, and QSS. Returns the effective mode."""
    effective = resolve_mode(mode)
    family = load_fonts()
    if app.style().objectName().lower() != "fusion":
        app.setStyle("Fusion")
    app.setFont(QFont(family, 10))
    app.setStyleSheet(build_qss(effective, family))
    return effective


def build_qss(mode: str, family: str = FONT_FAMILY) -> str:
    """Build the application stylesheet for ``mode`` (``light`` or ``dark``)."""
    p = PALETTES["dark" if mode == "dark" else "light"]
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
            padding: 2px 24px;
            font-weight: 700;
            font-size: 11pt;
            min-width: 180px;
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
            color: #FFFFFF;
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
        }}
        QPushButton#exitButton {{
            background: transparent;
            border-color: {p.rule};
            color: {p.muted};
        }}
        QPushButton#exitButton:hover {{
            background: {p.rubric};
            border-color: {p.rubric};
            color: #FFFFFF;
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
        #StageName {{
            font-weight: 600;
        }}
        #StageName[state="pending"], #StageName[state="disabled"] {{
            color: {p.muted};
            font-weight: 500;
        }}
        #StageName[state="failed"] {{
            color: {p.rubric};
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
