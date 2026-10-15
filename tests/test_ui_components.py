from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QWidget

from services import config_service
from ui_qt import theme
from ui_qt.components import (
    AppButton,
    Card,
    Chip,
    Drawer,
    EmptyState,
    IconButton,
    InspectorTabs,
    ListRow,
    NavRail,
    SegmentedControl,
    tokens,
)


class ComponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        tokens.configure_motion("on")
        self.addCleanup(tokens.configure_motion, "auto")
        theme.apply_theme(self._app, "dark")
        self.addCleanup(theme.apply_theme, self._app, "system")

    def test_button_variants_and_icon_accessibility(self) -> None:
        button = AppButton("Go", "primary")
        self.assertEqual(button.property("variant"), "primary")
        button.set_variant("bogus")
        self.assertEqual(button.property("variant"), "secondary")
        icon = IconButton("✕", "Close panel")
        self.assertEqual(icon.accessibleName(), "Close panel")
        self.assertEqual(icon.property("variant"), "icon")

    def test_interactive_card_activates_on_click_and_keys(self) -> None:
        card = Card(interactive=True)
        card.resize(100, 40)
        card.show()
        hits: list[int] = []
        card.clicked.connect(lambda: hits.append(1))
        QTest.mouseClick(card, Qt.LeftButton)
        QTest.keyClick(card, Qt.Key_Return)
        QTest.keyClick(card, Qt.Key_Space)
        self.assertEqual(len(hits), 3)
        card.set_selected(True)
        self.assertTrue(card.property("selected"))

    def test_plain_card_does_not_emit(self) -> None:
        card = Card()
        card.resize(100, 40)
        card.show()
        hits: list[int] = []
        card.clicked.connect(lambda: hits.append(1))
        QTest.mouseClick(card, Qt.LeftButton)
        self.assertEqual(hits, [])

    def test_segmented_control_is_exclusive(self) -> None:
        control = SegmentedControl([("a", "A"), ("b", "B"), ("c", "C")])
        seen: list[str] = []
        control.changed.connect(seen.append)
        self.assertEqual(control.current(), "a")
        QTest.mouseClick(control._buttons["c"], Qt.LeftButton)
        self.assertEqual(control.current(), "c")
        self.assertEqual(seen, ["c"])
        self.assertEqual(control._buttons["a"].property("segment"), "left")
        self.assertEqual(control._buttons["b"].property("segment"), "mid")
        self.assertEqual(control._buttons["c"].property("segment"), "right")

    def test_chip_variant(self) -> None:
        chip = Chip("Local only", "success")
        self.assertEqual(chip.property("chip"), "success")
        chip.set_variant("nope")
        self.assertEqual(chip.property("chip"), "neutral")

    def test_empty_state_action(self) -> None:
        state = EmptyState("Nothing yet", "Import a recording.", "New transcription")
        hits: list[int] = []
        state.action_clicked.connect(lambda: hits.append(1))
        state.action_button.click()
        self.assertEqual(hits, [1])
        self.assertIsNone(EmptyState("t", "m").action_button)

    def test_list_row_activates(self) -> None:
        row = ListRow("Quarterly Review", "52:17")
        row.resize(200, 40)
        row.show()
        hits: list[int] = []
        row.activated.connect(lambda: hits.append(1))
        QTest.keyClick(row, Qt.Key_Return)
        self.assertEqual(hits, [1])
        self.assertTrue(row.subtitle_label.isVisibleTo(row))
        bare = ListRow("x")
        self.assertFalse(bare.subtitle_label.isVisibleTo(bare))

    def test_drawer_opens_and_closes_instantly_with_reduced_motion(self) -> None:
        host = QWidget()
        drawer = Drawer("Performance", width=300, parent=host)
        host.resize(600, 300)
        host.show()
        states: list[bool] = []
        drawer.visibility_changed.connect(states.append)
        self.assertFalse(drawer.isVisible())
        drawer.open_drawer()
        self.assertTrue(drawer.is_open())
        self.assertTrue(drawer.isVisible())
        self.assertEqual(drawer.maximumWidth(), 300)
        drawer.setFocus()
        QTest.keyClick(drawer, Qt.Key_Escape)
        self.assertFalse(drawer.is_open())
        self.assertFalse(drawer.isVisible())
        self.assertEqual(states, [True, False])

    def test_drawer_animates_when_motion_allowed(self) -> None:
        tokens.configure_motion("off")
        host = QWidget()
        drawer = Drawer("Details", width=300, parent=host)
        host.show()
        drawer.open_drawer()
        self.assertTrue(drawer._animation.duration() > 0)
        drawer._animation.stop()

    def test_inspector_tabs_one_view_and_collapse(self) -> None:
        tabs = InspectorTabs()
        for key, label in (("summary", "Summary"), ("actions", "Action items"), ("speakers", "Speakers")):
            tabs.add_tab(key, label, QLabel(label))
        tabs.resize(400, 300)
        tabs.show()
        self.assertEqual(tabs.current(), "summary")
        tabs.set_current("speakers")
        self.assertEqual(tabs.current(), "speakers")
        self.assertEqual(sum(1 for b in tabs._buttons.values() if b.isChecked()), 1)
        tabs.set_collapsed(True)
        self.assertTrue(tabs.is_collapsed())
        self.assertFalse(tabs.stack.isVisible())
        self.assertEqual(tabs.maximumWidth(), 44)
        tabs.set_current("summary")  # picking a tab re-expands
        self.assertFalse(tabs.is_collapsed())
        self.assertTrue(tabs.stack.isVisible())

    def test_nav_rail_selection_and_arrow_keys(self) -> None:
        rail = NavRail([("projects", "Projects"), ("transcribe", "Transcribe"), ("live", "Live")])
        rail.show()
        seen: list[str] = []
        rail.selected.connect(seen.append)
        rail.set_current("projects")
        QTest.keyClick(rail, Qt.Key_Down)
        self.assertEqual(rail.current(), "transcribe")
        QTest.keyClick(rail, Qt.Key_Up)
        QTest.keyClick(rail, Qt.Key_Up)
        self.assertEqual(rail.current(), "live")  # wraps
        self.assertEqual(seen, ["projects", "transcribe", "projects", "live"])
        self.assertEqual(rail.button("live").accessibleName(), "Live")

    def test_stylesheet_has_component_selectors(self) -> None:
        qss = self._app.styleSheet()
        for selector in ('QPushButton[variant="primary"]', "QPushButton[nav]", "QPushButton[tab]",
                         'QFrame[studioCard="true"]', "#Drawer", 'QLabel[chip="success"]', "#TranscriptEditor"):
            self.assertIn(selector, qss)


class MotionAndConfigTests(unittest.TestCase):
    def tearDown(self) -> None:
        tokens.configure_motion("auto")
        tokens.set_transcript_px(tokens.TRANSCRIPT_PX_DEFAULT)
        os.environ.pop("PYSCRIBE_REDUCED_MOTION", None)

    def test_motion_setting(self) -> None:
        self.assertTrue(tokens.configure_motion("on"))
        self.assertEqual(tokens.motion_ms("base"), 0)
        self.assertFalse(tokens.configure_motion("off"))
        self.assertEqual(tokens.motion_ms("base"), tokens.MOTION_MS["base"])
        os.environ["PYSCRIBE_REDUCED_MOTION"] = "1"
        self.assertTrue(tokens.configure_motion("auto"))
        os.environ.pop("PYSCRIBE_REDUCED_MOTION")
        self.assertFalse(tokens.configure_motion("auto"))

    def test_transcript_px_is_clamped(self) -> None:
        self.assertEqual(tokens.set_transcript_px(5), tokens.TRANSCRIPT_PX_MIN)
        self.assertEqual(tokens.set_transcript_px(99), tokens.TRANSCRIPT_PX_MAX)
        self.assertEqual(tokens.set_transcript_px(18), 18)

    def _load(self, payload: dict) -> config_service.AppConfig:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return config_service.load_config(path)

    def test_old_config_gets_defaults(self) -> None:
        cfg = self._load({})
        self.assertEqual(cfg.reduced_motion, "auto")
        self.assertEqual(cfg.transcript_font_px, 16)

    def test_bad_values_are_sanitized(self) -> None:
        cfg = self._load({"reduced_motion": "sometimes", "transcript_font_px": "huge"})
        self.assertEqual((cfg.reduced_motion, cfg.transcript_font_px), ("auto", 16))
        cfg = self._load({"reduced_motion": "ON", "transcript_font_px": 400})
        self.assertEqual((cfg.reduced_motion, cfg.transcript_font_px), ("on", 28))
        cfg = self._load({"transcript_font_px": 1e999})
        self.assertEqual(cfg.transcript_font_px, 16)
        cfg = self._load({"transcript_font_px": True})
        self.assertEqual(cfg.transcript_font_px, 16)


if __name__ == "__main__":
    unittest.main()
