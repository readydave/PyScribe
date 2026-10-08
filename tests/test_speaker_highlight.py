from __future__ import annotations

import os
import unittest

from PySide6.QtGui import QTextDocument
from PySide6.QtWidgets import QApplication

from services.ui_tokens import speaker_color
from ui_qt import theme
from ui_qt.speaker_highlight import SpeakerHighlighter


class SpeakerHighlighterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls._app = QApplication.instance() or QApplication([])

    def _formats(self, text: str, mode: str) -> list[tuple[int, int, str]]:
        theme.apply_theme(self._app, mode)
        self.addCleanup(theme.apply_theme, self._app, "system")
        doc = QTextDocument()
        highlighter = SpeakerHighlighter(doc)
        doc.setPlainText(text)
        highlighter.rehighlight()
        ranges = doc.firstBlock().layout().formats()
        return [(r.start, r.length, r.format.foreground().color().name()) for r in ranges]

    def test_label_is_coloured_and_text_is_not(self) -> None:
        formats = self._formats("[S2] Hello there", "light")
        self.assertEqual(formats, [(0, 4, speaker_color("S2", "light").lower())])

    def test_colour_follows_theme(self) -> None:
        light = self._formats("[S1] Hi", "light")[0][2]
        dark = self._formats("[S1] Hi", "dark")[0][2]
        self.assertNotEqual(light, dark)

    def test_unknown_speaker_and_plain_lines(self) -> None:
        self.assertEqual(len(self._formats("[S?] mumble", "light")), 1)
        self.assertEqual(self._formats("No speaker label here", "light"), [])


if __name__ == "__main__":
    unittest.main()
