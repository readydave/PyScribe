"""The Listener follows the studio hierarchy: basics up front, Advanced behind a disclosure, tabs for output."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import pytest

pytest.importorskip("gradio")

import gradio as gr  # noqa: E402

import app  # noqa: E402


def build():
    with (
        patch.object(app, "_LISTENER_RUNTIME_READY", True),
        patch.object(app, "ALL_MODELS", ["small"]),
        patch.object(app, "RECOMMENDED_MODEL", "small"),
        patch.object(app, "AVAILABLE_DIAR_BACKENDS", ["accurate"]),
    ):
        return app.create_interface()


def blocks_of(iface, kind, label=None):
    found = [b for b in iface.blocks.values() if isinstance(b, kind)]
    return [b for b in found if label is None or getattr(b, "label", None) == label]


def ancestors(block):
    chain = []
    while getattr(block, "parent", None) is not None:
        block = block.parent
        chain.append(block)
    return chain


class ListenerLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.iface = build()

    def test_model_and_speaker_toggle_are_up_front_not_in_advanced(self) -> None:
        (model,) = blocks_of(self.iface, gr.Dropdown, "Select Transcription Model")
        (speakers,) = blocks_of(self.iface, gr.Checkbox, "Identify speakers")
        for block in (model, speakers):
            self.assertFalse([a for a in ancestors(block) if isinstance(a, gr.Accordion)], block.label)

    def test_tuning_lives_in_a_closed_advanced_accordion(self) -> None:
        (advanced,) = blocks_of(self.iface, gr.Accordion, "Advanced")
        self.assertFalse(advanced.open)
        for label in ("Run mode", "Names / terms (optional)", "Faster GPU decoding (may drop short utterances)"):
            (block,) = [b for b in self.iface.blocks.values() if getattr(b, "label", None) == label]
            self.assertIn(advanced, ancestors(block), label)

    def test_transcript_and_ai_summary_are_tabs(self) -> None:
        tabs = [t.label for t in blocks_of(self.iface, gr.Tab)]
        self.assertEqual(tabs, ["Transcript", "AI summary"])
        (transcript,) = blocks_of(self.iface, gr.Textbox, "Transcription")
        (ai_profile,) = blocks_of(self.iface, gr.Dropdown, "LLM profile")
        self.assertTrue([a for a in ancestors(transcript) if isinstance(a, gr.Tab) and a.label == "Transcript"])
        self.assertTrue([a for a in ancestors(ai_profile) if isinstance(a, gr.Tab) and a.label == "AI summary"])

    def test_no_llm_accordion_is_left_and_help_moved_to_the_bottom(self) -> None:
        self.assertEqual(blocks_of(self.iface, gr.Accordion, "LLM Post-Processing (Beta)"), [])
        labels = [a.label for a in blocks_of(self.iface, gr.Accordion)]
        self.assertEqual(labels, ["Advanced", "About models and LLM post-processing"])

    def test_footer_states_where_processing_happens(self) -> None:
        values = [getattr(b, "value", "") for b in blocks_of(self.iface, gr.Markdown)]
        self.assertIn("All processing happens on the machine running this Listener.", values)

    def test_footer_style_follows_the_theme_and_text_size(self) -> None:
        css = app.build_css()
        self.assertIn(".pyscribe-footer", css)
        self.assertIn("--pyscribe-muted", css)
        self.assertIn("var(--text-sm)", css.split(".pyscribe-footer", 1)[1].split("}", 1)[0])

    def test_every_event_handler_is_still_registered(self) -> None:
        # The layout moved components between containers; the 14 handlers (transcribe, visibility, LLM...) must be unchanged.
        self.assertEqual(len(self.iface.fns), 14)
        handlers = sorted(getattr(dep.fn, "__name__", "") for dep in self.iface.fns.values())
        for name in ("transcribe", "_update_mode_visibility"):
            self.assertIn(name, handlers)


if __name__ == "__main__":
    unittest.main()
