"""Tests for visual OCR helper logic."""

from __future__ import annotations

from collections import Counter
import unittest

from services.multimodal_service import (
    _format_visual_report,
    _is_low_value_chat_line,
    _is_low_value_slide_line,
    _looks_like_person_name,
    _resolve_effective_sample_seconds,
)


class MultimodalServiceTests(unittest.TestCase):
    def test_resolve_effective_sample_seconds_spreads_sampling_across_full_video(self) -> None:
        self.assertEqual(
            _resolve_effective_sample_seconds(
                requested_sample_seconds=1.0,
                max_frames=180,
                media_duration_seconds=2232.0,
            ),
            12.4,
        )
        self.assertEqual(
            _resolve_effective_sample_seconds(
                requested_sample_seconds=4.0,
                max_frames=360,
                media_duration_seconds=900.0,
            ),
            4.0,
        )

    def test_name_like_lines_are_filtered_more_aggressively(self) -> None:
        self.assertTrue(_looks_like_person_name("Allison M. Owens"))
        self.assertTrue(_is_low_value_slide_line("Paul C. Phelps"))
        self.assertTrue(_is_low_value_chat_line("1359"))
        self.assertTrue(_is_low_value_chat_line("Billy l.Stuecken"))
        self.assertFalse(_is_low_value_slide_line("Visits per DVM Day [0.07, 0.43]"))
        self.assertFalse(_is_low_value_chat_line("Ralph: I thought Kir was a troubleshooter."))

    def test_visual_report_prefers_meaningful_slide_and_chat_lines(self) -> None:
        canonical_lines = {
            "allison m owens": "Allison M. Owens",
            "metric": "Visits per DVM Day [0.07, 0.43]",
            "pareto": "Pareto Chart: Euthanasia Reasons (Excluding Other)",
            "clock": "1359",
            "chatmsg": "Ralph: I thought Kir was a troubleshooter.",
        }
        line_counts = Counter(
            {
                "allison m owens": 20,
                "metric": 11,
                "pareto": 10,
                "clock": 30,
                "chatmsg": 8,
            }
        )
        line_source = {
            "allison m owens": "slide",
            "metric": "slide",
            "pareto": "slide",
            "clock": "chat",
            "chatmsg": "chat",
        }

        report = _format_visual_report(
            partial=False,
            sample_seconds=12.4,
            frames_scanned=180,
            visual_profile="fast",
            requested_backend="auto",
            ocr_name="rapidocr",
            backend_note=None,
            ocr_calls_slide=30,
            ocr_calls_chat=6,
            dedupe_skipped_slide=150,
            dedupe_skipped_chat=174,
            canonical_lines=canonical_lines,
            line_counts=line_counts,
            line_source=line_source,
            total_frames=180,
            timeline=[],
        )

        self.assertIn("Visits per DVM Day [0.07, 0.43]", report)
        self.assertIn("Ralph: I thought Kir was a troubleshooter.", report)
        self.assertNotIn("Allison M. Owens", report)
        self.assertNotIn("1359", report)


if __name__ == "__main__":
    unittest.main()
