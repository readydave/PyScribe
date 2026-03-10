"""Tests for visual OCR helper logic."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from services.multimodal_service import (
    _format_visual_report,
    _is_low_value_chat_line,
    _is_low_value_slide_line,
    _prepare_verified_paddle_ocr_model_dirs,
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

    def test_prepare_verified_paddle_ocr_model_dirs_uses_expected_hf_repos(self) -> None:
        class _FakePaddleOCR:
            @staticmethod
            def _get_ocr_model_names(_self, lang, ppocr_version):
                self.assertEqual(lang, "en")
                self.assertIsNone(ppocr_version)
                return "PP-OCRv5_server_det", "en_PP-OCRv5_mobile_rec"

        verified_calls: list[tuple[str, str]] = []
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            "os.environ",
            {
                "PADDLE_PDX_MODEL_SOURCE": "huggingface",
                "PADDLE_PDX_CACHE_HOME": temp_dir,
            },
            clear=False,
        ), patch(
            "services.multimodal_service.ensure_hf_repo_local_dir_verified",
            side_effect=lambda repo_id, local_dir, **_: verified_calls.append((repo_id, local_dir)) or str(local_dir),
        ):
            kwargs = _prepare_verified_paddle_ocr_model_dirs(_FakePaddleOCR)

        base_dir = Path(temp_dir) / "official_models"
        self.assertEqual(kwargs["doc_orientation_classify_model_dir"], str(base_dir / "PP-LCNet_x1_0_doc_ori"))
        self.assertEqual(kwargs["doc_unwarping_model_dir"], str(base_dir / "UVDoc"))
        self.assertEqual(kwargs["text_detection_model_dir"], str(base_dir / "PP-OCRv5_server_det"))
        self.assertEqual(kwargs["textline_orientation_model_dir"], str(base_dir / "PP-LCNet_x1_0_textline_ori"))
        self.assertEqual(kwargs["text_recognition_model_dir"], str(base_dir / "en_PP-OCRv5_mobile_rec"))
        self.assertEqual(
            [repo_id for repo_id, _ in verified_calls],
            [
                "PaddlePaddle/PP-LCNet_x1_0_doc_ori",
                "PaddlePaddle/UVDoc",
                "PaddlePaddle/PP-OCRv5_server_det",
                "PaddlePaddle/PP-LCNet_x1_0_textline_ori",
                "PaddlePaddle/en_PP-OCRv5_mobile_rec",
            ],
        )


if __name__ == "__main__":
    unittest.main()
