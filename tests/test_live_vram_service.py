"""Tests for live transcription VRAM preflight helpers."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from services.live_vram_service import (
    GpuMemoryInfo,
    assess_live_vram_preflight,
    estimate_live_model_vram_gb,
)


class LiveVramServiceTests(unittest.TestCase):
    def test_estimate_known_live_model_vram(self) -> None:
        self.assertEqual(
            estimate_live_model_vram_gb("deepdml/faster-whisper-large-v3-turbo-ct2", compute_type="float16"),
            4.0,
        )

    def test_estimate_adjusts_int8_compute(self) -> None:
        self.assertEqual(estimate_live_model_vram_gb("large-v3", compute_type="int8"), 4.88)

    def test_assess_skips_cpu_mode(self) -> None:
        result = assess_live_vram_preflight(
            "large-v3",
            device="cpu",
            compute_type="int8",
            memory_info=GpuMemoryInfo(total_gb=8.0, free_gb=1.0, used_gb=7.0, source="test"),
        )

        self.assertEqual(result.status, "skipped")
        self.assertFalse(result.should_warn)
        self.assertIsNone(result.free_gb)

    def test_assess_warns_when_free_vram_below_estimate(self) -> None:
        result = assess_live_vram_preflight(
            "large-v3",
            device="cuda",
            compute_type="float16",
            memory_info=GpuMemoryInfo(total_gb=12.0, free_gb=2.0, used_gb=10.0, source="test"),
        )

        self.assertEqual(result.status, "low")
        self.assertTrue(result.should_warn)
        self.assertEqual(result.estimated_required_gb, 7.5)
        self.assertIn("large-v3", result.message)

    def test_assess_accepts_enough_free_vram(self) -> None:
        result = assess_live_vram_preflight(
            "large-v3",
            device="cuda",
            compute_type="float16",
            memory_info=GpuMemoryInfo(total_gb=12.0, free_gb=8.0, used_gb=4.0, source="test"),
        )

        self.assertEqual(result.status, "ok")
        self.assertFalse(result.should_warn)

    def test_assess_unavailable_memory_data_continues_without_warning(self) -> None:
        with patch("services.live_vram_service.get_gpu_memory_info", return_value=None):
            result = assess_live_vram_preflight(
                "large-v3",
                device="cuda",
                compute_type="float16",
                memory_info=None,
            )

        self.assertEqual(result.status, "unavailable")
        self.assertFalse(result.should_warn)


if __name__ == "__main__":
    unittest.main()
