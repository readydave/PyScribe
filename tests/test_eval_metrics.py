"""Tests for evaluation metrics (CPU-safe; skip when optional deps are missing)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from services import eval_metrics as em

try:
    import jiwer  # noqa: F401
except ImportError:  # pragma: no cover - depends on environment
    jiwer = None

try:
    import pyannote.metrics  # noqa: F401
except ImportError:  # pragma: no cover - depends on environment
    pyannote = None


class NormalizeTextTests(unittest.TestCase):
    def test_lowercases_and_strips_punctuation(self) -> None:
        self.assertEqual(em.normalize_text("Hello, World! It’s fine."), "hello world it's fine")

    def test_keeps_accents_and_splits_dashes(self) -> None:
        self.assertEqual(em.normalize_text("Napoleón—en Chamartín"), "napoleón en chamartín")


@unittest.skipIf(jiwer is None, "jiwer not installed")
class WerTests(unittest.TestCase):
    def test_identical_text_is_zero(self) -> None:
        scores = em.compute_wer_cer("The cat sat.", "the cat sat")
        self.assertEqual(scores["wer"], 0.0)
        self.assertEqual(scores["cer"], 0.0)

    def test_one_substitution(self) -> None:
        scores = em.compute_wer_cer("the cat sat down", "the cat sit down")
        self.assertAlmostEqual(scores["wer"], 0.25)

    def test_empty_reference_raises(self) -> None:
        with self.assertRaises(ValueError):
            em.compute_wer_cer("...", "anything")


class SpeakerCountTests(unittest.TestCase):
    def test_parse_rttm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ref.rttm"
            path.write_text(
                "SPEAKER f 1 0.00 2.50 <NA> <NA> A <NA> <NA>\n"
                "SPEAKER f 1 2.50 1.50 <NA> <NA> B <NA> <NA>\n",
                encoding="utf-8",
            )
            self.assertEqual(
                em.parse_rttm(path),
                [
                    {"start": 0.0, "end": 2.5, "speaker": "A"},
                    {"start": 2.5, "end": 4.0, "speaker": "B"},
                ],
            )

    def test_count_unknown_speakers(self) -> None:
        segs = [{"speaker": "S1"}, {"speaker": "S?"}, {}]
        self.assertEqual(em.count_unknown_speakers(segs), 2)

    def test_mislabelled_segments_uses_best_mapping(self) -> None:
        ref = [
            {"start": 0.0, "end": 5.0, "speaker": "A"},
            {"start": 5.0, "end": 10.0, "speaker": "B"},
        ]
        asr = [
            {"start": 0.0, "end": 2.5, "speaker": "S1"},
            {"start": 2.5, "end": 5.0, "speaker": "S1"},
            {"start": 5.0, "end": 8.0, "speaker": "S2"},
            {"start": 8.0, "end": 10.0, "speaker": "S1"},
        ]
        self.assertEqual(em.count_mislabelled_segments(asr, ref), 1)


@unittest.skipIf(pyannote is None, "pyannote.metrics not installed")
class DerTests(unittest.TestCase):
    def test_perfect_hypothesis_is_zero(self) -> None:
        ref = [
            {"start": 0.0, "end": 5.0, "speaker": "A"},
            {"start": 5.0, "end": 10.0, "speaker": "B"},
        ]
        hyp = [
            {"start": 0.0, "end": 5.0, "speaker": "S2"},
            {"start": 5.0, "end": 10.0, "speaker": "S1"},
        ]
        self.assertAlmostEqual(em.compute_der(ref, hyp), 0.0)

    def test_missed_speech_raises_der(self) -> None:
        ref = [{"start": 0.0, "end": 10.0, "speaker": "A"}]
        hyp = [{"start": 0.0, "end": 5.0, "speaker": "S1"}]
        self.assertGreater(em.compute_der(ref, hyp, collar=0.0), 0.4)


class RealTimeFactorTests(unittest.TestCase):
    def test_rtf(self) -> None:
        self.assertAlmostEqual(em.real_time_factor(5.0, 100.0), 0.05)
        self.assertEqual(em.real_time_factor(5.0, 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
