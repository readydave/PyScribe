"""Tests for merging ASR segments with diarization turns."""

from __future__ import annotations

import unittest

from services.speaker_assignment import assign_speakers


def _words(*items: tuple[float, float, str]) -> list[dict]:
    return [{"start": s, "end": e, "word": w, "probability": 0.9} for s, e, w in items]


class SegmentLevelTests(unittest.TestCase):
    def test_maximum_overlap_label_without_words(self) -> None:
        asr = [{"start": 0.0, "end": 4.0, "text": "hi"}, {"start": 10.0, "end": 12.0, "text": "bye"}]
        turns = [{"start": 0.0, "end": 3.0, "speaker": "S1"}, {"start": 3.0, "end": 4.0, "speaker": "S2"}]
        out = assign_speakers(asr, turns)
        self.assertEqual([s["speaker"] for s in out], ["S1", "S?"])


class WordLevelTests(unittest.TestCase):
    def test_splits_segment_at_speaker_change(self) -> None:
        asr = [
            {
                "start": 0.0,
                "end": 4.0,
                "text": "hello there general kenobi",
                "words": _words((0.0, 0.8, " hello"), (0.9, 1.8, " there"), (2.2, 3.0, " general"), (3.1, 4.0, " kenobi")),
            }
        ]
        turns = [{"start": 0.0, "end": 2.0, "speaker": "S1"}, {"start": 2.0, "end": 4.0, "speaker": "S2"}]
        out = assign_speakers(asr, turns)
        self.assertEqual([(s["speaker"], s["text"]) for s in out], [("S1", "hello there"), ("S2", "general kenobi")])
        self.assertEqual(out[1]["start"], 2.2)

    def test_word_in_gap_snaps_to_nearest_turn(self) -> None:
        asr = [{"start": 0.0, "end": 2.0, "text": "a b", "words": _words((0.0, 0.5, " a"), (1.1, 1.4, " b"))}]
        turns = [{"start": 0.0, "end": 0.6, "speaker": "S1"}, {"start": 1.6, "end": 3.0, "speaker": "S2"}]
        out = assign_speakers(asr, turns)
        self.assertEqual([s["speaker"] for s in out], ["S1", "S2"])

    def test_far_word_inherits_previous_speaker(self) -> None:
        asr = [{"start": 0.0, "end": 9.0, "text": "a b", "words": _words((0.0, 0.5, " a"), (8.0, 8.5, " b"))}]
        turns = [{"start": 0.0, "end": 1.0, "speaker": "S1"}]
        out = assign_speakers(asr, turns)
        self.assertEqual([(s["speaker"], s["text"]) for s in out], [("S1", "a b")])

    def test_no_turns_gives_unknown(self) -> None:
        asr = [{"start": 0.0, "end": 1.0, "text": "a", "words": _words((0.0, 0.5, " a"))}]
        self.assertEqual(assign_speakers(asr, [])[0]["speaker"], "S?")

    def test_adjacent_same_speaker_pieces_merge(self) -> None:
        asr = [
            {"start": 0.0, "end": 1.0, "text": "one", "words": _words((0.0, 1.0, " one"))},
            {"start": 1.2, "end": 2.0, "text": "two", "words": _words((1.2, 2.0, " two"))},
        ]
        turns = [{"start": 0.0, "end": 3.0, "speaker": "S1"}]
        out = assign_speakers(asr, turns)
        self.assertEqual([(s["speaker"], s["text"]) for s in out], [("S1", "one two")])

    def test_mixed_segments_fall_back_per_segment(self) -> None:
        asr = [
            {"start": 0.0, "end": 1.0, "text": "w", "words": _words((0.0, 1.0, " w"))},
            {"start": 5.0, "end": 6.0, "text": "plain"},
        ]
        turns = [{"start": 0.0, "end": 1.0, "speaker": "S1"}, {"start": 5.0, "end": 6.0, "speaker": "S2"}]
        out = assign_speakers(asr, turns)
        self.assertEqual([(s["speaker"], s["text"]) for s in out], [("S1", "w"), ("S2", "plain")])

    def test_overlapping_turns_pick_largest_overlap(self) -> None:
        asr = [{"start": 0.0, "end": 2.0, "text": "x", "words": _words((0.0, 2.0, " x"))}]
        turns = [{"start": 0.0, "end": 0.5, "speaker": "S1"}, {"start": 0.0, "end": 2.0, "speaker": "S2"}]
        self.assertEqual(assign_speakers(asr, turns)[0]["speaker"], "S2")


if __name__ == "__main__":
    unittest.main()
