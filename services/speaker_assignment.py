"""Merge ASR output with diarization turns (word-level when word timestamps exist)."""

from __future__ import annotations

from bisect import bisect_left
from typing import Any

UNKNOWN_SPEAKER = "S?"
# A word that falls in a gap between turns snaps to the nearest turn within this distance.
NEAREST_TURN_TOLERANCE_SECONDS = 0.75
# Adjacent same-speaker pieces closer than this are merged back into one line.
MERGE_GAP_SECONDS = 1.5

Segment = dict[str, Any]


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speakers_by_segment(asr_segments: list[Segment], spk_segments: list[Segment]) -> list[Segment]:
    """Label each ASR segment with the speaker that overlaps it most."""
    for seg in asr_segments:
        best_spk = None
        best_ov = 0.0
        for spk in spk_segments:
            ov = _overlap(seg["start"], seg["end"], spk["start"], spk["end"])
            if ov > best_ov:
                best_ov = ov
                best_spk = spk["speaker"]
        seg["speaker"] = best_spk or UNKNOWN_SPEAKER
    return asr_segments


class _TurnIndex:
    """Speaker turns sorted by start for fast per-word lookups."""

    def __init__(self, turns: list[Segment]) -> None:
        self.turns = sorted(turns, key=lambda t: t["start"])
        self.starts = [t["start"] for t in self.turns]
        self.max_len = max((t["end"] - t["start"] for t in self.turns), default=0.0)

    def speaker_for(self, start: float, end: float) -> str | None:
        """Speaker with the largest overlap with [start, end], else the nearest turn within tolerance."""
        if not self.turns:
            return None
        hi = bisect_left(self.starts, end)
        best_spk, best_ov = None, 0.0
        for i in range(hi - 1, -1, -1):
            turn = self.turns[i]
            if turn["start"] < start - self.max_len:
                break
            ov = _overlap(start, end, turn["start"], turn["end"])
            if ov > best_ov:
                best_spk, best_ov = turn["speaker"], ov
        if best_spk is not None:
            return best_spk
        mid = (start + end) / 2.0
        nearest, nearest_dist = None, NEAREST_TURN_TOLERANCE_SECONDS
        lo = max(0, bisect_left(self.starts, mid) - 2)
        for turn in self.turns[lo : lo + 6]:
            dist = max(turn["start"] - mid, mid - turn["end"], 0.0)
            if dist <= nearest_dist:
                nearest, nearest_dist = turn["speaker"], dist
        return nearest


def _word_text(words: list[dict[str, Any]]) -> str:
    return "".join(str(w["word"]) for w in words).strip()


def _split_segment_by_words(seg: Segment, index: _TurnIndex, previous_speaker: str | None) -> list[Segment]:
    pieces: list[Segment] = []
    current_speaker = previous_speaker
    for word in seg["words"]:
        speaker = index.speaker_for(word["start"], word["end"]) or current_speaker or UNKNOWN_SPEAKER
        if pieces and speaker == pieces[-1]["speaker"]:
            pieces[-1]["words"].append(word)
        else:
            pieces.append({"speaker": speaker, "words": [word]})
        current_speaker = speaker
    return [
        {
            "start": piece["words"][0]["start"],
            "end": piece["words"][-1]["end"],
            "text": _word_text(piece["words"]),
            "speaker": piece["speaker"],
            "words": piece["words"],
        }
        for piece in pieces
        if _word_text(piece["words"])
    ]


def _merge_adjacent(pieces: list[Segment]) -> list[Segment]:
    merged: list[Segment] = []
    for piece in pieces:
        last = merged[-1] if merged else None
        if (
            last is not None
            and "words" in last
            and "words" in piece
            and last["speaker"] == piece["speaker"]
            and piece["start"] - last["end"] <= MERGE_GAP_SECONDS
        ):
            last["words"] = last["words"] + piece["words"]
            last["text"] = f"{last['text']} {piece['text']}".strip()
            last["end"] = piece["end"]
        else:
            merged.append(piece)
    return merged


def assign_speakers(asr_segments: list[Segment], spk_segments: list[Segment]) -> list[Segment]:
    """Attach a `speaker` to every ASR segment.

    Segments carrying word timestamps are labelled per word and split where the
    speaker changes, so a turn in the middle of a segment is not lost. Segments
    without words keep the whole-segment maximum-overlap label.
    """
    if not any(seg.get("words") for seg in asr_segments):
        return assign_speakers_by_segment(asr_segments, spk_segments)

    index = _TurnIndex(spk_segments)
    result: list[Segment] = []
    previous_speaker: str | None = None
    for seg in asr_segments:
        if seg.get("words"):
            pieces = _split_segment_by_words(seg, index, previous_speaker)
            if not pieces:
                continue
            result.extend(pieces)
            previous_speaker = pieces[-1]["speaker"]
        else:
            labelled = assign_speakers_by_segment([seg], spk_segments)[0]
            result.append(labelled)
            previous_speaker = labelled["speaker"]
    return _merge_adjacent(result)
