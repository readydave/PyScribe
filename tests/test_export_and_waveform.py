from __future__ import annotations

import json
import math
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from services import export_service as export
from services import waveform_service as waveform
from services.project_store import Project, ProjectStore, sanitize_segments


def project(**overrides: object) -> Project:
    segments = sanitize_segments(
        [
            {"start": 0.0, "end": 4.2, "speaker": "S1", "text": "Thanks   for\njoining."},
            {"start": 4.2, "end": 9.0, "speaker": "S2", "text": "Happy to be here."},
            {"start": 3725.5, "end": 3725.6, "speaker": None, "text": "Late and short -->"},
        ]
    )
    base = dict(
        id="20260101-000000-abcdef12", title="Quarterly Review", segments=segments, duration_s=3726.0,
        speaker_names={"S1": "Sarah"}, media_path="/home/dave/private/rec.mp4",
    )
    base.update(overrides)
    return Project(**base)  # type: ignore[arg-type]


class ExportTests(unittest.TestCase):
    def test_timestamps(self) -> None:
        self.assertEqual(export.timestamp(3725.5), "01:02:05.500")
        self.assertEqual(export.timestamp(3725.5, comma=True), "01:02:05,500")
        self.assertEqual(export.timestamp(65, hours=False), "01:05")
        self.assertEqual(export.timestamp(-3), "00:00:00.000")

    def test_txt_uses_names_and_collapses_whitespace(self) -> None:
        lines = export.export_txt(project()).splitlines()
        self.assertEqual(lines[0], "[00:00] Sarah: Thanks for joining.")
        self.assertEqual(lines[1], "[00:04] S2: Happy to be here.")
        self.assertEqual(lines[2], "[62:05] Late and short -->")

    def test_srt_structure_and_minimum_cue_length(self) -> None:
        blocks = export.export_srt(project()).strip().split("\n\n")
        self.assertEqual(blocks[0], "1\n00:00:00,000 --> 00:00:04,200\nSarah: Thanks for joining.")
        self.assertTrue(blocks[2].startswith("3\n01:02:05,500 --> 01:02:06,000\n"))  # 0.1 s cue stretched to 0.5 s

    def test_vtt_header_and_arrow_in_text_is_defused(self) -> None:
        text = export.export_vtt(project())
        self.assertTrue(text.startswith("WEBVTT\n"))
        self.assertIn("00:00:00.000 --> 00:00:04.200\nSarah: Thanks for joining.", text)
        self.assertEqual(text.count("-->"), 3)  # only the three cue arrows; the one in the speech became "->"

    def test_markdown_has_title_facts_and_speakers(self) -> None:
        text = export.export_md(project())
        self.assertTrue(text.startswith("# Quarterly Review\n\n1:02:06 · 2 speakers\n"))
        self.assertIn("**Sarah** `00:00`  \nThanks for joining.", text)

    def test_json_round_trips_and_omits_local_paths(self) -> None:
        raw = export.export_json(project())
        data = json.loads(raw)
        self.assertEqual(data["speakers"], {"S1": "Sarah", "S2": "S2"})
        self.assertEqual(data["segments"][0]["speaker"], "Sarah")
        self.assertIsNone(data["segments"][2]["speaker"])
        for fmt in export.FORMAT_BY_KEY:
            self.assertNotIn("/home/dave/private", export.render(project(), fmt))

    def test_unknown_speaker_labels_are_not_printed(self) -> None:
        p = project(segments=sanitize_segments([{"start": 0, "end": 1, "speaker": "S?", "text": "hm"}]))
        self.assertEqual(export.export_txt(p), "[00:00] hm\n")

    def test_empty_project_exports_cleanly(self) -> None:
        p = project(segments=[], duration_s=0)
        self.assertEqual(export.export_txt(p), "")
        self.assertEqual(export.export_srt(p), "")
        self.assertEqual(export.export_vtt(p).strip(), "WEBVTT")
        self.assertEqual(json.loads(export.export_json(p))["segments"], [])

    def test_unsupported_format_and_filenames(self) -> None:
        with self.assertRaises(ValueError):
            export.render(project(), "docx")
        self.assertEqual(export.suggested_filename(project(title="Q3: Plan/Review?"), "srt"), "Q3_PlanReview.srt")
        self.assertEqual(export.suggested_filename(project(title="///"), "txt"), "transcript.txt")

    def test_export_to_path_writes_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "out.txt"
            export.export_to_path(project(title="Café"), "txt", target)
            self.assertIn("Sarah", target.read_text(encoding="utf-8"))
            with self.assertRaises(OSError):
                export.export_to_path(project(), "txt", Path(tmp) / "missing" / "out.txt")


def write_wav(path: Path, seconds: float = 4.0, loud_from: float = 2.0) -> None:
    rate = 8000
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        frames = bytearray()
        for i in range(int(rate * seconds)):
            amp = 20000 if i / rate >= loud_from else 2000
            frames += struct.pack("<h", int(amp * math.sin(i / 7.0)))
        handle.writeframes(bytes(frames))


class WaveformTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.wav = Path(self._tmp.name) / "tone.wav"
        write_wav(self.wav)

    def test_peaks_are_normalised_and_follow_loudness(self) -> None:
        peaks = waveform.compute_peaks(str(self.wav), buckets=100)
        self.assertEqual(len(peaks), 100)
        self.assertEqual(max(peaks), 1.0)
        self.assertLess(max(peaks[:40]), 0.2)  # quiet first half
        self.assertGreater(min(peaks[60:]), 0.8)  # loud second half

    def test_ffmpeg_fallback_matches_shape(self) -> None:
        with patch.object(waveform, "_peaks_from_soundfile", return_value=None):
            peaks = waveform.compute_peaks(str(self.wav), buckets=100)
        self.assertEqual(len(peaks), 100)
        self.assertLess(max(peaks[:40]), 0.2)
        self.assertGreater(min(peaks[60:]), 0.8)

    def test_missing_file_and_undecodable_file(self) -> None:
        with self.assertRaises(waveform.WaveformError):
            waveform.compute_peaks(str(Path(self._tmp.name) / "nope.wav"))
        junk = Path(self._tmp.name) / "junk.mp3"
        junk.write_bytes(b"this is not audio" * 10)
        with self.assertRaises(waveform.WaveformError):
            waveform.compute_peaks(str(junk))

    def test_missing_ffmpeg_is_reported(self) -> None:
        junk = Path(self._tmp.name) / "junk.mp3"
        junk.write_bytes(b"x" * 100)
        with patch("services.waveform_service.shutil.which", return_value=None):
            with self.assertRaisesRegex(waveform.WaveformError, "ffmpeg"):
                waveform.compute_peaks(str(junk))

    def test_silence_stays_flat_and_preview_downsamples(self) -> None:
        quiet = Path(self._tmp.name) / "quiet.wav"
        write_wav(quiet, seconds=1.0, loud_from=99)
        self.assertGreater(max(waveform.compute_peaks(str(quiet), buckets=10)), 0.0)
        self.assertEqual(waveform._bucketize(__import__("numpy").zeros(50), 10), [0.0] * 10)
        preview = waveform.preview_from([0.1, 0.9] * 500, 64)
        self.assertEqual(len(preview), 64)
        self.assertEqual(waveform.preview_from([], 64), [])

    def test_store_caches_peaks_and_preview(self) -> None:
        store = ProjectStore(Path(self._tmp.name) / "projects")
        created = store.create(title="x")
        peaks = waveform.compute_peaks(str(self.wav), buckets=500)
        self.assertTrue(store.set_peaks(created.id, peaks, waveform.preview_from(peaks)))
        self.assertEqual(store.load_peaks(created.id), peaks)
        self.assertEqual(len(store.list_summaries()[0].preview_peaks), 64)
        self.assertFalse(store.set_peaks("20260101-000000-abcdef12", peaks, []))
        self.assertEqual(store.load_peaks("20260101-000000-abcdef12"), [])


if __name__ == "__main__":
    unittest.main()
