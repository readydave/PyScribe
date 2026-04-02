"""Tests for diarization runtime safeguards."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import diarization


class DiarizationRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._patched_flag = diarization._TORCHAUDIO_SOUNDFILE_PATCHED

    def tearDown(self) -> None:
        diarization._TORCHAUDIO_SOUNDFILE_PATCHED = self._patched_flag

    def test_prefer_torchaudio_soundfile_backend_wraps_default_io(self) -> None:
        info_calls: list[dict] = []
        load_calls: list[dict] = []

        def fake_info(*args, **kwargs):
            info_calls.append(dict(kwargs))
            return "info"

        def fake_load(*args, **kwargs):
            load_calls.append(dict(kwargs))
            return "load"

        with patch.object(diarization.torchaudio, "list_audio_backends", return_value=["sox", "soundfile"]), patch.object(
            diarization.torchaudio,
            "info",
            fake_info,
        ), patch.object(
            diarization.torchaudio,
            "load",
            fake_load,
        ):
            diarization._TORCHAUDIO_SOUNDFILE_PATCHED = False

            backend = diarization._prefer_torchaudio_soundfile_backend()
            self.assertEqual(backend, "soundfile")

            self.assertEqual(diarization.torchaudio.info("clip.wav"), "info")
            self.assertEqual(diarization.torchaudio.load("clip.wav"), "load")
            self.assertEqual(info_calls[-1]["backend"], "soundfile")
            self.assertEqual(load_calls[-1]["backend"], "soundfile")

            diarization.torchaudio.info("clip.wav", backend="sox")
            diarization.torchaudio.load("clip.wav", backend="sox")
            self.assertEqual(info_calls[-1]["backend"], "sox")
            self.assertEqual(load_calls[-1]["backend"], "sox")

    def test_prefer_torchaudio_soundfile_backend_skips_when_unavailable(self) -> None:
        with patch.object(diarization.torchaudio, "list_audio_backends", return_value=["sox"]):
            diarization._TORCHAUDIO_SOUNDFILE_PATCHED = False
            self.assertIsNone(diarization._prefer_torchaudio_soundfile_backend())

    def test_run_diarization_reloads_pipeline_on_cpu_after_cuda_move_failure(self) -> None:
        status_updates: list[str] = []
        created: list[object] = []

        class _FakeTurn:
            def __init__(self, start: float, end: float) -> None:
                self.start = start
                self.end = end

        class _FakeAnnotation:
            def itertracks(self, yield_label: bool = False):
                return [(_FakeTurn(0.0, 1.0), None, "speaker-a")]

        class _FakePipelineInstance:
            def __init__(self, generation: int) -> None:
                self.generation = generation
                self.to_calls: list[str] = []

            def to(self, device) -> None:
                device_name = str(device)
                self.to_calls.append(device_name)
                if self.generation == 0 and device_name == "cuda":
                    raise RuntimeError("cuDNN mismatch")

            def __call__(self, audio_path: str, num_speakers: int | None = None):
                return _FakeAnnotation()

        class _FakePipelineFactory:
            @staticmethod
            def from_pretrained(model_name: str, use_auth_token=None):
                instance = _FakePipelineInstance(len(created))
                created.append(instance)
                return instance

        with patch("diarization.ensure_platform_sys_version_compat"), patch(
            "diarization._prefer_torchaudio_soundfile_backend",
            return_value="soundfile",
        ), patch(
            "diarization._lazy_import_pyannote",
            return_value=_FakePipelineFactory,
        ), patch(
            "diarization.get_hf_token",
            return_value=None,
        ):
            segments = diarization.run_diarization(
                "clip.wav",
                device="cuda",
                max_speakers=2,
                status_cb=status_updates.append,
            )

        self.assertEqual(
            segments,
            [{"start": 0.0, "end": 1.0, "speaker": "S1"}],
        )
        self.assertEqual(len(created), 2)
        self.assertEqual(created[0].to_calls, ["cuda"])
        self.assertEqual(created[1].to_calls, [])
        self.assertIn("Diarization backend fallback to CPU (requested CUDA)", status_updates)


if __name__ == "__main__":
    unittest.main()
