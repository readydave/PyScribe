"""Tests for diarization runtime safeguards."""

from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

import diarization


class DiarizationRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._patched_flag = diarization._TORCHAUDIO_SOUNDFILE_PATCHED
        self._had_info = hasattr(diarization.torchaudio, "info")
        self._original_info = getattr(diarization.torchaudio, "info", None)
        self._original_load = getattr(diarization.torchaudio, "load", None)
        self._had_load_with_torchcodec = hasattr(diarization.torchaudio, "load_with_torchcodec")
        self._original_load_with_torchcodec = getattr(diarization.torchaudio, "load_with_torchcodec", None)

    def tearDown(self) -> None:
        diarization._TORCHAUDIO_SOUNDFILE_PATCHED = self._patched_flag
        if self._had_info:
            diarization.torchaudio.info = self._original_info
        elif hasattr(diarization.torchaudio, "info"):
            delattr(diarization.torchaudio, "info")
        if self._original_load:
            diarization.torchaudio.load = self._original_load
        if self._had_load_with_torchcodec:
            diarization.torchaudio.load_with_torchcodec = self._original_load_with_torchcodec
        elif hasattr(diarization.torchaudio, "load_with_torchcodec"):
            delattr(diarization.torchaudio, "load_with_torchcodec")

    def test_prefer_torchaudio_soundfile_backend_wraps_default_io(self) -> None:
        info_calls: list[dict] = []
        load_calls: list[dict] = []

        def fake_info(*args, **kwargs):
            info_calls.append(dict(kwargs))
            return "info"

        def fake_load(*args, **kwargs):
            load_calls.append(dict(kwargs))
            return "load"

        # Helper to conditionally patch list_audio_backends
        def maybe_patch_list_backends(return_val):
            if hasattr(diarization.torchaudio, "list_audio_backends"):
                return patch.object(diarization.torchaudio, "list_audio_backends", return_value=return_val)
            # If not present, we don't need to patch it for the success case
            # as our code handles its absence.
            return patch("builtins.dir", return_value=[])

        patches = [
            maybe_patch_list_backends(["sox", "soundfile"]),
            patch.object(diarization.torchaudio, "info", fake_info, create=True),
            patch.object(diarization.torchaudio, "load", fake_load),
        ]

        from contextlib import ExitStack
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
                
            diarization._TORCHAUDIO_SOUNDFILE_PATCHED = False

            backend = diarization._prefer_torchaudio_soundfile_backend()
            self.assertEqual(backend, "soundfile")

            # The wrappers should be installed now
            self.assertTrue(getattr(diarization.torchaudio.info, "__pyscribe_soundfile_default__", False))
            self.assertEqual(diarization.torchaudio.info("clip.wav"), "info")
            self.assertEqual(info_calls[-1]["backend"], "soundfile")

            self.assertTrue(getattr(diarization.torchaudio.load, "__pyscribe_soundfile_default__", False))
            self.assertEqual(diarization.torchaudio.load("clip.wav"), "load")
            self.assertEqual(load_calls[-1]["backend"], "soundfile")

            diarization.torchaudio.info("clip.wav", backend="sox")
            self.assertEqual(info_calls[-1]["backend"], "sox")

            diarization.torchaudio.load("clip.wav", backend="sox")
            self.assertEqual(load_calls[-1]["backend"], "sox")

    def test_prefer_torchaudio_soundfile_backend_adds_missing_info_shim(self) -> None:
        if hasattr(diarization.torchaudio, "info"):
            delattr(diarization.torchaudio, "info")

        with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
            sf.write(handle.name, np.zeros(160, dtype=np.float32), 16000)
            diarization._TORCHAUDIO_SOUNDFILE_PATCHED = False

            backend = diarization._prefer_torchaudio_soundfile_backend()
            metadata = diarization.torchaudio.info(handle.name)

        self.assertEqual(backend, "soundfile")
        self.assertEqual(metadata.sample_rate, 16000)
        self.assertEqual(metadata.num_channels, 1)
        self.assertEqual(metadata.num_frames, 160)

    def test_load_with_torchcodec_shim_uses_direct_soundfile_loader(self) -> None:
        def failing_torchcodec(*args, **kwargs):
            raise RuntimeError("torchcodec should not be used")

        diarization.torchaudio.load_with_torchcodec = failing_torchcodec
        diarization._TORCHAUDIO_SOUNDFILE_PATCHED = False

        with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
            sf.write(handle.name, np.zeros((160, 2), dtype=np.float32), 16000)
            diarization._prefer_torchaudio_soundfile_backend()

            waveform, sample_rate = diarization.torchaudio.load_with_torchcodec(handle.name)

        self.assertEqual(sample_rate, 16000)
        self.assertEqual(tuple(waveform.shape), (2, 160))

    def test_prefer_torchaudio_soundfile_backend_skips_when_unavailable(self) -> None:
        if not hasattr(diarization.torchaudio, "list_audio_backends"):
            self.skipTest("torchaudio 2.9+ dispatcher logic does not support 'unavailable' soundfile check via list_audio_backends")

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

            def __call__(self, audio, num_speakers: int | None = None, hook=None):
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
        ), patch(
            "diarization._load_audio_for_pyannote",
            return_value={"waveform": None, "sample_rate": 16000},
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

    def test_run_diarization_propagates_inference_failure(self) -> None:
        class _FakePipelineInstance:
            def to(self, device) -> None:
                pass

            def __call__(self, audio, num_speakers: int | None = None, hook=None):
                raise RuntimeError("torchaudio.info missing")

        class _FakePipelineFactory:
            @staticmethod
            def from_pretrained(model_name: str, use_auth_token=None):
                return _FakePipelineInstance()

        with patch("diarization.ensure_platform_sys_version_compat"), patch(
            "diarization._prefer_torchaudio_soundfile_backend",
            return_value="soundfile",
        ), patch(
            "diarization._lazy_import_pyannote",
            return_value=_FakePipelineFactory,
        ), patch(
            "diarization.get_hf_token",
            return_value=None,
        ), patch(
            "diarization._load_audio_for_pyannote",
            return_value={"waveform": None, "sample_rate": 16000},
        ):
            with self.assertRaisesRegex(RuntimeError, "torchaudio.info missing"):
                diarization.run_diarization("clip.wav", device="cpu")


class PipelineLoadingTests(unittest.TestCase):
    def test_community_first_on_pyannote_4_then_legacy(self) -> None:
        with patch("diarization._pyannote_major_version", return_value=4):
            labels = [label for label, _ in diarization._pipeline_candidates()]
        self.assertEqual(labels, ["community-1", "3.1", "3.0"])
        with patch("diarization._pyannote_major_version", return_value=3):
            labels = [label for label, _ in diarization._pipeline_candidates()]
        self.assertEqual(labels, ["3.1", "3.0"])

    def test_falls_back_to_next_candidate(self) -> None:
        seen: list[str] = []

        class _Factory:
            @staticmethod
            def from_pretrained(name: str, token=None):
                seen.append(name)
                if name.endswith("community-1"):
                    raise RuntimeError("403 gated")
                return object()

        with patch("diarization._pyannote_major_version", return_value=4):
            _, label = diarization._load_pyannote_pipeline(_Factory, "tok", "cpu")
        self.assertEqual(label, "3.1")
        self.assertEqual(len(seen), 2)

    def test_all_candidates_failing_explains_how_to_get_access(self) -> None:
        class _Factory:
            @staticmethod
            def from_pretrained(name: str, token=None):
                raise RuntimeError("403 gated")

        with patch("diarization._pyannote_major_version", return_value=4):
            with self.assertRaisesRegex(RuntimeError, "Accept the model terms"):
                diarization._load_pyannote_pipeline(_Factory, "tok", "cpu")

    def test_legacy_use_auth_token_keyword_still_supported(self) -> None:
        class _Factory:
            @staticmethod
            def from_pretrained(name: str, use_auth_token=None):
                return ("loaded", use_auth_token)

        self.assertEqual(diarization._from_pretrained(_Factory, "x", "tok"), ("loaded", "tok"))


class OutputAndProgressTests(unittest.TestCase):
    def test_speaker_annotation_prefers_exclusive(self) -> None:
        class _Out:
            speaker_diarization = "full"
            exclusive_speaker_diarization = "exclusive"

        self.assertEqual(diarization._speaker_annotation(_Out()), "exclusive")
        self.assertEqual(diarization._speaker_annotation("annotation"), "annotation")

    def test_progress_hook_maps_steps_to_overall_range(self) -> None:
        values: list[float] = []
        hook = diarization._ProgressHook(values.append)
        hook("segmentation", None, total=10, completed=5)
        hook("embeddings", None, total=4, completed=4)
        hook("speaker_counting", None)
        hook("embeddings", None, total=None, completed=None)
        self.assertEqual(values, [20.0, 95.0])


if __name__ == "__main__":
    unittest.main()
