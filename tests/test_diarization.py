"""Tests for diarization runtime safeguards."""

from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

import diarization


class DiarizationRuntimeTests(unittest.TestCase):
    def test_load_audio_for_pyannote_returns_waveform_dict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/clip.wav"
            sf.write(path, np.zeros(1600, dtype="float32"), 16000)
            audio = diarization._load_audio_for_pyannote(path)
        self.assertEqual(audio["sample_rate"], 16000)
        self.assertEqual(tuple(audio["waveform"].shape), (1, 1600))

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


class PyannoteImportNoiseTests(unittest.TestCase):
    def _import_with_warning(self, message: str) -> list:
        import builtins
        import types
        import warnings as _warnings

        fake_audio = types.SimpleNamespace(Pipeline=type("Pipeline", (), {}))
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pyannote.audio":
                _warnings.warn(message, UserWarning)
                return fake_audio
            return real_import(name, *args, **kwargs)

        with _warnings.catch_warnings(record=True) as seen, patch.object(builtins, "__import__", fake_import):
            _warnings.simplefilter("always")
            result = diarization._lazy_import_pyannote()
        self.assertIs(result, fake_audio.Pipeline)
        return seen

    def test_torchcodec_warning_is_swallowed(self) -> None:
        seen = self._import_with_warning("torchcodec is not installed correctly so built-in audio decoding will fail.")
        self.assertEqual([w for w in seen if "torchcodec" in str(w.message)], [])

    def test_other_warnings_still_surface(self) -> None:
        seen = self._import_with_warning("something unrelated deserves attention")
        self.assertEqual([str(w.message) for w in seen], ["something unrelated deserves attention"])


if __name__ == "__main__":
    unittest.main()
