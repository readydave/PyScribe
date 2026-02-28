"""Unit tests for listener security helpers and diarization backend compatibility."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from services import listener_security_service


def _import_diar_backends_with_stub() -> types.ModuleType:
    stub = types.ModuleType("diarization")
    stub.run_diarization = lambda *args, **kwargs: []
    with patch.dict(sys.modules, {"diarization": stub}):
        sys.modules.pop("diar_backends", None)
        return importlib.import_module("diar_backends")


class ListenerSecurityServiceTests(unittest.TestCase):
    def test_reject_legacy_auth_pass_flag(self) -> None:
        with self.assertRaises(SystemExit):
            listener_security_service.reject_legacy_auth_pass_flag(["main.py", "--auth-pass=secret"])

    def test_resolve_listener_auth_from_env(self) -> None:
        with patch.dict(
            os.environ,
            {"PYSCRIBE_AUTH_USER": "alice", "PYSCRIBE_AUTH_PASS": "secret"},
            clear=False,
        ):
            user, password = listener_security_service.resolve_listener_auth(None)
        self.assertEqual(user, "alice")
        self.assertEqual(password, "secret")

    def test_validate_listener_security_requires_auth_for_nonlocal(self) -> None:
        with self.assertRaises(SystemExit):
            listener_security_service.validate_listener_security(
                "0.0.0.0",
                auth_user=None,
                auth_pass=None,
                allow_nonlocal_host=True,
            )

    def test_validate_listener_security_requires_auth_for_share(self) -> None:
        with self.assertRaises(SystemExit):
            listener_security_service.validate_listener_security(
                "127.0.0.1",
                auth_user=None,
                auth_pass=None,
                allow_nonlocal_host=False,
                share=True,
            )


class DiarBackendsCompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.diar_backends = _import_diar_backends_with_stub()

    def tearDown(self) -> None:
        sys.modules.pop("diar_backends", None)

    def test_is_sortformer_available_checks_nemo_and_cuda(self) -> None:
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
        real_import_module = importlib.import_module

        def _fake_import_module(name: str, *args: object, **kwargs: object) -> object:
            if name == "nemo.collections.asr":
                return object()
            return real_import_module(name, *args, **kwargs)

        with patch("importlib.import_module", side_effect=_fake_import_module):
            with patch.dict(sys.modules, {"torch": fake_torch}):
                self.assertTrue(self.diar_backends._is_sortformer_available())

    def test_run_nemo_sortformer_modern_neural_diarizer_path(self) -> None:
        class _Turn:
            def __init__(self, start: float, end: float) -> None:
                self.start = start
                self.end = end

        class _Annotation:
            def itertracks(self, yield_label: bool = True):  # noqa: ANN001
                del yield_label
                yield _Turn(0.0, 1.2), None, "speaker_a"
                yield _Turn(1.3, 2.1), None, "speaker_b"

        class _NeuralDiarizer:
            last_instance = None

            @classmethod
            def from_pretrained(cls, model_name: str):  # noqa: ANN206
                del model_name
                cls.last_instance = cls()
                return cls.last_instance

            def to(self, device: object) -> None:
                self.device = device

            def __call__(
                self,
                audio_filepath: str,
                batch_size: int = 64,
                num_workers: int = 1,
                max_speakers: int | None = None,
                num_speakers: int | None = None,
                out_dir: str | None = None,
                verbose: bool = False,
            ) -> _Annotation:
                self.call_args = {
                    "audio_filepath": audio_filepath,
                    "batch_size": batch_size,
                    "num_workers": num_workers,
                    "max_speakers": max_speakers,
                    "num_speakers": num_speakers,
                    "out_dir": out_dir,
                    "verbose": verbose,
                }
                return _Annotation()

        fake_msdd = types.SimpleNamespace(NeuralDiarizer=_NeuralDiarizer)
        fake_asr_models = types.SimpleNamespace(msdd_models=fake_msdd)
        fake_asr_module = types.ModuleType("nemo.collections.asr")
        fake_asr_module.models = fake_asr_models
        fake_collections_module = types.ModuleType("nemo.collections")
        fake_collections_module.asr = fake_asr_module
        fake_nemo_module = types.ModuleType("nemo")
        fake_nemo_module.collections = fake_collections_module

        fake_torch = types.ModuleType("torch")
        fake_torch.device = lambda value: value

        with patch.dict(
            sys.modules,
            {
                "nemo": fake_nemo_module,
                "nemo.collections": fake_collections_module,
                "nemo.collections.asr": fake_asr_module,
                "torch": fake_torch,
            },
        ):
            segments = self.diar_backends.run_nemo_sortformer(
                audio_path="sample.wav",
                device="cuda",
                max_speakers=2,
            )

        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0]["speaker"], "S1")
        self.assertEqual(segments[1]["speaker"], "S2")
        self.assertEqual(_NeuralDiarizer.last_instance.call_args["num_workers"], 0)

    def test_run_nemo_sortformer_legacy_rttm_path(self) -> None:
        class _SpeakerDiarizer:
            @classmethod
            def from_pretrained(cls, model_name: str):  # noqa: ANN206
                del model_name
                return cls()

            def to(self, device: object) -> None:
                self.device = device

            def diarize(
                self,
                *,
                paths2audio_files: list[str],
                out_rttm_file: str,
                num_speakers: int | None,
            ) -> None:
                del paths2audio_files, num_speakers
                Path(out_rttm_file).write_text(
                    "SPEAKER file 1 0.00 1.25 <NA> <NA> spk1 <NA> <NA>\n",
                    encoding="utf-8",
                )

        fake_msdd = types.SimpleNamespace(SpeakerDiarizer=_SpeakerDiarizer)
        fake_asr_models = types.SimpleNamespace(msdd_models=fake_msdd)
        fake_asr_module = types.ModuleType("nemo.collections.asr")
        fake_asr_module.models = fake_asr_models
        fake_collections_module = types.ModuleType("nemo.collections")
        fake_collections_module.asr = fake_asr_module
        fake_nemo_module = types.ModuleType("nemo")
        fake_nemo_module.collections = fake_collections_module

        fake_torch = types.ModuleType("torch")
        fake_torch.device = lambda value: value

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = str(Path(temp_dir) / "audio.wav")
            Path(audio_path).write_text("", encoding="utf-8")
            with patch.dict(
                sys.modules,
                {
                    "nemo": fake_nemo_module,
                    "nemo.collections": fake_collections_module,
                    "nemo.collections.asr": fake_asr_module,
                    "torch": fake_torch,
                },
            ):
                segments = self.diar_backends.run_nemo_sortformer(
                    audio_path=audio_path,
                    device="cuda",
                    max_speakers=2,
                )

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["speaker"], "spk1")
        self.assertEqual(segments[0]["start"], 0.0)
        self.assertEqual(segments[0]["end"], 1.25)


if __name__ == "__main__":
    unittest.main()
