from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from services.config_service import AppConfig, load_config, save_config
from services.live_transcription_service import resolve_live_runtime
from services.model_service import RuntimeInfo

GPU = RuntimeInfo(device="cuda", compute_type="float16", gpu_name="GPU", vram_gb=12.0, cpu_count=8)
CPU = RuntimeInfo(device="cpu", compute_type="int8", gpu_name="N/A", vram_gb=0.0, cpu_count=8)
WHISPER = "large-v3"
NEMOTRON = "nvidia/nemotron-speech-streaming-en-0.6b"


def _pair(*args: object) -> tuple[str, str]:
    choice = resolve_live_runtime(*args)  # type: ignore[arg-type]
    return choice.device, choice.compute_type


class ResolveLiveRuntimeTests(unittest.TestCase):
    def test_auto_follows_detected_runtime(self) -> None:
        self.assertEqual(_pair("auto", "auto", GPU, WHISPER), ("cuda", "float16"))
        self.assertEqual(_pair("auto", "auto", CPU, WHISPER), ("cpu", "int8"))

    def test_cpu_forced_on_gpu_machine(self) -> None:
        self.assertEqual(_pair("cpu", "auto", GPU, WHISPER), ("cpu", "int8"))
        choice = resolve_live_runtime("cpu", "float16", GPU, WHISPER)
        self.assertEqual((choice.device, choice.compute_type), ("cpu", "int8"))
        self.assertIn("float16", choice.note)

    def test_gpu_without_cuda_falls_back_with_note(self) -> None:
        choice = resolve_live_runtime("gpu", "auto", CPU, WHISPER)
        self.assertEqual((choice.device, choice.compute_type), ("cpu", "int8"))
        self.assertIn("CUDA", choice.note)

    def test_gpu_precision_choices(self) -> None:
        self.assertEqual(_pair("gpu", "int8", GPU, WHISPER), ("cuda", "int8"))
        self.assertEqual(_pair("gpu", "float16", GPU, WHISPER), ("cuda", "float16"))

    def test_nemotron_ignores_precision(self) -> None:
        self.assertEqual(_pair("gpu", "int8", GPU, NEMOTRON), ("cuda", "float16"))
        self.assertEqual(_pair("cpu", "int8", GPU, NEMOTRON), ("cpu", "float32"))
        self.assertEqual(_pair("auto", "float16", CPU, NEMOTRON), ("cpu", "float32"))


class LiveRuntimeConfigTests(unittest.TestCase):
    def test_legacy_config_defaults_to_auto(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text("{}", encoding="utf-8")
            cfg = load_config(path)
        self.assertEqual((cfg.live_device_mode, cfg.live_compute_type), ("auto", "auto"))

    def test_round_trip_and_invalid_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            save_config(AppConfig(live_device_mode="cpu", live_compute_type="int8"), path)
            cfg = load_config(path)
            self.assertEqual((cfg.live_device_mode, cfg.live_compute_type), ("cpu", "int8"))
            path.write_text(json.dumps({"live_device_mode": "tpu", "live_compute_type": 7}), encoding="utf-8")
            cfg = load_config(path)
        self.assertEqual((cfg.live_device_mode, cfg.live_compute_type), ("auto", "auto"))


if __name__ == "__main__":
    unittest.main()
