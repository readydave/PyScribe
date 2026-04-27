# diar_backends.py
# Abstraction layer for selectable diarization backends.

from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable

from diarization import run_diarization as run_pyannote

ProgressCB = Optional[Callable[[float], None]]
StatusCB = Optional[Callable[[str], None]]


@dataclass(frozen=True)
class BackendAvailability:
    available: bool
    reason: str | None = None


def _bump(cb: ProgressCB, value: float) -> None:
    if cb:
        cb(min(max(value, 0), 100))


def run_pyannote_fast(
    audio_path: str,
    device: str,
    max_speakers: Optional[int],
    progress_cb: ProgressCB = None,
    status_cb: StatusCB = None,
) -> List[Dict]:
    """
    A slightly faster variant of the pyannote pipeline.
    Today this reuses the same pipeline but keeps a hook to tweak settings later.
    """
    _bump(progress_cb, 35)
    return run_pyannote(audio_path, device=device, max_speakers=max_speakers, status_cb=status_cb)


def run_nemo_sortformer(
    audio_path: str,
    device: str,
    max_speakers: Optional[int],
    progress_cb: ProgressCB = None,
    status_cb: StatusCB = None,
) -> List[Dict]:
    """
    Optional NVIDIA NeMo Sortformer diarization backend.
    Requires nemo_toolkit[asr] >= 1.23 and a CUDA GPU.
    """
    try:
        import nemo.collections.asr as nemo_asr  # type: ignore
        import torch
    except ImportError as e:
        raise RuntimeError("NeMo diarization not installed. Install with: pip install nemo_toolkit[asr]") from e

    if device != "cuda":
        raise RuntimeError("NeMo Sortformer requires CUDA GPU.")
    if status_cb:
        status_cb("Diarization backend: sortformer | Device: CUDA")

    _bump(progress_cb, 20)
    # API guard: NeMo releases use different class names
    if hasattr(nemo_asr.models.msdd_models, "SpeakerDiarizer"):
        diarizer = nemo_asr.models.msdd_models.SpeakerDiarizer.from_pretrained(model_name="diar_msdd_telephonic")
    elif hasattr(nemo_asr.models.msdd_models, "DiarizationModel"):
        diarizer = nemo_asr.models.msdd_models.DiarizationModel.from_pretrained(model_name="diar_msdd_telephonic")
    elif hasattr(nemo_asr.models.msdd_models, "NeuralDiarizer"):
        diarizer = nemo_asr.models.msdd_models.NeuralDiarizer.from_pretrained(model_name="diar_msdd_telephonic")
    else:
        raise RuntimeError(
            "NeMo Sortformer backend unavailable: your nemo_toolkit version does not provide a known diarizer class. "
            "Try installing a compatible version, e.g.: pip install 'nemo_toolkit[asr]==1.23.0'"
        )
    try:
        diarizer.to(torch.device(device))
    except Exception:
        diarizer.to(device)

    # NeMo 2.x `NeuralDiarizer` exposes a callable inference API.
    call_supports_audio = False
    try:
        call_sig = inspect.signature(type(diarizer).__call__)
        call_supports_audio = "audio_filepath" in call_sig.parameters
    except Exception:
        call_supports_audio = False

    if call_supports_audio:
        # NeMo 2.x can fail with DataLoader worker spawning on Windows unless workers are disabled.
        annotation = diarizer(
            audio_path,
            max_speakers=max_speakers,
            num_workers=0,
            verbose=False,
        )
        _bump(progress_cb, 90)
        segments: List[Dict] = []
        speaker_map: Dict[str, str] = {}
        speaker_idx = 1
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            spk_key = str(speaker)
            if spk_key not in speaker_map:
                speaker_map[spk_key] = f"S{speaker_idx}"
                speaker_idx += 1
            start = float(turn.start)
            end = float(turn.end)
            if end <= start:
                continue
            segments.append({"start": start, "end": end, "speaker": speaker_map[spk_key]})
        _bump(progress_cb, 100)
        return segments

    # Legacy NeMo API path: diarizer writes RTTM directly.
    tmp_rttm = audio_path + ".nemo.rttm"
    diarizer.diarize(paths2audio_files=[audio_path], out_rttm_file=tmp_rttm, num_speakers=max_speakers)
    _bump(progress_cb, 90)

    segments: List[Dict] = []
    try:
        with open(tmp_rttm, "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) < 8:
                    continue
                start = float(parts[3])
                dur = float(parts[4])
                spk = parts[7]
                segments.append({"start": start, "end": start + dur, "speaker": spk})
    finally:
        try:
            import os
            os.remove(tmp_rttm)
        except Exception:
            pass
    _bump(progress_cb, 100)
    return segments


BACKENDS = {
    "accurate": {
        "label": "Accurate (pyannote 3.1)",
        "runner": run_pyannote,
        "requires": "pyannote.audio",
        "desc": "Highest accuracy; slower on long files."
    },
    "fast": {
        "label": "Fast (approx)",
        "runner": run_pyannote_fast,
        "requires": "pyannote.audio",
        "desc": "Slightly faster settings; good trade-off."
    },
    "sortformer": {
        "label": "GPU Sortformer (NeMo)",
        "runner": run_nemo_sortformer,
        "requires": "nemo_toolkit[asr] + CUDA",
        "desc": "Fast on NVIDIA GPUs; extra install."
    },
    "off": {
        "label": "Off (no speakers)",
        "runner": None,
        "requires": None,
        "desc": "Skip diarization."
    },
}


def _sortformer_availability() -> BackendAvailability:
    """
    Return Sortformer dependency availability and an actionable unavailable reason.

    Keep this check lightweight so UI startup/probing does not trigger heavyweight
    imports. Runtime execution still validates actual CUDA usability.
    """
    import importlib.util

    try:
        if importlib.util.find_spec("nemo.collections.asr") is None:
            return BackendAvailability(
                False,
                "NeMo ASR is not installed. Install with: pip install nemo_toolkit[asr]",
            )
    except ModuleNotFoundError:
        return BackendAvailability(
            False,
            "NeMo ASR is not installed. Install with: pip install nemo_toolkit[asr]",
        )
    except Exception as exc:
        return BackendAvailability(False, f"Unable to inspect NeMo ASR availability: {exc}")

    try:
        if importlib.util.find_spec("torch") is None:
            return BackendAvailability(
                False,
                "PyTorch is not installed. Install a CUDA-enabled PyTorch build for Sortformer.",
            )
    except Exception as exc:
        return BackendAvailability(False, f"Unable to inspect PyTorch availability: {exc}")

    # If torch is already loaded, use its CUDA signal without importing torch here.
    torch_module = sys.modules.get("torch")
    if torch_module is not None:
        try:
            cuda_attr = getattr(torch_module, "cuda", None)
            is_available = getattr(cuda_attr, "is_available", None)
            if callable(is_available):
                if bool(is_available()):
                    return BackendAvailability(True)
                return BackendAvailability(
                    False,
                    "CUDA is unavailable to PyTorch. Install/repair the NVIDIA driver and a CUDA-enabled PyTorch build.",
                )
        except Exception as exc:
            return BackendAvailability(False, f"Unable to verify PyTorch CUDA availability: {exc}")

    # Avoid importing torch during capability probe.
    return BackendAvailability(True)


def _is_sortformer_available() -> bool:
    return _sortformer_availability().available


def _module_exists(module_name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def backend_availability() -> Dict[str, BackendAvailability]:
    """Return a map of backend_id -> availability with unavailable reason."""
    availability: Dict[str, BackendAvailability] = {}
    for key, meta in BACKENDS.items():
        if key == "sortformer":
            availability[key] = _sortformer_availability()
            continue
        req = meta["requires"]
        if req is None:
            availability[key] = BackendAvailability(True)
        else:
            module_name = req.split()[0].split("[")[0].split("==")[0]
            available = _module_exists(module_name)
            reason = None if available else f"Required module '{module_name}' is not installed."
            availability[key] = BackendAvailability(available, reason)
    return availability


def available_backends() -> Dict[str, bool]:
    """Return a map of backend_id -> available (installed)."""
    return {key: status.available for key, status in backend_availability().items()}


def run_diarization_backend(
    audio_path: str,
    backend: str,
    device: str = "cpu",
    max_speakers: Optional[int] = None,
    progress_cb: ProgressCB = None,
    status_cb: StatusCB = None,
) -> List[Dict]:
    """
    Dispatch diarization to the selected backend.
    """
    if backend == "off":
        _bump(progress_cb, 0)
        return []
    meta = BACKENDS.get(backend)
    if not meta:
        raise RuntimeError(f"Unknown diarization backend: {backend}")
    runner = meta["runner"]
    if runner is None:
        return []
    return runner(
        audio_path=audio_path,
        device=device,
        max_speakers=max_speakers,
        progress_cb=progress_cb,
        status_cb=status_cb,
    )
