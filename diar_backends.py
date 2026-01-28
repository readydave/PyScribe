# diar_backends.py
# Abstraction layer for selectable diarization backends.

from __future__ import annotations

from typing import List, Dict, Optional, Callable

from diarization import run_diarization as run_pyannote

ProgressCB = Optional[Callable[[float], None]]


def _bump(cb: ProgressCB, value: float):
    if cb:
        cb(min(max(value, 0), 100))


def run_pyannote_fast(audio_path: str, device: str, max_speakers: Optional[int], progress_cb: ProgressCB = None) -> List[Dict]:
    """
    A slightly faster variant of the pyannote pipeline.
    Today this reuses the same pipeline but keeps a hook to tweak settings later.
    """
    _bump(progress_cb, 35)
    return run_pyannote(audio_path, device=device, max_speakers=max_speakers)


def run_nemo_sortformer(audio_path: str, device: str, max_speakers: Optional[int], progress_cb: ProgressCB = None) -> List[Dict]:
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
    diarizer.to(device)
    diarizer.segmentation.parameters().device = torch.device(device)
    diarizer.clustering.parameters().device = torch.device(device)

    # Running diarization; NeMo writes RTTM, so we collect segments from output manifest.
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


def available_backends() -> Dict[str, bool]:
    """Return a map of backend_id -> available (installed)."""
    import importlib
    availability = {}
    for key, meta in BACKENDS.items():
        req = meta["requires"]
        if req is None:
            availability[key] = True
        else:
            # crude check: import the top-level package name
            top = req.split()[0].split("[")[0].split("==")[0]
            try:
                importlib.import_module(top.split(".")[0])
                availability[key] = True
            except ImportError:
                availability[key] = False
    return availability


def run_diarization_backend(
    audio_path: str,
    backend: str,
    device: str = "cpu",
    max_speakers: Optional[int] = None,
    progress_cb: ProgressCB = None,
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
    return runner(audio_path=audio_path, device=device, max_speakers=max_speakers, progress_cb=progress_cb)
