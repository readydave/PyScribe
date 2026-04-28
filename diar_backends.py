# diar_backends.py
# Abstraction layer for selectable diarization backends.

from __future__ import annotations

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
    "off": {
        "label": "Off (no speakers)",
        "runner": None,
        "requires": None,
        "desc": "Skip diarization."
    },
}


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
