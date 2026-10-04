"""GPU memory preflight helpers for Qt live transcription."""

from __future__ import annotations

from dataclasses import dataclass

from .model_download_service import normalize_model_name, resolve_repo_id


LIVE_VRAM_SAFETY_BUFFER_GB = 1.0

_LIVE_MODEL_VRAM_GB: dict[str, float] = {
    "tiny": 0.5,
    "base": 0.8,
    "small": 1.5,
    "small.en": 1.5,
    "medium": 3.0,
    "large-v2": 6.0,
    "large-v3": 6.5,
    "distil-whisper/distil-large-v3": 3.0,
    "deepdml/faster-whisper-large-v3-turbo-ct2": 4.0,
    "guillaumekln/whisper-large-v2-ct2": 6.0,
    "guillaumekln/whisper-large-v3-ct2": 6.5,
}


@dataclass(frozen=True)
class GpuMemoryInfo:
    total_gb: float
    free_gb: float
    used_gb: float
    source: str


@dataclass(frozen=True)
class LiveVramPreflight:
    status: str
    model_name: str
    estimated_required_gb: float
    model_estimate_gb: float
    safety_buffer_gb: float
    free_gb: float | None
    total_gb: float | None
    used_gb: float | None
    message: str

    @property
    def should_warn(self) -> bool:
        return self.status == "low"


def get_gpu_memory_info(device_index: int = 0) -> GpuMemoryInfo | None:
    """Returns current GPU memory from NVML when available."""
    pynvml = None
    initialized = False
    try:
        import pynvml as _pynvml  # type: ignore

        pynvml = _pynvml
        pynvml.nvmlInit()
        initialized = True
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total = float(mem.total) / (1024**3)
        used = float(mem.used) / (1024**3)
        free = float(mem.free) / (1024**3)
        return GpuMemoryInfo(
            total_gb=round(total, 2),
            free_gb=round(free, 2),
            used_gb=round(used, 2),
            source="nvml",
        )
    except Exception:
        return None
    finally:
        if initialized and pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def estimate_live_model_vram_gb(model_name: str, *, compute_type: str = "float16") -> float:
    """Returns a conservative live-ASR VRAM estimate for a selected model."""
    normalized = normalize_model_name(model_name)
    repo_id = resolve_repo_id(normalized)
    keys = [normalized, repo_id or ""]
    for key in keys:
        if key in _LIVE_MODEL_VRAM_GB:
            return _adjust_for_compute_type(_LIVE_MODEL_VRAM_GB[key], compute_type)

    lower = (repo_id or normalized).lower()
    if "tiny" in lower:
        estimate = 0.5
    elif "base" in lower:
        estimate = 0.8
    elif "small" in lower:
        estimate = 1.5
    elif "medium" in lower:
        estimate = 3.0
    elif "distil" in lower or "turbo" in lower:
        estimate = 4.0
    else:
        estimate = 6.5
    return _adjust_for_compute_type(estimate, compute_type)


def assess_live_vram_preflight(
    model_name: str,
    *,
    device: str,
    compute_type: str,
    memory_info: GpuMemoryInfo | None = None,
) -> LiveVramPreflight:
    """Assesses whether live transcription has enough currently free VRAM."""
    model_estimate = estimate_live_model_vram_gb(model_name, compute_type=compute_type)
    required = round(model_estimate + LIVE_VRAM_SAFETY_BUFFER_GB, 2)
    display_name = normalize_model_name(model_name) or model_name
    if str(device or "").strip().lower() != "cuda":
        return LiveVramPreflight(
            status="skipped",
            model_name=display_name,
            estimated_required_gb=required,
            model_estimate_gb=model_estimate,
            safety_buffer_gb=LIVE_VRAM_SAFETY_BUFFER_GB,
            free_gb=None,
            total_gb=None,
            used_gb=None,
            message="VRAM preflight skipped because CUDA is not the active transcription device.",
        )

    memory = memory_info if memory_info is not None else get_gpu_memory_info()
    if memory is None:
        return LiveVramPreflight(
            status="unavailable",
            model_name=display_name,
            estimated_required_gb=required,
            model_estimate_gb=model_estimate,
            safety_buffer_gb=LIVE_VRAM_SAFETY_BUFFER_GB,
            free_gb=None,
            total_gb=None,
            used_gb=None,
            message="Current GPU memory could not be read; live transcription will continue without a VRAM preflight.",
        )

    status = "ok" if memory.free_gb >= required else "low"
    message = (
        f"GPU VRAM available: {memory.free_gb:.1f} GB free of {memory.total_gb:.1f} GB total. "
        f"Estimated live transcription need for '{display_name}': {required:.1f} GB."
    )
    return LiveVramPreflight(
        status=status,
        model_name=display_name,
        estimated_required_gb=required,
        model_estimate_gb=model_estimate,
        safety_buffer_gb=LIVE_VRAM_SAFETY_BUFFER_GB,
        free_gb=memory.free_gb,
        total_gb=memory.total_gb,
        used_gb=memory.used_gb,
        message=message,
    )


def _adjust_for_compute_type(estimate_gb: float, compute_type: str) -> float:
    compute = str(compute_type or "").strip().lower()
    if compute in {"int8", "int8_float16", "int8_float32"}:
        return round(max(0.4, estimate_gb * 0.75), 2)
    return round(estimate_gb, 2)
