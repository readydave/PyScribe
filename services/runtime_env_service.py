"""Runtime environment helpers for portable packaging and model downloads."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import tempfile


def get_cache_root() -> str:
    """
    Returns a writable cache root for PyScribe.

    Priority:
      1) PYSCRIBE_CACHE_DIR
      2) XDG_CACHE_HOME/pyscribe
      3) ~/.cache/pyscribe
      4) /tmp/pyscribe-cache (fallback)
    """
    explicit = os.environ.get("PYSCRIBE_CACHE_DIR")
    if explicit:
        return _ensure_writable_dir(os.path.abspath(os.path.expanduser(explicit)))

    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return _ensure_writable_dir(os.path.join(os.path.abspath(os.path.expanduser(xdg_cache_home)), "pyscribe"))

    default_home_cache = os.path.join(os.path.expanduser("~"), ".cache", "pyscribe")
    return _ensure_writable_dir(default_home_cache)


def configure_runtime_environment() -> dict[str, str]:
    """
    Sets safe defaults for OCR model download/runtime paths.

    Uses `setdefault` so users can override via environment variables.
    """
    cache_root = get_cache_root()
    paddle_cache = os.path.join(cache_root, "paddle")
    paddlex_cache = os.path.join(cache_root, "paddlex")
    hf_home_fallback = os.path.join(cache_root, "huggingface")
    hf_hub_cache_fallback = os.path.join(hf_home_fallback, "hub")
    modelscope_cache_fallback = os.path.join(cache_root, "modelscope")
    os.makedirs(paddle_cache, exist_ok=True)
    os.makedirs(paddlex_cache, exist_ok=True)
    os.makedirs(hf_hub_cache_fallback, exist_ok=True)
    os.makedirs(modelscope_cache_fallback, exist_ok=True)

    os.environ.setdefault("PYSCRIBE_CACHE_DIR", cache_root)
    os.environ.setdefault("PADDLE_HOME", paddle_cache)
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", paddlex_cache)
    # Avoid slow/fragile hoster health checks in restricted networks.
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    # Work around certain Paddle/PIR + oneDNN runtime crashes seen on some hosts.
    os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")
    os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")
    # Keep default source explicit and overridable.
    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "huggingface")
    os.environ.setdefault("PADDLE_PDX_HUGGING_FACE_ENDPOINT", "https://huggingface.co")
    # Keep existing HF/ModelScope defaults when home cache is writable so we reuse
    # previously downloaded models. Only redirect to fallback cache when needed.
    if "HF_HOME" not in os.environ and "HUGGINGFACE_HUB_CACHE" not in os.environ:
        default_hf_hub = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        if not _is_writable_dir(default_hf_hub):
            os.environ.setdefault("HF_HOME", hf_home_fallback)
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hf_hub_cache_fallback)
    if "MODELSCOPE_CACHE" not in os.environ:
        default_modelscope = os.path.join(os.path.expanduser("~"), ".cache", "modelscope")
        if not _is_writable_dir(default_modelscope):
            os.environ.setdefault("MODELSCOPE_CACHE", modelscope_cache_fallback)
    # Ensure libraries that honor XDG cache also write to user-writable paths.
    os.environ.setdefault("XDG_CACHE_HOME", os.path.dirname(cache_root))

    return {
        "cache_root": cache_root,
        "paddle_cache": paddle_cache,
        "paddlex_cache": paddlex_cache,
    }


def prepare_linux_dynamic_loader_environment() -> dict[str, str | bool]:
    """
    On Linux, prepends runtime library paths needed by OCR and CUDA diarization, and
    strips known conflicting library-injection paths. Returns change metadata.
    """
    result: dict[str, str | bool] = {
        "changed": False,
        "paddle_lib_dir": "",
        "nvidia_lib_dirs": "",
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
    }
    if not sys.platform.startswith("linux"):
        return result

    original_ld = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in original_ld.split(":") if p]
    cleaned = _strip_conflicting_library_paths(parts)

    nvidia_lib_dirs = _detect_nvidia_cuda_lib_dirs()
    if nvidia_lib_dirs:
        preferred = list(nvidia_lib_dirs)
        cleaned = [p for p in cleaned if p not in preferred]
        cleaned = preferred + cleaned
        result["nvidia_lib_dirs"] = ":".join(preferred)

    paddle_lib_dir = _detect_paddle_libs_dir()
    if paddle_lib_dir:
        cleaned = [p for p in cleaned if p != paddle_lib_dir]
        cleaned = [paddle_lib_dir] + cleaned
        result["paddle_lib_dir"] = paddle_lib_dir

    new_ld = ":".join(cleaned)
    if new_ld != original_ld:
        os.environ["LD_LIBRARY_PATH"] = new_ld
        result["changed"] = True
    result["ld_library_path"] = os.environ.get("LD_LIBRARY_PATH", "")
    return result


def reexec_if_loader_env_changed() -> bool:
    """
    Re-executes the current process once when Linux dynamic loader env was changed.
    Returns True if re-exec was triggered.
    """
    if not sys.platform.startswith("linux"):
        return False
    if os.environ.get("PYSCRIBE_REEXEC_DONE") == "1":
        return False

    prep = prepare_linux_dynamic_loader_environment()
    if not prep.get("changed"):
        return False

    os.environ["PYSCRIBE_REEXEC_DONE"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)
    return True


def _ensure_writable_dir(path: str) -> str:
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".pyscribe_write_test")
        with open(probe, "w", encoding="utf-8") as handle:
            handle.write("ok")
        os.remove(probe)
        return path
    except Exception:
        fallback = os.path.join(tempfile.gettempdir(), "pyscribe-cache")
        os.makedirs(fallback, exist_ok=True)
        return fallback


def _is_writable_dir(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".pyscribe_write_test")
        with open(probe, "w", encoding="utf-8") as handle:
            handle.write("ok")
        os.remove(probe)
        return True
    except Exception:
        return False


def _detect_paddle_libs_dir() -> str:
    spec = importlib.util.find_spec("paddle")
    if spec and spec.origin:
        candidate = str((Path(spec.origin).resolve().parent / "libs"))
        if os.path.isdir(candidate):
            return candidate

    for search_path in sys.path:
        candidate = os.path.join(search_path, "paddle", "libs")
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)
    return ""


def _detect_nvidia_cuda_lib_dirs() -> list[str]:
    """
    Detects venv/site-packages CUDA runtime library dirs installed via PyTorch wheels.
    Returns directories in preferred lookup order.
    """
    pkg_order = (
        "cuda_nvrtc",
        "cuda_runtime",
        "cudnn",
        "cublas",
        "cufft",
        "curand",
        "cusolver",
        "cusparse",
        "nccl",
        "nvtx",
        "cuda_cupti",
    )
    candidates: list[str] = []
    search_roots: list[Path] = []

    for entry in sys.path:
        try:
            root = Path(entry).resolve()
        except Exception:
            continue
        if root.is_dir():
            search_roots.append(root)

    version_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
    prefix_site = Path(sys.prefix) / "lib" / version_tag / "site-packages"
    if prefix_site.is_dir():
        search_roots.insert(0, prefix_site)

    seen_roots: set[str] = set()
    unique_roots: list[Path] = []
    for root in search_roots:
        root_str = str(root)
        if root_str in seen_roots:
            continue
        seen_roots.add(root_str)
        unique_roots.append(root)

    for root in unique_roots:
        nvidia_root = root / "nvidia"
        if not nvidia_root.is_dir():
            continue
        for pkg in pkg_order:
            lib_dir = nvidia_root / pkg / "lib"
            if lib_dir.is_dir():
                lib_str = str(lib_dir)
                if lib_str not in candidates:
                    candidates.append(lib_str)

    return candidates


def _strip_conflicting_library_paths(parts: list[str]) -> list[str]:
    blocked_markers = (
        "pinokio",
        "facefusion-pinokio.git/.env/lib",
    )
    return [p for p in parts if not any(marker in p for marker in blocked_markers)]
