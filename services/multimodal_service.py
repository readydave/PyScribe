"""Optional video-stream analysis helpers (slides/chat OCR)."""

from __future__ import annotations

from collections import Counter
import ctypes
from difflib import SequenceMatcher
import importlib.util
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from threading import Event
from typing import Callable

import ffmpeg

from services.runtime_env_service import configure_runtime_environment
from utils import get_ffmpeg_cmd

StatusCallback = Callable[[str], None]
ProgressCallback = Callable[[float], None]
_UI_NOISE_TERMS = (
    "chat people raise react",
    "camera mic share leave",
    "take control",
    "pop out",
    "copilot",
    "apps",
    "mute mic",
    "share leave",
    "sysadmin meeting",
    "banfield",
)
_PADDLE_OCR = None
_SURYA_DET_PREDICTOR = None
_SURYA_REC_PREDICTOR = None
_VISUAL_PROFILE_SETTINGS: dict[str, dict[str, float | int]] = {
    # Faster runtime: fewer chat scans + more aggressive dedupe.
    "fast": {
        "slide_change_threshold": 0.020,
        "chat_change_threshold": 0.028,
        "chat_stride": 3,
        "max_frames": 180,
    },
    # Good default quality/speed balance.
    "balanced": {
        "slide_change_threshold": 0.014,
        "chat_change_threshold": 0.020,
        "chat_stride": 2,
        "max_frames": 240,
    },
    # Most complete capture (slowest).
    "accurate": {
        "slide_change_threshold": 0.010,
        "chat_change_threshold": 0.014,
        "chat_stride": 1,
        "max_frames": 360,
    },
}


@dataclass
class VisualAnalysisResult:
    report: str
    frames_scanned: int
    elapsed_seconds: float
    cancelled: bool
    available: bool
    reason: str | None = None


def check_ocr_backend_ready(backend: str) -> tuple[bool, str | None]:
    requested = (backend or "auto").strip().lower()
    if requested == "auto":
        return True, None

    if requested == "paddleocr":
        if importlib.util.find_spec("paddleocr") is None:
            return False, "Install PaddleOCR: pip install paddleocr paddlepaddle"
        return True, None

    if requested == "surya":
        if importlib.util.find_spec("surya") is None:
            return (
                False,
                "Install Surya OCR: pip install surya-ocr (note: this may require a newer Torch stack).",
            )
        return True, None

    if requested == "pytesseract":
        if importlib.util.find_spec("pytesseract") is None:
            return False, "Install pytesseract: pip install pytesseract"
        if shutil.which("tesseract") is None:
            return False, "Install the OS package for Tesseract and ensure the `tesseract` command is in PATH."
        return True, None

    return False, f"Unknown OCR backend '{backend}'."


def analyze_video_stream(
    media_path: str,
    *,
    ocr_backend: str = "paddleocr",
    visual_profile: str = "balanced",
    sample_seconds: float = 1.0,
    max_frames: int | None = None,
    cancel_event: Event | None = None,
    on_status: StatusCallback | None = None,
    on_progress: ProgressCallback | None = None,
) -> VisualAnalysisResult:
    """
    Samples video frames and extracts on-screen text (OCR) when available.
    """
    started = time.perf_counter()
    visual_profile = _normalize_visual_profile(visual_profile)
    profile_cfg = _VISUAL_PROFILE_SETTINGS[visual_profile]
    sample_seconds = max(0.5, float(sample_seconds or 1.0))
    if max_frames is None:
        max_frames = int(profile_cfg["max_frames"])
    max_frames = max(1, int(max_frames))

    if not _has_video_stream(media_path):
        return VisualAnalysisResult(
            report="",
            frames_scanned=0,
            elapsed_seconds=time.perf_counter() - started,
            cancelled=False,
            available=False,
            reason="Input has no video stream.",
        )

    requested_backend = (ocr_backend or "auto").strip().lower()
    ocr_fn, ocr_name, ocr_reason, backend_note = _build_ocr_fn(ocr_backend, on_status=on_status)
    if ocr_fn is None:
        if on_status:
            on_status(f"Visual analysis unavailable: {ocr_reason}")
        return VisualAnalysisResult(
            report=_format_unavailable_report(ocr_reason, requested_backend=requested_backend),
            frames_scanned=0,
            elapsed_seconds=time.perf_counter() - started,
            cancelled=False,
            available=False,
            reason=ocr_reason,
        )

    if on_status:
        on_status(
            f"Analyzing video frames for on-screen text ({ocr_name}, profile={visual_profile})..."
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        frames = _extract_sampled_frames(
            media_path=media_path,
            out_dir=temp_dir,
            sample_seconds=sample_seconds,
            max_frames=max_frames,
        )

        if not frames:
            reason = "No frames were extracted from the video stream."
            if on_status:
                on_status(f"Visual analysis: {reason}")
            return VisualAnalysisResult(
                report=_format_unavailable_report(reason, requested_backend=requested_backend),
                frames_scanned=0,
                elapsed_seconds=time.perf_counter() - started,
                cancelled=False,
                available=False,
                reason=reason,
            )

        from PIL import Image

        canonical_lines: dict[str, str] = {}
        line_counts: Counter[str] = Counter()
        line_source: dict[str, str] = {}
        timeline: list[str] = []
        prev_keys: set[str] = set()
        prev_slide_sig = None
        prev_chat_sig = None
        last_visible_slide_keys: set[str] = set()
        last_visible_chat_keys: set[str] = set()
        slide_change_threshold = float(profile_cfg["slide_change_threshold"])
        chat_change_threshold = float(profile_cfg["chat_change_threshold"])
        chat_stride = max(1, int(profile_cfg["chat_stride"]))
        ocr_calls_slide = 0
        ocr_calls_chat = 0
        dedupe_skipped_slide = 0
        dedupe_skipped_chat = 0
        for idx, frame_path in enumerate(frames, start=1):
            if cancel_event and cancel_event.is_set():
                return VisualAnalysisResult(
                    report=_format_visual_report(
                        partial=True,
                        sample_seconds=sample_seconds,
                        frames_scanned=idx - 1,
                        visual_profile=visual_profile,
                        requested_backend=requested_backend,
                        ocr_name=ocr_name,
                        backend_note=backend_note,
                        ocr_calls_slide=ocr_calls_slide,
                        ocr_calls_chat=ocr_calls_chat,
                        dedupe_skipped_slide=dedupe_skipped_slide,
                        dedupe_skipped_chat=dedupe_skipped_chat,
                        canonical_lines=canonical_lines,
                        line_counts=line_counts,
                        line_source=line_source,
                        total_frames=idx - 1,
                        timeline=timeline,
                    ),
                    frames_scanned=idx - 1,
                    elapsed_seconds=time.perf_counter() - started,
                    cancelled=True,
                    available=True,
                    reason="Cancelled during visual analysis.",
                )

            with Image.open(frame_path) as frame_img:
                rois = _extract_rois(frame_img)
                frame_lines: list[tuple[str, str]] = []
                current_keys: set[str] = set()
                newly_visible: list[tuple[str, str]] = []
                detected_slide_keys: set[str] = set()
                detected_chat_keys: set[str] = set()
                try:
                    slide_sig = _roi_signature(rois["slide"])
                    should_ocr_slide = idx == 1 or _signature_changed(prev_slide_sig, slide_sig, slide_change_threshold)
                    prev_slide_sig = slide_sig
                    if should_ocr_slide:
                        slide_text = ocr_fn(rois["slide"], mode="slide")
                        frame_lines.extend(("slide", line) for line in _extract_ocr_lines(slide_text))
                        ocr_calls_slide += 1
                    else:
                        dedupe_skipped_slide += 1
                        for key in last_visible_slide_keys:
                            line_counts[key] += 1
                            current_keys.add(key)

                    if "chat" in rois:
                        chat_sig = _roi_signature(rois["chat"])
                        chat_changed = idx == 1 or _signature_changed(prev_chat_sig, chat_sig, chat_change_threshold)
                        prev_chat_sig = chat_sig
                        should_ocr_chat = chat_changed and (idx == 1 or idx % chat_stride == 0)
                        if should_ocr_chat:
                            chat_text = ocr_fn(rois["chat"], mode="chat")
                            frame_lines.extend(("chat", line) for line in _extract_ocr_lines(chat_text))
                            ocr_calls_chat += 1
                        else:
                            dedupe_skipped_chat += 1
                            for key in last_visible_chat_keys:
                                line_counts[key] += 1
                                current_keys.add(key)
                    else:
                        prev_chat_sig = None
                        last_visible_chat_keys.clear()
                except Exception as exc:
                    reason = f"OCR runtime error: {exc}"
                    if on_status:
                        on_status(f"Visual analysis unavailable: {reason}")
                    return VisualAnalysisResult(
                        report=_format_unavailable_report(reason, requested_backend=requested_backend),
                        frames_scanned=idx - 1,
                        elapsed_seconds=time.perf_counter() - started,
                        cancelled=False,
                        available=False,
                        reason=reason,
                    )

            for source, line in frame_lines:
                if source == "slide" and _is_ui_noise_line(line):
                    continue
                if source == "chat" and _is_ui_noise_line(line) and not _is_chat_like(line):
                    continue
                key = _line_key(line)
                if not key:
                    continue
                canonical = _canonicalize_key(key, canonical_lines.keys())
                if canonical not in canonical_lines:
                    canonical_lines[canonical] = line
                    line_source[canonical] = source
                line_counts[canonical] += 1
                if line_source.get(canonical) == "slide" and source == "chat":
                    # If seen in chat panel later, classify as chat (higher value for chat capture).
                    line_source[canonical] = "chat"
                current_keys.add(canonical)
                if source == "slide":
                    detected_slide_keys.add(canonical)
                else:
                    detected_chat_keys.add(canonical)
                existing_new_lines = [text for _, text in newly_visible]
                if canonical not in prev_keys and not _line_exists_similar(canonical_lines[canonical], existing_new_lines):
                    newly_visible.append((line_source.get(canonical, source), canonical_lines[canonical]))

            if detected_slide_keys:
                last_visible_slide_keys = detected_slide_keys
            elif frame_lines:
                last_visible_slide_keys = set()
            if detected_chat_keys:
                last_visible_chat_keys = detected_chat_keys
            elif any(src == "chat" for src, _ in frame_lines):
                last_visible_chat_keys = set()

            if newly_visible:
                timestamp = _seconds_to_hhmmss((idx - 1) * sample_seconds)
                excerpt = " | ".join(f"[{src}] {txt}" for src, txt in newly_visible[:2])
                timeline.append(f"- [{timestamp}] {excerpt}")
            prev_keys = current_keys
            if on_progress:
                on_progress((idx / len(frames)) * 100.0)

        report = _format_visual_report(
            partial=False,
            sample_seconds=sample_seconds,
            frames_scanned=len(frames),
            visual_profile=visual_profile,
            requested_backend=requested_backend,
            ocr_name=ocr_name,
            backend_note=backend_note,
            ocr_calls_slide=ocr_calls_slide,
            ocr_calls_chat=ocr_calls_chat,
            dedupe_skipped_slide=dedupe_skipped_slide,
            dedupe_skipped_chat=dedupe_skipped_chat,
            canonical_lines=canonical_lines,
            line_counts=line_counts,
            line_source=line_source,
            total_frames=len(frames),
            timeline=timeline,
        )
        if on_status and not canonical_lines:
            on_status(
                "Visual analysis found no readable text. Try a shorter sample interval (0.5-1.0s) or higher-resolution source video."
            )
        return VisualAnalysisResult(
            report=report,
            frames_scanned=len(frames),
            elapsed_seconds=time.perf_counter() - started,
            cancelled=False,
            available=True,
        )


def _has_video_stream(media_path: str) -> bool:
    try:
        probe = ffmpeg.probe(media_path)
        streams = probe.get("streams", [])
    except Exception:
        return False
    return any(s.get("codec_type") == "video" for s in streams)


def _extract_sampled_frames(
    *,
    media_path: str,
    out_dir: str,
    sample_seconds: float,
    max_frames: int,
) -> list[str]:
    ffmpeg_cmd = get_ffmpeg_cmd(tool="ffmpeg")
    if not ffmpeg_cmd:
        return []

    pattern = os.path.join(out_dir, "frame_%06d.jpg")
    cmd = [
        ffmpeg_cmd,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        media_path,
        "-vf",
        f"fps=1/{sample_seconds}",
        "-frames:v",
        str(max_frames),
        "-q:v",
        "4",
        pattern,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return []
    frames = sorted(
        os.path.join(out_dir, name)
        for name in os.listdir(out_dir)
        if name.lower().endswith(".jpg")
    )
    return frames


def _build_ocr_fn(backend: str, *, on_status: StatusCallback | None = None):
    requested = (backend or "auto").strip().lower()
    attempts: list[str] = []
    attempt_reason_by_backend: dict[str, str] = {}

    backend_builders: dict[str, Callable[..., tuple[Callable | None, str | None]]] = {
        "paddleocr": _build_paddle_ocr_fn,
        "surya": _build_surya_ocr_fn,
        "pytesseract": lambda **_: _build_tesseract_ocr_fn(),
    }
    if requested == "auto":
        order = ["paddleocr", "surya", "pytesseract"]
    elif requested == "surya":
        order = ["surya", "paddleocr", "pytesseract"]
    elif requested == "paddleocr":
        order = ["paddleocr", "pytesseract"]
    elif requested == "pytesseract":
        order = ["pytesseract"]
    else:
        order = ["paddleocr", "surya", "pytesseract"]

    for name in order:
        fn, reason = backend_builders[name](on_status=on_status)
        if fn is not None:
            fallback_note = None
            if on_status and name != requested and requested != "auto":
                on_status(f"Requested OCR backend '{requested}' unavailable; using '{name}' fallback.")
                fallback_reason = attempt_reason_by_backend.get(requested, "unavailable")
                fallback_note = (
                    f"Requested backend '{requested}' unavailable: {fallback_reason}. "
                    f"Using '{name}' fallback."
                )
            elif on_status and requested == "auto" and name != "paddleocr":
                on_status(f"Using '{name}' OCR backend.")
                fallback_note = f"Auto mode selected '{name}' (higher-priority backends unavailable)."
            return fn, name, None, fallback_note
        attempt_reason_by_backend[name] = reason or "backend unavailable"
        attempts.append(f"{name}: {reason}")

    fn, reason = _build_tesseract_ocr_fn()
    if fn is not None:
        if on_status and attempts:
            on_status("Using pytesseract fallback OCR backend.")
        fallback_note = (
            f"Requested backend '{requested}' unavailable; using 'pytesseract' fallback."
            if requested != "pytesseract"
            else None
        )
        return fn, "pytesseract", None, fallback_note

    if attempts:
        reason = "; ".join(attempts)
    return None, "", reason or "No OCR backend available.", None


def _build_tesseract_ocr_fn():
    try:
        import pytesseract

        pytesseract.get_tesseract_version()

        def _ocr(image: "Image.Image", mode: str = "slide") -> str:
            config = "--psm 6"
            return pytesseract.image_to_string(image, config=config)

        return _ocr, None
    except Exception as exc:
        reason = (
            "Install OCR dependencies to enable visual analysis: "
            "pip install pytesseract pillow and install Tesseract OCR on the OS."
        )
        return None, f"{reason} ({exc})"


def _build_paddle_ocr_fn(*, on_status: StatusCallback | None = None):
    global _PADDLE_OCR
    configure_runtime_environment()
    _sanitize_ld_library_path(on_status=on_status)
    try:
        import paddle

        paddle_lib_dir = os.path.join(os.path.dirname(paddle.__file__), "libs")
        if os.path.isdir(paddle_lib_dir):
            current_ld = os.environ.get("LD_LIBRARY_PATH", "")
            paths = [p for p in current_ld.split(":") if p]
            if paddle_lib_dir not in paths:
                os.environ["LD_LIBRARY_PATH"] = ":".join([paddle_lib_dir, *paths])
            iomp_lib = os.path.join(paddle_lib_dir, "libiomp5.so")
            if os.path.isfile(iomp_lib):
                ctypes.CDLL(iomp_lib, mode=ctypes.RTLD_GLOBAL)
            mkl_lib = os.path.join(paddle_lib_dir, "libmklml_intel.so")
            if os.path.isfile(mkl_lib):
                ctypes.CDLL(mkl_lib, mode=ctypes.RTLD_GLOBAL)
            # Preload common Paddle bundled runtime libs before importing paddleocr/paddlex.
            for name in ("libphi.so", "libphi_core.so", "libdnnl.so.3"):
                path = os.path.join(paddle_lib_dir, name)
                if os.path.isfile(path):
                    try:
                        ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                    except Exception:
                        pass
        from paddleocr import PaddleOCR
    except Exception as exc:
        return None, (
            "Install/repair PaddleOCR backend: pip install paddleocr paddlepaddle; "
            "if you use Pinokio/Conda shell shims, run PyScribe in a clean shell with "
            "`unset LD_LIBRARY_PATH` "
            f"({exc})"
        )

    try:
        if _PADDLE_OCR is None:
            if on_status:
                on_status("Initializing PaddleOCR (first run may download OCR model files)...")
            _PADDLE_OCR = PaddleOCR(use_angle_cls=True, lang="en")

        def _ocr(image: "Image.Image", mode: str = "slide") -> str:
            import numpy as np

            image_np = np.array(image)
            min_conf = 0.45 if mode == "slide" else 0.35
            result = None
            predict_exc: Exception | None = None
            ocr_exc: Exception | None = None

            # PaddleOCR >=3 prefers `predict`; older versions expose `ocr`.
            if hasattr(_PADDLE_OCR, "predict"):
                try:
                    result = _PADDLE_OCR.predict(image_np)
                except Exception as exc:
                    predict_exc = exc

            if result is None and hasattr(_PADDLE_OCR, "ocr"):
                try:
                    result = _PADDLE_OCR.ocr(image_np, cls=True)
                except TypeError as exc:
                    if "unexpected keyword argument 'cls'" not in str(exc):
                        ocr_exc = exc
                    else:
                        try:
                            result = _PADDLE_OCR.ocr(image_np)
                        except Exception as sub_exc:
                            ocr_exc = sub_exc
                except Exception as exc:
                    ocr_exc = exc

            if result is None and (predict_exc or ocr_exc):
                details = []
                if predict_exc:
                    details.append(f"predict: {predict_exc}")
                if ocr_exc:
                    details.append(f"ocr: {ocr_exc}")
                raise RuntimeError("; ".join(details))

            lines = _extract_paddle_lines(result, min_conf=min_conf)
            return "\n".join(lines)

        return _ocr, None
    except Exception as exc:
        return None, f"PaddleOCR init/runtime error: {exc}"


def _extract_paddle_lines(result, *, min_conf: float) -> list[str]:
    lines: list[str] = []

    def _append_text(text, conf):
        text_val = str(text or "").strip()
        if not text_val:
            return
        try:
            conf_val = float(conf) if conf is not None else 1.0
        except Exception:
            conf_val = 1.0
        if conf_val >= min_conf:
            lines.append(text_val)

    def _consume(item):
        if item is None:
            return

        # PaddleOCR <=2 style: [poly, (text, conf)]
        if isinstance(item, (list, tuple)):
            if (
                len(item) >= 2
                and isinstance(item[1], (list, tuple))
                and len(item[1]) >= 2
                and isinstance(item[1][0], (str, bytes))
            ):
                _append_text(item[1][0], item[1][1])
                return
            for sub in item:
                _consume(sub)
            return

        # Dict-like outputs (PaddleOCR >=3 / PaddleX OCRResult serialized)
        mapping = None
        if isinstance(item, dict):
            mapping = item
        elif hasattr(item, "keys"):
            try:
                mapping = {k: item[k] for k in item.keys()}
            except Exception:
                mapping = None

        if mapping is not None:
            rec_texts = mapping.get("rec_texts")
            rec_scores = mapping.get("rec_scores")
            if isinstance(rec_texts, (list, tuple)):
                for idx, txt in enumerate(rec_texts):
                    conf = rec_scores[idx] if isinstance(rec_scores, (list, tuple)) and idx < len(rec_scores) else None
                    _append_text(txt, conf)
            elif rec_texts is not None:
                _append_text(rec_texts, mapping.get("rec_score") or mapping.get("score"))

            if "text" in mapping:
                _append_text(mapping.get("text"), mapping.get("confidence") or mapping.get("score"))

            for key in ("result", "results", "data", "ocr_res", "output"):
                if key in mapping:
                    _consume(mapping.get(key))
            return

        # Generic object fallback
        text = getattr(item, "text", None)
        if text is not None:
            _append_text(text, getattr(item, "confidence", None))

    _consume(result)
    return lines


def _sanitize_ld_library_path(*, on_status: StatusCallback | None = None) -> None:
    """
    Remove known conflicting runtime library paths injected by external app launchers.
    """
    raw = os.environ.get("LD_LIBRARY_PATH", "")
    if not raw:
        return
    parts = [p for p in raw.split(":") if p]
    blocked_markers = ("pinokio", "facefusion-pinokio.git/.env/lib")
    kept = [p for p in parts if not any(marker in p for marker in blocked_markers)]
    if len(kept) != len(parts):
        os.environ["LD_LIBRARY_PATH"] = ":".join(kept)
        if on_status:
            on_status("Detected conflicting LD_LIBRARY_PATH entries; using a cleaned runtime library path for OCR.")


def _build_surya_ocr_fn(*, on_status: StatusCallback | None = None):
    global _SURYA_DET_PREDICTOR, _SURYA_REC_PREDICTOR
    try:
        from surya.detection import DetectionPredictor
        from surya.recognition import RecognitionPredictor
    except Exception as exc:
        return None, f"Install Surya OCR backend: pip install surya-ocr ({exc})"

    try:
        if _SURYA_DET_PREDICTOR is None or _SURYA_REC_PREDICTOR is None:
            if on_status:
                on_status("Initializing Surya OCR (first run may download OCR model files)...")
            _SURYA_DET_PREDICTOR = DetectionPredictor()
            _SURYA_REC_PREDICTOR = RecognitionPredictor()

        def _ocr(image: "Image.Image", mode: str = "slide") -> str:
            predictions = _SURYA_REC_PREDICTOR([image], [["en"]], _SURYA_DET_PREDICTOR)
            lines: list[str] = []
            for pred in predictions or []:
                text_lines = getattr(pred, "text_lines", None) or getattr(pred, "lines", None) or []
                for line in text_lines:
                    text = getattr(line, "text", None)
                    if text is None and isinstance(line, dict):
                        text = line.get("text")
                    conf = getattr(line, "confidence", None)
                    if conf is None and isinstance(line, dict):
                        conf = line.get("confidence")
                    try:
                        conf_val = float(conf) if conf is not None else 1.0
                    except Exception:
                        conf_val = 1.0
                    min_conf = 0.45 if mode == "slide" else 0.35
                    if text and conf_val >= min_conf:
                        lines.append(str(text).strip())
                if not text_lines and getattr(pred, "text", None):
                    lines.append(str(pred.text).strip())
            return "\n".join([line for line in lines if line])

        return _ocr, None
    except Exception as exc:
        return None, f"Surya init/runtime error: {exc}"


def _normalize_visual_profile(profile: str) -> str:
    normalized = str(profile or "").strip().lower()
    if normalized in _VISUAL_PROFILE_SETTINGS:
        return normalized
    return "balanced"


def _roi_signature(image: "Image.Image"):
    import numpy as np

    # Small grayscale signature for cheap change detection.
    sig = image.convert("L").resize((32, 32))
    return np.asarray(sig, dtype=np.uint8)


def _signature_changed(prev_sig, cur_sig, threshold: float) -> bool:
    if prev_sig is None:
        return True
    try:
        import numpy as np

        delta = np.mean(np.abs(cur_sig.astype(np.int16) - prev_sig.astype(np.int16))) / 255.0
        return float(delta) > float(threshold)
    except Exception:
        return True


def _extract_ocr_lines(text: str) -> list[str]:
    if not text:
        return []
    lines: list[str] = []
    for raw in text.splitlines():
        normalized = re.sub(r"\s+", " ", raw).strip()
        if len(normalized) < 4 or len(normalized) > 180:
            continue
        if not re.search(r"[A-Za-z0-9]{3,}", normalized):
            continue
        if re.fullmatch(r"[-_=+|/\\.,:;!?~`'\"()\[\]{}<>* ]+", normalized):
            continue
        lines.append(normalized[:180])
    return lines


def _extract_rois(image: "Image.Image") -> dict[str, "Image.Image"]:
    w, h = image.size
    top = int(h * 0.12)
    bottom = int(h * 0.93)
    left = int(w * 0.01)
    slide_right = int(w * 0.84)
    if slide_right - left < 240:
        slide_right = int(w * 0.99)

    rois = {
        "slide": image.crop((left, top, slide_right, bottom)),
    }
    chat_left = int(w * 0.84)
    if w - chat_left >= 180:
        rois["chat"] = image.crop((chat_left, top, int(w * 0.995), int(h * 0.95)))
    return rois


def _line_key(line: str) -> str:
    key = re.sub(r"[^a-z0-9: ]+", " ", line.lower())
    key = re.sub(r"\s+", " ", key).strip()
    if len(key) < 4:
        return ""
    return key


def _canonicalize_key(key: str, existing_keys) -> str:
    for existing in existing_keys:
        if key == existing:
            return existing
        if SequenceMatcher(None, key, existing).ratio() >= 0.92:
            return existing
    return key


def _line_exists_similar(candidate: str, existing_lines: list[str]) -> bool:
    for line in existing_lines:
        if SequenceMatcher(None, candidate.lower(), line.lower()).ratio() >= 0.94:
            return True
    return False


def _is_chat_like(line: str) -> bool:
    return bool(re.match(r"^[A-Za-z][A-Za-z0-9_. -]{0,24}:\s+\S+", line))


def _is_ui_noise_line(line: str) -> bool:
    lower = line.lower()
    if any(term in lower for term in _UI_NOISE_TERMS):
        return True
    if re.search(r"\b(chat|people|raise|react|camera|mic|share|leave|copilot|apps|view|notes)\b", lower):
        if len(lower.split()) >= 4:
            return True
    if re.match(r"^o\s*\d{3,}", lower):
        return True
    return False


def _is_persistent_noise(line: str, count: int, total_frames: int) -> bool:
    if total_frames <= 0:
        return False
    if count >= max(8, int(total_frames * 0.6)):
        if len(line) <= 36 or _is_ui_noise_line(line):
            return True
    return False


def _is_low_value_slide_line(line: str) -> bool:
    normalized = re.sub(r"\s+", " ", line or "").strip()
    if not normalized:
        return True
    if re.search(r"\d", normalized):
        return False
    words = re.findall(r"[A-Za-z]+", normalized)
    if len(words) <= 1:
        return True
    # Filter likely participant-name fragments from side panes.
    if len(words) <= 2 and all(w[:1].isupper() for w in words):
        return True
    return False


def _seconds_to_hhmmss(total_seconds: float) -> str:
    seconds = max(0, int(total_seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _format_visual_report(
    *,
    partial: bool,
    sample_seconds: float,
    frames_scanned: int,
    visual_profile: str,
    requested_backend: str,
    ocr_name: str,
    backend_note: str | None,
    ocr_calls_slide: int,
    ocr_calls_chat: int,
    dedupe_skipped_slide: int,
    dedupe_skipped_chat: int,
    canonical_lines: dict[str, str],
    line_counts: Counter[str],
    line_source: dict[str, str],
    total_frames: int,
    timeline: list[str],
) -> str:
    title = "=== Visual Analysis (Beta, Partial) ===" if partial else "=== Visual Analysis (Beta) ==="
    unique_count = len(canonical_lines)
    lines = [
        title,
        f"- Visual mode: {visual_profile}",
        f"- OCR backend requested: {requested_backend}",
        f"- OCR engine used: {ocr_name}",
        f"- Frames sampled: {frames_scanned} (every {sample_seconds:.1f}s)",
        (
            "- OCR calls: "
            f"slide={ocr_calls_slide}, chat={ocr_calls_chat} "
            f"(dedupe skipped: slide={dedupe_skipped_slide}, chat={dedupe_skipped_chat})"
        ),
        f"- Unique text snippets: {unique_count}",
    ]
    if backend_note:
        lines.append(f"- Backend note: {backend_note}")

    if unique_count == 0:
        lines.append("- On-screen text: no readable text detected.")
        return "\n".join(lines)

    filtered_keys = [
        k
        for k in line_counts.keys()
        if not _is_persistent_noise(canonical_lines.get(k, ""), line_counts[k], total_frames)
    ]
    sorted_keys = sorted(filtered_keys, key=lambda k: (-line_counts[k], canonical_lines.get(k, "")))
    slide_keys = [
        k
        for k in sorted_keys
        if line_source.get(k, "slide") == "slide"
        and not _is_low_value_slide_line(canonical_lines.get(k, ""))
    ][:12]
    top_chat = [k for k in sorted_keys if line_source.get(k, "slide") == "chat"][:12]

    if slide_keys:
        lines.append("- Likely slide/presentation text:")
        for key in slide_keys:
            lines.append(f"  - {canonical_lines[key]} (seen {line_counts[key]}x)")

    if top_chat:
        lines.append("- Likely chat/right-panel text:")
        for key in top_chat:
            lines.append(f"  - {canonical_lines[key]} (seen {line_counts[key]}x)")

    if timeline:
        lines.append("- Timeline (new on-screen text):")
        lines.extend(timeline[:25])

    return "\n".join(lines)


def _format_unavailable_report(reason: str, *, requested_backend: str) -> str:
    return "\n".join(
        [
            "=== Visual Analysis (Beta) ===",
            f"- OCR backend requested: {requested_backend}",
            f"- Unavailable: {reason}",
        ]
    )
