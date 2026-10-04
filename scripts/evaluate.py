#!/usr/bin/env python3
"""Reproducible ASR/diarization evaluation (CLI only).

Usage:
    python scripts/evaluate.py --device cuda --compute-type float16
    python scripts/evaluate.py --device cpu --compute-type int8 --manifest eval/manifest.json

Manifest items (JSON): id, audio, language, reference_text (optional),
reference_rttm (optional, enables DER and speaker-label counts), diarize
(optional bool), max_speakers (optional int). Paths resolve against the repo
root. Results are written to eval_results/ (git-ignored).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

LOGGER = logging.getLogger("evaluate")

DEFAULT_MANIFEST = REPO_ROOT / "evaluation" / "manifest.json"
RESULTS_DIR = REPO_ROOT / "eval_results"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PyScribe ASR and diarization quality.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--compute-type", default=None, help="Default: float16 on cuda, int8 on cpu.")
    parser.add_argument("--diar-backend", default="accurate")
    parser.add_argument("--label", default="baseline", help="Config label stored with the results.")
    parser.add_argument("--only", default=None, help="Run only the manifest item with this id.")
    parser.add_argument("--out", type=Path, default=None, help="Results JSON path.")
    return parser.parse_args(argv)


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else REPO_ROOT / path


def _evaluate_item(item: dict[str, Any], args: argparse.Namespace, model: Any) -> dict[str, Any]:
    from services import eval_metrics as em
    from services.model_service import resolve_transcription_model
    from services.transcription_service import transcribe_prepared_audio
    from utils import convert_to_16k_mono, get_ffmpeg_cmd

    audio = _resolve(item["audio"])
    if not audio.is_file():
        raise FileNotFoundError(f"Audio not found: {audio}")
    reference_rttm = item.get("reference_rttm")
    diarize = bool(item.get("diarize", False))
    spec = resolve_transcription_model(args.model)

    with tempfile.TemporaryDirectory() as tmpdir, em.ResourceMonitor() as monitor:
        wav = convert_to_16k_mono(str(audio), tmpdir, get_ffmpeg_cmd())
        started = time.perf_counter()
        result = transcribe_prepared_audio(
            wav,
            model,
            item.get("language"),
            model_spec=spec,
            use_diarization=diarize,
            diar_backend=args.diar_backend,
            device=args.device,
            max_speakers=item.get("max_speakers"),
        )
        wall_seconds = time.perf_counter() - started

    row: dict[str, Any] = {
        "id": item["id"],
        "audio_seconds": round(result.duration_seconds, 2),
        "asr_seconds": round(result.transcription_seconds, 2),
        "diarization_seconds": round(result.diarization_seconds, 2),
        "wall_seconds": round(wall_seconds, 2),
        "rtf": round(em.real_time_factor(wall_seconds, result.duration_seconds), 4),
        "peak_ram_mb": round(monitor.peaks.ram_mb),
        "peak_vram_mb": round(monitor.peaks.vram_mb),
    }
    if item.get("reference_text"):
        ref_text = _resolve(item["reference_text"]).read_text(encoding="utf-8")
        scores = em.compute_wer_cer(ref_text, result.transcript_only)
        row.update({k: round(v, 4) for k, v in scores.items()})
    if diarize:
        row["unknown_speaker_lines"] = em.count_unknown_speakers(result.segments)
        if reference_rttm:
            ref_turns = em.parse_rttm(_resolve(reference_rttm))
            hyp_turns = [s for s in result.segments if s.get("speaker") != em.UNKNOWN_SPEAKER]
            row["der"] = round(em.compute_der(ref_turns, hyp_turns), 4)
            row["mislabelled_segments"] = em.count_mislabelled_segments(result.segments, ref_turns)
    return row


def _format_table(rows: list[dict[str, Any]]) -> str:
    cols = ("id", "wer", "cer", "der", "unknown_speaker_lines", "wall_seconds", "rtf", "peak_vram_mb", "peak_ram_mb")
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    compute_type = args.compute_type or ("float16" if args.device == "cuda" else "int8")
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.error("Cannot read manifest %s: %s", args.manifest, exc)
        return 2
    items = [i for i in manifest.get("items", []) if not args.only or i["id"] == args.only]
    if not items:
        LOGGER.error("No manifest items to run.")
        return 2

    from services.model_service import load_model, resolve_transcription_model

    spec = resolve_transcription_model(args.model)
    LOGGER.info("Loading %s on %s (%s)", args.model, args.device, compute_type)
    model = load_model(args.model, device=args.device, compute_type=compute_type, model_spec=spec)

    rows: list[dict[str, Any]] = []
    for item in items:
        LOGGER.info("Evaluating %s", item["id"])
        try:
            rows.append(_evaluate_item(item, args, model))
        except Exception as exc:  # keep going so one bad item doesn't lose the others
            LOGGER.exception("Item %s failed", item["id"])
            rows.append({"id": item["id"], "error": str(exc)})

    print(_format_table(rows))
    out = args.out or RESULTS_DIR / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{args.label}-{args.device}.json"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {"label": args.label, "model": args.model, "device": args.device,
                 "compute_type": compute_type, "rows": rows},
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        LOGGER.error("Could not write results to %s: %s", out, exc)
        return 1
    LOGGER.info("Results written to %s", out)
    return 1 if any("error" in r for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
