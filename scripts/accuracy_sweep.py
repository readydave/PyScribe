#!/usr/bin/env python3
"""Sweep faster-whisper settings and report WER on items with reference transcripts.

Scores every configuration on each manifest item twice: raw WER, and WER with spoken fillers
(uh, um, mm...) removed from both sides, because conversational references keep fillers.

    python scripts/accuracy_sweep.py --manifest evaluation/manifest.json --manifest eval/manifest-ami-asr.json
    python scripts/accuracy_sweep.py --manifest ... --only turbo-fp16-beam5,large-v3-fp16
Results go to eval_results/ (git-ignored).
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

LOGGER = logging.getLogger("accuracy_sweep")
FILLERS = {"uh", "um", "mm", "hmm", "erm", "mhm", "mmhmm", "ah", "eh", "huh", "uhm", "uhh", "umm", "hm"}

# name -> options. model/compute_type pick the model; the rest are transcribe() settings.
CONFIGS: dict[str, dict[str, Any]] = {
    "turbo-fp16-beam5": {"model": "large-v3-turbo", "compute_type": "float16", "beam_size": 5},
    "turbo-fp16-beam5-batched16": {"model": "large-v3-turbo", "compute_type": "float16", "beam_size": 5, "batch_size": 16},
    "turbo-int8f16-beam5": {"model": "large-v3-turbo", "compute_type": "int8_float16", "beam_size": 5},
    "turbo-fp32-beam5": {"model": "large-v3-turbo", "compute_type": "float32", "beam_size": 5},
    "turbo-fp16-beam1": {"model": "large-v3-turbo", "compute_type": "float16", "beam_size": 1},
    "turbo-fp16-beam10": {"model": "large-v3-turbo", "compute_type": "float16", "beam_size": 10},
    "turbo-fp16-noprev": {"model": "large-v3-turbo", "compute_type": "float16", "beam_size": 5, "condition_on_previous_text": False},
    "turbo-fp16-novad": {"model": "large-v3-turbo", "compute_type": "float16", "beam_size": 5, "vad_filter": False},
    "turbo-fp16-rep1.1": {"model": "large-v3-turbo", "compute_type": "float16", "beam_size": 5, "repetition_penalty": 1.1},
    "turbo-fp16-ngram4": {"model": "large-v3-turbo", "compute_type": "float16", "beam_size": 5, "no_repeat_ngram_size": 4},
    "turbo-fp16-halluc2": {"model": "large-v3-turbo", "compute_type": "float16", "beam_size": 5, "word_timestamps": True, "hallucination_silence_threshold": 2.0},
    "turbo-fp16-cr2.0": {"model": "large-v3-turbo", "compute_type": "float16", "beam_size": 5, "compression_ratio_threshold": 2.0},
    "turbo-fp32-beam10": {"model": "large-v3-turbo", "compute_type": "float32", "beam_size": 10},
    "turbo-int8f16-beam10": {"model": "large-v3-turbo", "compute_type": "int8_float16", "beam_size": 10},
    "turbo-fp32-ngram4": {"model": "large-v3-turbo", "compute_type": "float32", "beam_size": 5, "no_repeat_ngram_size": 4},
    "large-v3-fp16": {"model": "large-v3", "compute_type": "float16", "beam_size": 5},
    "large-v3-fp16-noprev": {"model": "large-v3", "compute_type": "float16", "beam_size": 5, "condition_on_previous_text": False},
    "large-v3-fp32": {"model": "large-v3", "compute_type": "float32", "beam_size": 5},
}


def _strip_fillers(text: str) -> str:
    return " ".join(word for word in text.split() if word not in FILLERS)


def _load_items(manifests: list[Path]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for manifest in manifests:
        for item in json.loads(manifest.read_text(encoding="utf-8"))["items"]:
            if item.get("reference_text"):
                items.append(item)
    return items


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else REPO_ROOT / path


def _decode(model: Any, audio: Any, language: str | None, options: dict[str, Any]) -> str:
    settings = {k: v for k, v in options.items() if k not in {"model", "compute_type", "batch_size"}}
    batch_size = options.get("batch_size")
    if batch_size:
        from faster_whisper import BatchedInferencePipeline

        settings.setdefault("vad_filter", True)
        segments, _ = BatchedInferencePipeline(model).transcribe(audio, language=language, batch_size=batch_size, **settings)
    else:
        settings.setdefault("vad_filter", True)
        segments, _ = model.transcribe(audio, language=language, **settings)
    return " ".join(segment.text.strip() for segment in segments)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--only", default=None, help="Comma-separated config names (default: all).")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from services import eval_metrics as em
    from services.model_service import load_model
    from utils import convert_to_16k_mono, get_ffmpeg_cmd, load_audio_waveform

    names = [n.strip() for n in args.only.split(",")] if args.only else list(CONFIGS)
    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        LOGGER.error("Unknown configs: %s", unknown)
        return 2
    items = _load_items(args.manifest)
    audio_cache: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for item in items:
            audio_cache[item["id"]] = load_audio_waveform(convert_to_16k_mono(str(_resolve(item["audio"])), tmp, get_ffmpeg_cmd()))

    results: list[dict[str, Any]] = []
    for name in names:
        options = CONFIGS[name]
        model = load_model(options["model"], device=args.device, compute_type=options["compute_type"], use_cache=False)
        row: dict[str, Any] = {"config": name, "items": {}}
        started = time.perf_counter()
        for item in items:
            reference = _resolve(item["reference_text"]).read_text(encoding="utf-8")
            hypothesis = _decode(model, audio_cache[item["id"]], item.get("language"), options)
            raw = em.compute_wer_cer(reference, hypothesis)["wer"]
            ref_n, hyp_n = em.normalize_text(reference), em.normalize_text(hypothesis)
            nofill = em.compute_wer_cer(_strip_fillers(ref_n), _strip_fillers(hyp_n))["wer"]
            row["items"][item["id"]] = {"wer": round(raw, 4), "wer_nofill": round(nofill, 4), "words": len(hyp_n.split())}
        row["seconds"] = round(time.perf_counter() - started, 1)
        results.append(row)
        LOGGER.info("%s done in %.0fs", name, row["seconds"])
        del model

    ids = [item["id"] for item in items]
    ami = [i for i in ids if i.startswith("ami")]
    print(f"{'config':30s} " + " ".join(f"{i[-12:]:>12s}" for i in ids) + "  ami_mean  ami_nofill  sec")
    for row in results:
        cells = " ".join(f"{row['items'][i]['wer'] * 100:11.2f}%" for i in ids)
        mean = statistics.mean(row["items"][i]["wer"] for i in ami) * 100 if ami else float("nan")
        mean_nf = statistics.mean(row["items"][i]["wer_nofill"] for i in ami) * 100 if ami else float("nan")
        print(f"{row['config']:30s} {cells}  {mean:7.2f}%  {mean_nf:9.2f}%  {row['seconds']:5.0f}")
    out = REPO_ROOT / "eval_results" / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-accuracy-sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    LOGGER.info("Results written to %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
