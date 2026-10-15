#!/usr/bin/env python3
"""Spike: Granite Speech 4.1 2B Plus word timestamps and speaker-attributed ASR.

Chunks the audio (the model card says timestamps are reliable up to 3.5 min), runs the timestamp prompt,
and reports WER against a reference, speed, VRAM, and word-time agreement with faster-whisper. Optional
`--saa` runs the speaker-attribution prompt on the first chunk and prints the speaker-tag counts.

    python scripts/spikes/granite_plus_spike.py --audio A.wav --reference A.ref.txt --chunk-seconds 180 --max-minutes 15
Results are written as JSON to --out (default: scratchpad-style path you pass; nothing is committed).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

LOGGER = logging.getLogger("granite_plus_spike")
MODEL = "ibm-granite/granite-speech-4.1-2b-plus"
SYSTEM_PROMPT = "Knowledge Cutoff Date: April 2024.\nToday's Date: December 19, 2024.\nYou are Granite, developed by IBM. You are a helpful AI assistant"
TS_PROMPT = "<|audio|> Timestamps: Transcribe the speech. After each word, add a timestamp tag showing the end time in centiseconds, e.g. hello [T:45] world [T:82]"
SAA_PROMPT = "<|audio|> Speaker attribution: Transcribe and denote who is speaking by adding [Speaker 1]: and [Speaker 2]: tags before speaker turns."
SAMPLE_RATE = 16_000


def parse_timestamped(text: str, base: float) -> list[dict[str, Any]]:
    """Turn 'word [T:45] word [T:82]' into words with end times (10 s rollover unwrapped); `_` is silence."""
    parts = re.split(r"\[T:(\d+)\]", text)
    words: list[dict[str, Any]] = []
    last_end, offset, prev_end = 0.0, 0.0, 0.0
    for word, stamp in zip(parts[::2], parts[1::2]):
        end = float(stamp) / 100
        while end + offset < last_end:
            offset += 10
        last_end = end + offset
        token = word.strip()
        if token and token != "_":
            words.append({"text": token, "start": base + prev_end, "end": base + last_end})
        prev_end = last_end
    return words


def run_model(audio: Any, prompt: str, *, bundle: tuple[Any, Any, Any, str], max_new_tokens: int) -> str:
    import torch

    processor, tokenizer, model, device = bundle
    chat = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = processor(prompt_text, audio, device=device, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)
    return str(tokenizer.decode(out[0, inputs["input_ids"].shape[-1] :], add_special_tokens=False, skip_special_tokens=True))


def vad_chunks(audio: Any, max_seconds: float, max_gap: float = 8.0) -> list[tuple[int, int]]:
    """Group Silero speech segments into (start, end) sample spans of at most `max_seconds`.

    A new span starts at a silence longer than `max_gap` seconds: the model's timestamp tag keeps only
    the last three digits (10 s rollover), so long silences inside a chunk cannot be unwrapped.
    """
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    speech = get_speech_timestamps(audio, VadOptions(max_speech_duration_s=30.0))
    spans: list[tuple[int, int]] = []
    for seg in speech:
        start, end = int(seg["start"]), int(seg["end"])
        if spans and start - spans[-1][1] <= max_gap * SAMPLE_RATE and end - spans[-1][0] <= max_seconds * SAMPLE_RATE:
            spans[-1] = (spans[-1][0], end)
        else:
            spans.append((start, end))
    return spans


def whisper_words(audio: Any) -> list[dict[str, Any]]:
    from faster_whisper import WhisperModel

    model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
    segments, _ = model.transcribe(audio, word_timestamps=True, vad_filter=True, beam_size=5, no_repeat_ngram_size=4)
    return [{"text": w.word.strip(), "start": w.start, "end": w.end} for s in segments for w in (s.words or [])]


def time_agreement(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> dict[str, float]:
    """Match words in order (difflib on normalized text) and compare start times."""
    from difflib import SequenceMatcher

    from services.eval_metrics import normalize_text

    ta = [normalize_text(w["text"]) for w in a]
    tb = [normalize_text(w["text"]) for w in b]
    diffs = []
    for block in SequenceMatcher(None, ta, tb, autojunk=False).get_matching_blocks():
        for k in range(block.size):
            diffs.append(a[block.a + k]["start"] - b[block.b + k]["start"])
    if not diffs:
        return {}
    absd = sorted(abs(d) for d in diffs)
    return {
        "matched_words": len(diffs),
        "median_signed_ms": statistics.median(diffs) * 1000,
        "median_abs_ms": statistics.median(absd) * 1000,
        "p90_abs_ms": absd[int(len(absd) * 0.9)] * 1000,
        "within_640ms": sum(d <= 0.64 for d in absd) / len(absd),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--reference", help="plain-text reference for the processed span (omit with --max-minutes unless it matches)")
    parser.add_argument("--chunk-seconds", type=float, default=180.0)
    parser.add_argument("--max-minutes", type=float, default=0.0, help="process only the first N minutes (0 = all)")
    parser.add_argument("--vad-chunks", action="store_true", help="cut chunks at silences instead of fixed windows")
    parser.add_argument("--saa", action="store_true", help="also run the speaker-attribution prompt on the first chunk")
    parser.add_argument("--whisper", action="store_true", help="compare word times with faster-whisper")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import librosa
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    from services.eval_metrics import compute_wer_cer

    audio, _ = librosa.load(args.audio, sr=SAMPLE_RATE, mono=True)
    if args.max_minutes:
        audio = audio[: int(args.max_minutes * 60 * SAMPLE_RATE)]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL, device_map=device, dtype=torch.bfloat16).eval()
    bundle = (processor, processor.tokenizer, model, device)

    step = int(args.chunk_seconds * SAMPLE_RATE)
    words: list[dict[str, Any]] = []
    started = time.perf_counter()
    spans = vad_chunks(audio, args.chunk_seconds) if args.vad_chunks else [(o, min(o + step, audio.size)) for o in range(0, audio.size, step)]
    for offset, stop in spans:
        chunk = torch.from_numpy(audio[offset:stop]).unsqueeze(0)
        text = run_model(chunk, TS_PROMPT, bundle=bundle, max_new_tokens=10_000)
        got = parse_timestamped(text, offset / SAMPLE_RATE)
        LOGGER.info("chunk %.0fs: %d words", offset / SAMPLE_RATE, len(got))
        words.extend(got)
    elapsed = time.perf_counter() - started
    result: dict[str, Any] = {
        "audio_seconds": audio.size / SAMPLE_RATE,
        "seconds": elapsed,
        "rtf": elapsed / (audio.size / SAMPLE_RATE),
        "words": len(words),
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None,
    }
    if args.reference:
        reference = Path(args.reference).read_text(encoding="utf-8")
        result["wer"] = compute_wer_cer(reference, " ".join(w["text"] for w in words))
    if args.saa:
        saa = run_model(torch.from_numpy(audio[:step]).unsqueeze(0), SAA_PROMPT, bundle=bundle, max_new_tokens=4000)
        result["saa_speaker_tags"] = sorted(set(re.findall(r"\[Speaker \d+\]", saa)))
        result["saa_turns"] = len(re.findall(r"\[Speaker \d+\]:", saa))
        result["saa_head"] = saa[:400]
    del model
    torch.cuda.empty_cache()
    if args.whisper:
        reference_words = whisper_words(audio)
        result["vs_whisper"] = time_agreement(words, reference_words)
        result["_words"] = {"granite": words, "whisper": reference_words}
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    LOGGER.info(json.dumps({k: v for k, v in result.items() if k not in ("saa_head", "_words")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
