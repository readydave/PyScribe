"""Granite Speech runtime helpers."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Sequence


GRANITE_TRANSCRIBE_PROMPT = "<|audio|>can you transcribe the speech into a written format?"
GRANITE_SAMPLE_RATE = 16000
GRANITE_CHUNK_SECONDS = 30
_CONTROL_TOKEN_RE = re.compile(r"<\|[^|>]+?\|>")


@dataclass(frozen=True)
class GraniteSpeechModelBundle:
    processor: object
    model: object
    model_name: str
    device: str


def build_granite_prompt(*, keywords: Sequence[str] | None = None) -> str:
    """Build the Granite prompt, optionally biasing toward domain keywords."""
    prompt = GRANITE_TRANSCRIBE_PROMPT
    cleaned_keywords = [str(keyword).strip() for keyword in (keywords or ()) if str(keyword).strip()]
    if cleaned_keywords:
        prompt = f"{prompt} Keywords: {', '.join(cleaned_keywords)}"
    return prompt


def build_granite_chat_prompt(bundle: GraniteSpeechModelBundle, *, keywords: Sequence[str] | None = None) -> str:
    """Build the Granite chat-templated prompt expected by the processor."""
    tokenizer = getattr(bundle.processor, "tokenizer", None)
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError("Granite processor is missing a tokenizer with apply_chat_template().")
    chat = [{"role": "user", "content": build_granite_prompt(keywords=keywords)}]
    return str(tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))


def _sanitize_granite_text(text: str) -> str:
    cleaned = _CONTROL_TOKEN_RE.sub(" ", str(text or ""))
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _transcribe_granite_chunk(
    bundle: GraniteSpeechModelBundle,
    audio_chunk: object,
    *,
    keywords: Sequence[str] | None = None,
    max_new_tokens: int = 256,
) -> str:
    prompt = build_granite_chat_prompt(bundle, keywords=keywords)
    inputs = bundle.processor(prompt, audio_chunk, device=bundle.device, return_tensors="pt")
    if hasattr(inputs, "to"):
        inputs = inputs.to(bundle.device)

    generated_ids = bundle.model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
    )
    input_ids = getattr(inputs, "input_ids", None)
    prompt_token_count = 0
    if input_ids is not None and hasattr(input_ids, "shape") and len(input_ids.shape) >= 2:
        prompt_token_count = int(input_ids.shape[-1])
    if prompt_token_count:
        generated_ids = generated_ids[:, prompt_token_count:]

    tokenizer = getattr(bundle.processor, "tokenizer", None)
    if tokenizer is None or not hasattr(tokenizer, "batch_decode"):
        raise RuntimeError("Granite processor is missing a tokenizer with batch_decode().")
    decoded = tokenizer.batch_decode(generated_ids, add_special_tokens=False, skip_special_tokens=True)
    if not decoded:
        return ""
    return _sanitize_granite_text(str(decoded[0]))


def _resolve_torch_dtype(torch_module: object, *, device: str, compute_type: str):
    if device == "cuda":
        if hasattr(torch_module, "cuda") and hasattr(torch_module.cuda, "is_bf16_supported"):
            try:
                if bool(torch_module.cuda.is_bf16_supported()):
                    return getattr(torch_module, "bfloat16")
            except Exception:
                pass
        if compute_type == "float16":
            return getattr(torch_module, "float16")
    return getattr(torch_module, "float32")


def load_granite_model(
    model_name: str,
    *,
    device: str,
    compute_type: str,
) -> GraniteSpeechModelBundle:
    """Load a Granite Speech processor/model pair."""
    try:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "Granite Speech requires the 'transformers' and 'peft' packages. "
            "Install them in the project virtual environment before using this model."
        ) from exc

    load_name = str(model_name or "").strip()
    local_files_only = os.path.isdir(load_name)
    torch_dtype = _resolve_torch_dtype(torch, device=device, compute_type=compute_type)
    processor = AutoProcessor.from_pretrained(
        load_name,
        local_files_only=local_files_only,
        trust_remote_code=False,
    )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        load_name,
        local_files_only=local_files_only,
        trust_remote_code=False,
        torch_dtype=torch_dtype,
    )
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    return GraniteSpeechModelBundle(
        processor=processor,
        model=model,
        model_name=load_name,
        device=device,
    )


def transcribe_granite_audio(
    bundle: GraniteSpeechModelBundle,
    audio_np: object,
    *,
    keywords: Sequence[str] | None = None,
    max_new_tokens: int = 256,
    chunk_seconds: int = GRANITE_CHUNK_SECONDS,
    progress_cb: object | None = None,
) -> str:
    """Generate a transcript from Granite Speech for a mono 16 kHz waveform."""
    try:
        total_samples = int(len(audio_np))
    except TypeError:
        return _transcribe_granite_chunk(bundle, audio_np, keywords=keywords, max_new_tokens=max_new_tokens)

    chunk_size = max(int(chunk_seconds * GRANITE_SAMPLE_RATE), GRANITE_SAMPLE_RATE)
    if total_samples <= chunk_size:
        return _transcribe_granite_chunk(bundle, audio_np, keywords=keywords, max_new_tokens=max_new_tokens)

    parts: list[str] = []
    total_chunks = max((total_samples + chunk_size - 1) // chunk_size, 1)
    for index, start in enumerate(range(0, total_samples, chunk_size), start=1):
        end = min(start + chunk_size, total_samples)
        piece = audio_np[start:end]
        text = _transcribe_granite_chunk(bundle, piece, keywords=keywords, max_new_tokens=max_new_tokens)
        if text:
            parts.append(text)
        if callable(progress_cb):
            progress_cb((index / total_chunks) * 100.0)
    return "\n".join(parts).strip()
