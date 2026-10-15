"""LLM post-processing execution service."""

from __future__ import annotations

import base64
from collections.abc import Callable
from dataclasses import dataclass
import json
import logging
import mimetypes
import os
import re
import ssl
import threading
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

from services.llm_connection_service import (
    CLI_PROVIDERS,
    LLMConnectionProfile,
    anthropic_endpoint_url,
    effective_context_tokens,
    effective_max_output_tokens,
    evaluate_profile_scope_policy,
    open_url,
    openai_endpoint_url,
    openai_max_tokens_field,
    provider_auth_headers,
    resolve_profile_api_key,
)
from services.multimodal_service import extract_text_from_images
from services.prompt_template_service import PromptTemplate
from services import cli_llm_provider
from services.secret_store import SecretStoreError


LOGGER = logging.getLogger(__name__)
_POSTPROCESS_TIMEOUT_RETRY_SECONDS = 30.0
_MULTIMODAL_MODEL_HINTS = (
    "vision",
    "vl",
    "llava",
    "bakllava",
    "qwen-vl",
    "qwen2-vl",
    "qwen2.5-vl",
    "minicpm-v",
    "internvl",
    "moondream",
    "phi-3.5-vision",
    "llama3.2-vision",
    "gpt-4o",
    "gpt-4.1",
    "gpt-5",
    "claude",
    "gemini",
    "pixtral",
    "llama-4",
    "gemma-3",
    "mistral-small-3",
)
OutputChunkCallback = Callable[[str], None]


class LLMRunControl:
    """Thread-safe cancel control for an in-flight LLM post-process request."""

    def __init__(self) -> None:
        self._cancel_event = threading.Event()
        self._lock = threading.Lock()
        self._active_response: Any | None = None

    def request_cancel(self) -> None:
        self._cancel_event.set()
        self._close_active_response()

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def set_active_response(self, response: Any) -> None:
        with self._lock:
            self._active_response = response
        if self._cancel_event.is_set():
            try:
                response.close()
            except Exception:
                pass

    def clear_active_response(self, response: Any | None = None) -> None:
        with self._lock:
            if response is None or self._active_response is response:
                self._active_response = None

    def _close_active_response(self) -> None:
        response = None
        with self._lock:
            response = self._active_response
        if response is None:
            return
        try:
            response.close()
        except Exception:
            pass


@dataclass(frozen=True)
class LLMPostprocessRequest:
    transcript_text: str
    ocr_text: str = ""
    notes_text: str = ""
    selected_model: str | None = None
    extra_context_text: str = ""
    image_paths: tuple[str, ...] = ()
    include_images: bool = True
    image_ocr_backend: str = "auto"
    ocr_fallback_for_images: bool = True


@dataclass(frozen=True)
class LLMPreparedPayload:
    status: str
    provider: str
    model: str | None
    payload_text: str
    image_paths_for_payload: tuple[str, ...]
    info_note: str | None
    error_code: str | None
    error_detail: str | None
    # The request the payload was built from (after any image-OCR fallback); used to split long transcripts.
    request_for_payload: LLMPostprocessRequest | None = None


@dataclass(frozen=True)
class LLMPostprocessResult:
    status: str
    provider: str
    model: str | None
    output_text: str
    error_code: str | None
    error_detail: str | None
    info_note: str | None = None


def prepare_llm_postprocess_payload(
    profile: LLMConnectionProfile,
    template: PromptTemplate,
    request: LLMPostprocessRequest,
) -> LLMPreparedPayload:
    """Validate and prepare the exact payload + image strategy for model submission."""
    transcript_text = (request.transcript_text or "").strip()
    if not transcript_text:
        return LLMPreparedPayload(
            status="fail",
            provider=profile.provider,
            model=request.selected_model or profile.default_model,
            payload_text="",
            image_paths_for_payload=(),
            info_note=None,
            error_code="missing_transcript",
            error_detail="Transcript text is required for post-processing.",
        )

    model = (request.selected_model or profile.default_model or "").strip() or None
    if model is None:
        return LLMPreparedPayload(
            status="fail",
            provider=profile.provider,
            model=None,
            payload_text="",
            image_paths_for_payload=(),
            info_note=None,
            error_code="missing_model",
            error_detail="No model is selected/configured for this profile.",
        )

    request_with_context = request
    image_paths = _normalize_image_paths(request.image_paths if request.include_images else ())
    image_payload_paths: tuple[str, ...] = ()
    info_note: str | None = None

    if image_paths:
        if _is_model_multimodal(model):
            image_payload_paths = image_paths
        elif request.ocr_fallback_for_images:
            image_ocr_text, backend_used, detail = extract_text_from_images(
                list(image_paths),
                ocr_backend=request.image_ocr_backend,
            )
            if not image_ocr_text:
                return LLMPreparedPayload(
                    status="fail",
                    provider=profile.provider,
                    model=model,
                    payload_text="",
                    image_paths_for_payload=(),
                    info_note=None,
                    error_code="image_context_unavailable",
                    error_detail=detail or "Image OCR fallback did not produce usable text.",
                )
            merged_extra = (request.extra_context_text or "").strip()
            merged_extra = (
                f"{merged_extra}\n\nImage OCR Context:\n{image_ocr_text}".strip()
                if merged_extra
                else f"Image OCR Context:\n{image_ocr_text}"
            )
            request_with_context = LLMPostprocessRequest(
                transcript_text=request.transcript_text,
                ocr_text=request.ocr_text,
                notes_text=request.notes_text,
                selected_model=request.selected_model,
                extra_context_text=merged_extra,
                image_paths=request.image_paths,
                include_images=False,
                image_ocr_backend=request.image_ocr_backend,
                ocr_fallback_for_images=request.ocr_fallback_for_images,
            )
            detail_part = f" ({detail})" if detail else ""
            info_note = (
                f"Model '{model}' treated as text-only. Used image OCR fallback"
                f"{f' with {backend_used}' if backend_used else ''}{detail_part}."
            )
        else:
            return LLMPreparedPayload(
                status="fail",
                provider=profile.provider,
                model=model,
                payload_text="",
                image_paths_for_payload=(),
                info_note=None,
                error_code="model_not_multimodal",
                error_detail=(
                    f"Model '{model}' does not appear multimodal and OCR fallback is disabled for image context."
                ),
            )

    payload_text = build_llm_payload_preview(
        template=template,
        request=request_with_context,
    )
    return LLMPreparedPayload(
        status="pass",
        provider=profile.provider,
        model=model,
        payload_text=payload_text,
        image_paths_for_payload=image_payload_paths,
        info_note=info_note,
        error_code=None,
        error_detail=None,
        request_for_payload=request_with_context,
    )


def run_llm_postprocess(
    profile: LLMConnectionProfile,
    template: PromptTemplate,
    request: LLMPostprocessRequest,
    *,
    prepared_payload: LLMPreparedPayload | None = None,
    on_output_chunk: OutputChunkCallback | None = None,
    run_control: LLMRunControl | None = None,
    on_status: Callable[[str], None] | None = None,
) -> LLMPostprocessResult:
    """Run LLM post-processing and return model output or error details.

    Transcripts too long for the model's context are split into parts, processed one by one,
    and merged (see ``_run_long``); ``on_status`` reports that progress.
    """
    if run_control and run_control.is_cancelled():
        return LLMPostprocessResult(
            status="fail",
            provider=profile.provider,
            model=request.selected_model or profile.default_model,
            output_text="",
            error_code="cancelled",
            error_detail="Generation cancelled by user.",
            info_note=None,
        )
    prepared = prepared_payload or prepare_llm_postprocess_payload(profile, template, request)
    if prepared.status != "pass":
        return LLMPostprocessResult(
            status="fail",
            provider=prepared.provider,
            model=prepared.model,
            output_text="",
            error_code=prepared.error_code,
            error_detail=prepared.error_detail,
            info_note=None,
        )

    policy_ok, policy_code, policy_detail = evaluate_profile_scope_policy(profile)
    if not policy_ok:
        return LLMPostprocessResult(
            status="fail",
            provider=prepared.provider,
            model=prepared.model,
            output_text="",
            error_code=policy_code,
            error_detail=policy_detail,
            info_note=None,
        )

    model = prepared.model
    assert model is not None
    limits = _limits_for(profile)
    image_paths = prepared.image_paths_for_payload
    LOGGER.info(
        "llm.run.start provider=%s scope=%s model=%s template_id=%s input_chars=%d images=%d",
        profile.provider,
        profile.scope,
        model,
        template.id,
        len(request.transcript_text or ""),
        len(image_paths),
    )
    notes: list[str | None] = [prepared.info_note]
    try:
        if _fits_in_context(template.system_prompt, prepared.payload_text, limits):
            outcome = _call_model(
                profile=profile,
                model=model,
                system_prompt=template.system_prompt,
                user_payload=prepared.payload_text,
                image_paths=image_paths,
                limits=limits,
                on_output_chunk=on_output_chunk,
                run_control=run_control,
            )
            notes.append(outcome.note())
            text = outcome.text
        else:
            text, long_notes = _run_long(
                profile=profile,
                model=model,
                template=template,
                request=prepared.request_for_payload or request,
                image_paths=image_paths,
                limits=limits,
                on_output_chunk=on_output_chunk,
                on_status=on_status,
                run_control=run_control,
            )
            notes.extend(long_notes)
    except _LLMPostprocessException as exc:
        detail = str(exc)
        if exc.code == "timeout":
            detail = (
                f"{detail} (request timeout {profile.timeout_seconds:.1f}s). "
                "Increase timeout in LLM Connections for larger/cold models."
            )
        return _fail(
            profile=profile,
            model=model,
            code=exc.code,
            detail=detail,
            info_note=_join_notes(notes),
            output_text=(exc.partial_output or "").strip(),
        )
    if not text:
        return _fail(
            profile=profile,
            model=model,
            code="empty_response",
            detail="Model returned an empty response.",
            info_note=_join_notes(notes),
        )
    LOGGER.info(
        "llm.run.complete provider=%s model=%s template_id=%s output_chars=%d",
        profile.provider,
        model,
        template.id,
        len(text),
    )
    return LLMPostprocessResult(
        status="pass",
        provider=profile.provider,
        model=model,
        output_text=text,
        error_code=None,
        error_detail=None,
        info_note=_join_notes(notes),
    )


# --- context budgeting and long transcripts ---------------------------------------------------

CHARS_PER_TOKEN = 3.2  # conservative for English; code and other languages use more tokens per character
_PROMPT_OVERHEAD_TOKENS = 600
_MIN_INPUT_TOKENS = 1000
_PART_INSTRUCTION = (
    "\n\nThis is part {index} of {total} of one long transcript. Produce the same kind of output for this part "
    "only. Keep every concrete fact, name, number, owner, date, and decision, because a later step will merge "
    "the parts."
)
_MERGE_INSTRUCTION = (
    "\n\nThe input contains {count} partial results made from consecutive parts of one long transcript, in order. "
    "Combine them into one complete result in the format described above. Remove duplicates, keep every distinct "
    "item, and keep names, numbers, owners, dates, and decisions exact."
)
_INTERMEDIATE_MERGE_INSTRUCTION = (
    "\n\nThe input contains {count} partial results made from consecutive parts of one long transcript, in order. "
    "Combine them into a single shorter partial result that keeps every distinct fact, name, number, owner, date, "
    "and decision. A later step will produce the final formatted output."
)


@dataclass(frozen=True)
class _Limits:
    context_tokens: int
    max_output_tokens: int
    temperature: float | None

    @property
    def input_budget(self) -> int:
        """Tokens available for the system prompt plus the user payload."""
        output = min(self.max_output_tokens, self.context_tokens // 2)
        return max(_MIN_INPUT_TOKENS, self.context_tokens - output - _PROMPT_OVERHEAD_TOKENS)

    @property
    def output_tokens(self) -> int:
        return min(self.max_output_tokens, max(256, self.context_tokens // 2))


def _limits_for(profile: LLMConnectionProfile) -> _Limits:
    return _Limits(
        context_tokens=effective_context_tokens(profile),
        max_output_tokens=effective_max_output_tokens(profile),
        temperature=profile.temperature,
    )


def estimate_tokens(text: str) -> int:
    """Rough token count (no tokenizer is bundled); deliberately on the high side."""
    return int(len(text or "") / CHARS_PER_TOKEN) + 1


def _fits_in_context(system_prompt: str, user_payload: str, limits: _Limits) -> bool:
    return estimate_tokens(system_prompt) + estimate_tokens(user_payload) <= limits.input_budget


def split_text_by_tokens(text: str, max_tokens: int) -> list[str]:
    """Split ``text`` on line boundaries into pieces of at most about ``max_tokens`` tokens."""
    max_chars = max(200, int(max_tokens * CHARS_PER_TOKEN))
    chunks: list[str] = []
    current: list[str] = []
    size = 0

    def flush() -> None:
        nonlocal current, size
        joined = "\n".join(current).strip()
        if joined:
            chunks.append(joined)
        current, size = [], 0

    for line in (text or "").splitlines():
        pieces = [line[i : i + max_chars] for i in range(0, len(line), max_chars)] or [""]
        for piece in pieces:
            if current and size + len(piece) + 1 > max_chars:
                flush()
            current.append(piece)
            size += len(piece) + 1
    flush()
    return chunks


def _group_by_budget(parts: list[str], budget_tokens: int) -> list[list[str]]:
    groups: list[list[str]] = []
    current: list[str] = []
    used = 0
    for part in parts:
        cost = estimate_tokens(part)
        if current and used + cost > budget_tokens:
            groups.append(current)
            current, used = [], 0
        current.append(part)
        used += cost
    if current:
        groups.append(current)
    return groups


def _run_long(
    *,
    profile: LLMConnectionProfile,
    model: str,
    template: PromptTemplate,
    request: LLMPostprocessRequest,
    image_paths: tuple[str, ...],
    limits: _Limits,
    on_output_chunk: OutputChunkCallback | None,
    on_status: Callable[[str], None] | None,
    run_control: LLMRunControl | None,
) -> tuple[str, list[str | None]]:
    """Process a transcript that does not fit the context: summarise each part, then merge the parts."""
    notes: list[str | None] = []
    scaffold_tokens = estimate_tokens(template.user_prompt_scaffold) + estimate_tokens(template.system_prompt)
    part_budget = max(_MIN_INPUT_TOKENS // 2, limits.input_budget - scaffold_tokens - 400)
    parts_text = split_text_by_tokens(request.transcript_text, part_budget)
    total = len(parts_text)
    notes.append(f"Long transcript: processed in {total} parts and merged.")

    partials: list[str] = []
    for index, chunk in enumerate(parts_text, start=1):
        _raise_if_cancelled(run_control, "")
        if on_status:
            on_status(f"Long transcript: processing part {index} of {total}...")
        chunk_request = LLMPostprocessRequest(transcript_text=chunk, include_images=False)
        outcome = _call_model(
            profile=profile,
            model=model,
            system_prompt=template.system_prompt + _PART_INSTRUCTION.format(index=index, total=total),
            user_payload=build_llm_payload_preview(template=template, request=chunk_request),
            image_paths=(),
            limits=limits,
            on_output_chunk=None,
            run_control=run_control,
        )
        notes.append(outcome.note())
        if outcome.text:
            partials.append(outcome.text)
    if not partials:
        return "", notes

    merge_budget = max(_MIN_INPUT_TOKENS // 2, limits.input_budget - scaffold_tokens - 800)
    while len(partials) > 1 and sum(estimate_tokens(p) for p in partials) > merge_budget:
        groups = _group_by_budget(partials, merge_budget)
        if len(groups) >= len(partials):
            break  # cannot shrink further; the final call will report a context error if it really doesn't fit
        merged: list[str] = []
        for number, group in enumerate(groups, start=1):
            _raise_if_cancelled(run_control, "")
            if on_status:
                on_status(f"Long transcript: combining results ({number} of {len(groups)})...")
            outcome = _call_model(
                profile=profile,
                model=model,
                system_prompt=template.system_prompt + _INTERMEDIATE_MERGE_INSTRUCTION.format(count=len(group)),
                user_payload=_build_merge_payload(template, group, None),
                image_paths=(),
                limits=limits,
                on_output_chunk=None,
                run_control=run_control,
            )
            notes.append(outcome.note())
            merged.append(outcome.text or "\n\n".join(group))
        partials = merged

    _raise_if_cancelled(run_control, "")
    if on_status:
        on_status("Long transcript: writing the final result...")
    final = _call_model(
        profile=profile,
        model=model,
        system_prompt=template.system_prompt + _MERGE_INSTRUCTION.format(count=len(partials)),
        user_payload=_build_merge_payload(template, partials, request),
        image_paths=image_paths,
        limits=limits,
        on_output_chunk=on_output_chunk,
        run_control=run_control,
    )
    notes.append(final.note())
    return final.text, notes


def _build_merge_payload(template: PromptTemplate, parts: list[str], request: LLMPostprocessRequest | None) -> str:
    sections = [template.user_prompt_scaffold.strip(), "", "Context:", "Partial results (in transcript order):"]
    for number, part in enumerate(parts, start=1):
        sections.extend(["", f"--- Part {number} of {len(parts)} ---", part.strip()])
    if request is not None:
        for title, value in (
            ("OCR Report:", request.ocr_text),
            ("Additional Notes:", request.notes_text),
            ("Pasted Context:", request.extra_context_text),
        ):
            if (value or "").strip():
                sections.extend(["", title, value.strip()])
    sections.extend(["", "Keep output concise and faithful to provided context."])
    return "\n".join(sections).strip()


# --- provider calls ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CallOutcome:
    text: str
    retry_note: str | None = None
    truncated: bool = False

    def note(self) -> str | None:
        parts = [self.retry_note]
        if self.truncated:
            parts.append("The model stopped at the maximum output length, so the result may be cut off. Raise 'Max output tokens' in LLM Connections.")
        return _join_notes(parts)


def _call_model(
    *,
    profile: LLMConnectionProfile,
    model: str,
    system_prompt: str,
    user_payload: str,
    image_paths: tuple[str, ...],
    limits: _Limits,
    on_output_chunk: OutputChunkCallback | None,
    run_control: LLMRunControl | None,
) -> _CallOutcome:
    """One request to the provider; raises ``_LLMPostprocessException`` on failure."""
    stream = bool(on_output_chunk)
    if profile.provider in CLI_PROVIDERS:
        return _call_cli(
            profile=profile,
            model=model,
            system_prompt=system_prompt,
            user_payload=user_payload,
            on_output_chunk=on_output_chunk,
            run_control=run_control,
        )
    try:
        api_key = resolve_profile_api_key(profile)
    except SecretStoreError as exc:
        raise _LLMPostprocessException("auth_failed", str(exc)) from None
    headers = provider_auth_headers(profile.provider, api_key)
    if profile.provider == "ollama":
        options: dict[str, Any] = {"num_ctx": limits.context_tokens, "num_predict": limits.output_tokens}
        if limits.temperature is not None:
            options["temperature"] = limits.temperature
        payload: dict[str, Any] = {
            "model": model,
            "system": system_prompt,
            "prompt": user_payload,
            "stream": stream,
            "options": options,
        }
        if image_paths:
            payload["images"] = [_encode_image_bytes(path) for path in image_paths]
        url = f"{profile.base_url.rstrip('/')}/api/generate"
        kind = "ollama"
    elif profile.provider == "anthropic":
        content: list[dict[str, Any]] = [{"type": "text", "text": user_payload}]
        for image_path in image_paths:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mimetypes.guess_type(image_path)[0] or "image/png",
                        "data": _encode_image_bytes(image_path),
                    },
                }
            )
        payload = {
            "model": model,
            "max_tokens": limits.output_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": content}],
            "stream": stream,
        }
        if limits.temperature is not None:
            payload["temperature"] = limits.temperature
        url = anthropic_endpoint_url(profile.base_url, "/messages")
        kind = "anthropic"
    else:
        user_content: Any = user_payload
        if image_paths:
            user_content = [{"type": "text", "text": user_payload}] + [
                {"type": "image_url", "image_url": {"url": _encode_image_data_url(path)}} for path in image_paths
            ]
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "stream": stream,
            openai_max_tokens_field(profile.base_url): limits.output_tokens,
        }
        if limits.temperature is not None:
            payload["temperature"] = limits.temperature
        url = openai_endpoint_url(profile.base_url, "/chat/completions", profile.scope)
        kind = "openai"

    if stream:
        text, truncated, retry_note = _stream_with_timeout_retry(
            kind=kind,
            url=url,
            payload=payload,
            timeout_seconds=profile.timeout_seconds,
            headers=headers,
            verify_tls=profile.verify_tls,
            on_output_chunk=on_output_chunk,  # type: ignore[arg-type]
            run_control=run_control,
        )
        return _CallOutcome(text=text, retry_note=retry_note, truncated=truncated)
    result, retry_note = _post_json_with_timeout_retry(
        url=url,
        payload=payload,
        timeout_seconds=profile.timeout_seconds,
        headers=headers,
        verify_tls=profile.verify_tls,
        run_control=run_control,
    )
    text, truncated = _extract_response(kind, result)
    return _CallOutcome(text=text, retry_note=retry_note, truncated=truncated)


def _call_cli(
    *,
    profile: LLMConnectionProfile,
    model: str,
    system_prompt: str,
    user_payload: str,
    on_output_chunk: OutputChunkCallback | None,
    run_control: LLMRunControl | None,
) -> _CallOutcome:
    """Run the user's signed-in CLI (tools off). The answer arrives in one piece, so the chunk callback fires once."""
    try:
        result = cli_llm_provider.run_claude(
            cli_llm_provider.build_cli_prompt(system_prompt, user_payload),
            model=model,
            timeout_seconds=profile.timeout_seconds,
            cancel_check=run_control.is_cancelled if run_control else None,
        )
    except cli_llm_provider.CliError as exc:
        raise _LLMPostprocessException(exc.code, exc.detail) from None
    if on_output_chunk:
        on_output_chunk(result.text)
    return _CallOutcome(text=result.text)


def _extract_response(kind: str, result: Any) -> tuple[str, bool]:
    """Pull (text, was_truncated) out of a non-streaming response."""
    if not isinstance(result, dict):
        return "", False
    if kind == "ollama":
        return str(result.get("response") or "").strip(), result.get("done_reason") == "length"
    if kind == "anthropic":
        blocks = result.get("content")
        texts = [
            str(block.get("text") or "")
            for block in (blocks if isinstance(blocks, list) else [])
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "".join(texts).strip(), result.get("stop_reason") == "max_tokens"
    choices = result.get("choices")
    finish = choices[0].get("finish_reason") if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
    return _extract_openai_content(result), finish == "length"


def build_llm_payload_preview(*, template: PromptTemplate, request: LLMPostprocessRequest) -> str:
    """Build the exact user payload that will be sent to the model."""
    sections: list[str] = []
    sections.append(template.user_prompt_scaffold.strip())
    sections.append("")
    sections.append("Context:")
    sections.append("Transcript:")
    sections.append(request.transcript_text.strip())
    ocr_text = (request.ocr_text or "").strip()
    if ocr_text:
        sections.append("")
        sections.append("OCR Report:")
        sections.append(ocr_text)
    notes_text = (request.notes_text or "").strip()
    if notes_text:
        sections.append("")
        sections.append("Additional Notes:")
        sections.append(notes_text)
    extra_context = (request.extra_context_text or "").strip()
    if extra_context:
        sections.append("")
        sections.append("Pasted Context:")
        sections.append(extra_context)
    if request.include_images and request.image_paths:
        sections.append("")
        sections.append("Image Attachments:")
        for path in _normalize_image_paths(request.image_paths):
            sections.append(f"- {os.path.basename(path)}")
    sections.append("")
    sections.append("Keep output concise and faithful to provided context.")
    return "\n".join(sections).strip()


def _normalize_image_paths(paths: tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    for path in paths:
        candidate = str(path or "").strip()
        if not candidate or not os.path.isfile(candidate):
            continue
        if candidate not in normalized:
            normalized.append(candidate)
    return tuple(normalized)


def _is_model_multimodal(model: str) -> bool:
    lowered = model.strip().lower()
    if not lowered:
        return False
    return any(hint in lowered for hint in _MULTIMODAL_MODEL_HINTS)


def _encode_image_bytes(path: str) -> str:
    with open(path, "rb") as handle:
        raw = handle.read()
    return base64.b64encode(raw).decode("ascii")


def _encode_image_data_url(path: str) -> str:
    mime_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
    encoded = _encode_image_bytes(path)
    return f"data:{mime_type};base64,{encoded}"


def _extract_openai_content(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_text = part.get("text")
            if isinstance(part_text, str) and part_text.strip():
                text_parts.append(part_text.strip())
        return "\n".join(text_parts).strip()
    return ""


# --- transport --------------------------------------------------------------------------------

_SECRET_PATTERN = re.compile(r"(sk-[A-Za-z0-9_\-]{6,}|AIza[A-Za-z0-9_\-]{10,}|Bearer\s+[A-Za-z0-9._\-]{10,})")
_CONTEXT_ERROR_HINTS = (
    "context length",
    "context window",
    "maximum context",
    "context_length",
    "too many tokens",
    "prompt is too long",
    "token limit",
    "exceeds the model",
    "input is too long",
    "too long",
)


def _redact(text: str) -> str:
    return _SECRET_PATTERN.sub("[redacted]", text)


def _error_message_from_body(body: str) -> str:
    """Best-effort human message from an API error body (OpenAI, Anthropic, Ollama styles)."""
    text = (body or "").strip()
    if not text:
        return ""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return _redact(text[:300])
    message = ""
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            message = str(error.get("message") or error.get("type") or "")
        elif isinstance(error, str):
            message = error
        elif isinstance(data.get("message"), str):
            message = data["message"]
    elif isinstance(data, list) and data and isinstance(data[0], dict):  # Gemini wraps errors in a list
        error = data[0].get("error")
        if isinstance(error, dict):
            message = str(error.get("message") or "")
    return _redact(" ".join(message.split())[:300])


def _read_error_body(exc: urlerror.HTTPError) -> str:
    try:
        return exc.read(4096).decode("utf-8", errors="replace")
    except Exception:
        return ""


def _translate_error(exc: Exception, partial_output: str = "") -> _LLMPostprocessException:
    """Map a transport exception to a coded error with a useful, secret-free message."""
    if isinstance(exc, urlerror.HTTPError):
        code = exc.code
        message = _error_message_from_body(_read_error_body(exc))
        suffix = f" {message}" if message else ""
        lowered = message.lower()
        if code in {401, 403}:
            err = _LLMPostprocessException("auth_failed", f"Authentication failed with HTTP {code}.{suffix}")
        elif code == 404:
            err = _LLMPostprocessException("api_mismatch", f"Endpoint path or model not found (HTTP 404).{suffix}")
        elif code == 429:
            err = _LLMPostprocessException("rate_limited", f"Rate limit or quota reached (HTTP 429).{suffix}")
        elif code == 413 or (code == 400 and any(hint in lowered for hint in _CONTEXT_ERROR_HINTS)):
            err = _LLMPostprocessException(
                "context_exceeded", f"The request is larger than the model's context window.{suffix}"
            )
        elif code in {503, 529}:
            err = _LLMPostprocessException("server_busy", f"The provider is overloaded (HTTP {code}).{suffix}")
        else:
            err = _LLMPostprocessException("server_error", f"Server returned HTTP {code}.{suffix}")
    elif isinstance(exc, urlerror.URLError):
        reason = str(exc.reason).lower()
        if "timed out" in reason:
            err = _LLMPostprocessException("timeout", "Connection timed out.")
        elif "certificate" in reason or "ssl" in reason:
            err = _LLMPostprocessException("tls_error", "TLS/certificate validation failed.")
        elif "name or service not known" in reason or "getaddrinfo" in reason:
            err = _LLMPostprocessException("dns_failure", "DNS resolution failed.")
        elif "connection refused" in reason or "failed to establish a new connection" in reason:
            err = _LLMPostprocessException("tcp_unreachable", "TCP connection was refused by endpoint.")
        else:
            err = _LLMPostprocessException("tcp_unreachable", "Unable to reach endpoint over TCP.")
    else:
        err = _LLMPostprocessException("timeout", "Connection timed out.")
    err.partial_output = partial_output
    return err


def _open_request(request: urlrequest.Request, *, timeout_seconds: float, verify_tls: bool):
    """Open ``request`` honouring the TLS setting (and, via ``open_url``, refusing redirects with credentials)."""
    if request.full_url.lower().startswith("https://"):
        context = ssl.create_default_context() if verify_tls else ssl._create_unverified_context()
        return open_url(request, timeout=timeout_seconds, context=context)
    return open_url(request, timeout=timeout_seconds)


def _build_request(url: str, payload: dict[str, Any], headers: dict[str, str]) -> urlrequest.Request:
    req_headers = {"Content-Type": "application/json"}
    req_headers.update(headers)
    return urlrequest.Request(url=url, method="POST", headers=req_headers, data=json.dumps(payload).encode("utf-8"))


def _http_json_post(
    *,
    url: str,
    payload: dict[str, Any],
    timeout_seconds: float,
    headers: dict[str, str],
    verify_tls: bool,
    run_control: LLMRunControl | None,
) -> Any:
    request = _build_request(url, payload, headers)
    response_handle: Any | None = None
    try:
        with _open_request(request, timeout_seconds=timeout_seconds, verify_tls=verify_tls) as response:
            response_handle = response
            if run_control:
                run_control.set_active_response(response)
            _raise_if_cancelled(run_control, "")
            body = response.read().decode("utf-8", errors="replace")
            _raise_if_cancelled(run_control, "")
    except (urlerror.HTTPError, urlerror.URLError, TimeoutError) as exc:
        raise _translate_error(exc) from exc
    finally:
        if run_control:
            run_control.clear_active_response(response_handle)
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise _LLMPostprocessException("api_mismatch", "Endpoint did not return valid JSON.") from exc


def _post_json_with_timeout_retry(
    *,
    url: str,
    payload: dict[str, Any],
    timeout_seconds: float,
    headers: dict[str, str],
    verify_tls: bool,
    run_control: LLMRunControl | None,
) -> tuple[Any, str | None]:
    try:
        return (
            _http_json_post(
                url=url,
                payload=payload,
                timeout_seconds=timeout_seconds,
                headers=headers,
                verify_tls=verify_tls,
                run_control=run_control,
            ),
            None,
        )
    except _LLMPostprocessException as exc:
        if exc.code != "timeout":
            raise
        retry_timeout = max(_POSTPROCESS_TIMEOUT_RETRY_SECONDS, float(timeout_seconds) * 2.0)
        if retry_timeout <= float(timeout_seconds) + 0.1:
            raise
        LOGGER.info(
            "llm.run.timeout.retry provider_url=%s timeout=%.1f retry_timeout=%.1f",
            url,
            timeout_seconds,
            retry_timeout,
        )
        result = _http_json_post(
            url=url,
            payload=payload,
            timeout_seconds=retry_timeout,
            headers=headers,
            verify_tls=verify_tls,
            run_control=run_control,
        )
        return result, f"Retried after timeout with {retry_timeout:.1f}s timeout."


def _stream_with_timeout_retry(
    *,
    kind: str,
    url: str,
    payload: dict[str, Any],
    timeout_seconds: float,
    headers: dict[str, str],
    verify_tls: bool,
    on_output_chunk: OutputChunkCallback,
    run_control: LLMRunControl | None,
) -> tuple[str, bool, str | None]:
    """Stream a response, retrying once with a longer timeout if nothing arrived. Returns (text, truncated, note)."""

    def _runner(current_timeout: float) -> tuple[str, bool]:
        return _stream_response(
            kind=kind,
            url=url,
            payload=payload,
            timeout_seconds=current_timeout,
            headers=headers,
            verify_tls=verify_tls,
            on_output_chunk=on_output_chunk,
            run_control=run_control,
        )

    try:
        text, truncated = _runner(timeout_seconds)
        return text, truncated, None
    except _LLMPostprocessException as exc:
        if exc.code != "timeout" or (exc.partial_output or "").strip():
            raise
        retry_timeout = max(_POSTPROCESS_TIMEOUT_RETRY_SECONDS, float(timeout_seconds) * 2.0)
        if retry_timeout <= float(timeout_seconds) + 0.1:
            raise
        LOGGER.info(
            "llm.run.timeout.retry provider_url=%s timeout=%.1f retry_timeout=%.1f",
            url,
            timeout_seconds,
            retry_timeout,
        )
        text, truncated = _runner(retry_timeout)
        return text, truncated, f"Retried after timeout with {retry_timeout:.1f}s timeout."


def _stream_response(
    *,
    kind: str,
    url: str,
    payload: dict[str, Any],
    timeout_seconds: float,
    headers: dict[str, str],
    verify_tls: bool,
    on_output_chunk: OutputChunkCallback,
    run_control: LLMRunControl | None,
) -> tuple[str, bool]:
    """Read a streaming response of the given ``kind`` (ollama JSON lines, or OpenAI/Anthropic SSE)."""
    request = _build_request(url, payload, headers)
    chunks: list[str] = []
    truncated = False
    response_handle: Any | None = None
    try:
        with _open_request(request, timeout_seconds=timeout_seconds, verify_tls=verify_tls) as response:
            response_handle = response
            if run_control:
                run_control.set_active_response(response)
            _raise_if_cancelled(run_control, "")
            for raw_line in response:
                _raise_if_cancelled(run_control, "".join(chunks))
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                if kind != "ollama":
                    if not line.startswith("data:"):
                        continue
                    line = line[5:].strip()
                    if not line:
                        continue
                    if line == "[DONE]":
                        break
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                piece, finished, cut_off = _parse_stream_event(kind, event)
                truncated = truncated or cut_off
                if piece:
                    chunks.append(piece)
                    _raise_if_cancelled(run_control, "".join(chunks))
                    on_output_chunk(piece)
                    _raise_if_cancelled(run_control, "".join(chunks))
                if finished:
                    break
    except _LLMPostprocessException as exc:
        if not exc.partial_output:
            exc.partial_output = "".join(chunks)
        raise
    except (urlerror.HTTPError, urlerror.URLError, TimeoutError) as exc:
        raise _translate_error(exc, "".join(chunks)) from exc
    finally:
        if run_control:
            run_control.clear_active_response(response_handle)
    return "".join(chunks).strip(), truncated


def _parse_stream_event(kind: str, event: Any) -> tuple[str, bool, bool]:
    """Return (text piece, stream finished, output truncated) for one decoded stream event."""
    if not isinstance(event, dict):
        return "", False, False
    if kind == "ollama":
        piece = event.get("response")
        done = bool(event.get("done"))
        return (piece if isinstance(piece, str) else ""), done, done and event.get("done_reason") == "length"
    if kind == "anthropic":
        event_type = event.get("type")
        if event_type == "content_block_delta":
            delta = event.get("delta")
            if isinstance(delta, dict) and isinstance(delta.get("text"), str):
                return delta["text"], False, False
            return "", False, False
        if event_type == "message_delta":
            delta = event.get("delta")
            return "", False, isinstance(delta, dict) and delta.get("stop_reason") == "max_tokens"
        if event_type == "message_stop":
            return "", True, False
        if event_type == "error":
            error = event.get("error")
            message = _redact(str(error.get("message") if isinstance(error, dict) else "")[:300])
            code = "server_busy" if isinstance(error, dict) and error.get("type") == "overloaded_error" else "server_error"
            raise _LLMPostprocessException(code, f"The provider reported an error while streaming. {message}".strip())
        return "", False, False
    choices = event.get("choices")
    cut_off = (
        isinstance(choices, list)
        and bool(choices)
        and isinstance(choices[0], dict)
        and choices[0].get("finish_reason") == "length"
    )
    return _extract_openai_stream_delta(event), False, cut_off


def _join_notes(notes: list[str | None]) -> str | None:
    merged = " ".join(note.strip() for note in notes if note and note.strip())
    return merged or None


def _extract_openai_stream_delta(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    delta = first.get("delta")
    if not isinstance(delta, dict):
        return ""
    content = delta.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text:
                out.append(text)
        return "".join(out)
    return ""


def _merge_info_note(primary: str | None, secondary: str | None) -> str | None:
    left = (primary or "").strip()
    right = (secondary or "").strip()
    if left and right:
        return f"{left} {right}"
    if left:
        return left
    if right:
        return right
    return None


def _fail(
    *,
    profile: LLMConnectionProfile,
    model: str | None,
    code: str,
    detail: str,
    info_note: str | None,
    output_text: str = "",
) -> LLMPostprocessResult:
    LOGGER.info(
        "llm.run.error provider=%s model=%s code=%s",
        profile.provider,
        model or "none",
        code,
    )
    return LLMPostprocessResult(
        status="fail",
        provider=profile.provider,
        model=model,
        output_text=output_text,
        error_code=code,
        error_detail=detail,
        info_note=info_note,
    )


class _LLMPostprocessException(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.partial_output: str = ""


def _raise_if_cancelled(run_control: LLMRunControl | None, partial_output: str) -> None:
    if run_control and run_control.is_cancelled():
        err = _LLMPostprocessException("cancelled", "Generation cancelled by user.")
        err.partial_output = (partial_output or "").strip()
        raise err
