"""LLM post-processing execution service."""

from __future__ import annotations

import base64
from collections.abc import Callable
from dataclasses import dataclass
import json
import logging
import mimetypes
import os
import ssl
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

from services.llm_connection_service import LLMConnectionProfile, evaluate_profile_scope_policy
from services.multimodal_service import extract_text_from_images
from services.prompt_template_service import PromptTemplate


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
)
OutputChunkCallback = Callable[[str], None]


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
    )


def run_llm_postprocess(
    profile: LLMConnectionProfile,
    template: PromptTemplate,
    request: LLMPostprocessRequest,
    *,
    prepared_payload: LLMPreparedPayload | None = None,
    on_output_chunk: OutputChunkCallback | None = None,
) -> LLMPostprocessResult:
    """Run LLM post-processing and return model output or error details."""
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
    user_payload = prepared.payload_text
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
    if profile.provider == "ollama":
        return _run_ollama(
            profile=profile,
            template=template,
            model=model,
            user_payload=user_payload,
            image_paths=image_paths,
            info_note=prepared.info_note,
            on_output_chunk=on_output_chunk,
        )
    return _run_openai_compatible(
        profile=profile,
        template=template,
        model=model,
        user_payload=user_payload,
        image_paths=image_paths,
        info_note=prepared.info_note,
        on_output_chunk=on_output_chunk,
    )


def _run_ollama(
    *,
    profile: LLMConnectionProfile,
    template: PromptTemplate,
    model: str,
    user_payload: str,
    image_paths: tuple[str, ...],
    info_note: str | None,
    on_output_chunk: OutputChunkCallback | None,
) -> LLMPostprocessResult:
    url = f"{profile.base_url.rstrip('/')}/api/generate"
    payload: dict[str, Any] = {
        "model": model,
        "system": template.system_prompt,
        "prompt": user_payload,
        "stream": bool(on_output_chunk),
    }
    if image_paths:
        payload["images"] = [_encode_image_bytes(path) for path in image_paths]
    response_text = ""
    retry_note: str | None = None
    try:
        if on_output_chunk:
            response_text, retry_note = _stream_ollama_with_timeout_retry(
                url=url,
                payload=payload,
                timeout_seconds=profile.timeout_seconds,
                headers={},
                verify_tls=profile.verify_tls,
                on_output_chunk=on_output_chunk,
            )
        else:
            result, retry_note = _post_json_with_timeout_retry(
                url=url,
                payload=payload,
                timeout_seconds=profile.timeout_seconds,
                headers={},
                verify_tls=profile.verify_tls,
            )
            if isinstance(result, dict):
                response_text = str(result.get("response") or "").strip()
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
            info_note=info_note,
            output_text=(exc.partial_output or "").strip(),
        )
    if not response_text:
        return _fail(profile=profile, model=model, code="empty_response", detail="Model returned an empty response.", info_note=info_note)
    LOGGER.info(
        "llm.run.complete provider=%s model=%s template_id=%s output_chars=%d",
        profile.provider,
        model,
        template.id,
        len(response_text),
    )
    return LLMPostprocessResult(
        status="pass",
        provider=profile.provider,
        model=model,
        output_text=response_text,
        error_code=None,
        error_detail=None,
        info_note=_merge_info_note(info_note, retry_note),
    )


def _run_openai_compatible(
    *,
    profile: LLMConnectionProfile,
    template: PromptTemplate,
    model: str,
    user_payload: str,
    image_paths: tuple[str, ...],
    info_note: str | None,
    on_output_chunk: OutputChunkCallback | None,
) -> LLMPostprocessResult:
    url = f"{profile.base_url.rstrip('/')}/v1/chat/completions"
    headers = _auth_headers(profile.api_key)
    user_content: Any = user_payload
    if image_paths:
        content_parts: list[dict[str, Any]] = [{"type": "text", "text": user_payload}]
        for image_path in image_paths:
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _encode_image_data_url(image_path)},
                }
            )
        user_content = content_parts
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": template.system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.2,
        "stream": bool(on_output_chunk),
    }
    response_text = ""
    retry_note: str | None = None
    try:
        if on_output_chunk:
            response_text, retry_note = _stream_openai_with_timeout_retry(
                url=url,
                payload=payload,
                timeout_seconds=profile.timeout_seconds,
                headers=headers,
                verify_tls=profile.verify_tls,
                on_output_chunk=on_output_chunk,
            )
        else:
            result, retry_note = _post_json_with_timeout_retry(
                url=url,
                payload=payload,
                timeout_seconds=profile.timeout_seconds,
                headers=headers,
                verify_tls=profile.verify_tls,
            )
            response_text = _extract_openai_content(result)
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
            info_note=info_note,
            output_text=(exc.partial_output or "").strip(),
        )
    if not response_text:
        return _fail(profile=profile, model=model, code="empty_response", detail="Model returned an empty response.", info_note=info_note)
    LOGGER.info(
        "llm.run.complete provider=%s model=%s template_id=%s output_chars=%d",
        profile.provider,
        model,
        template.id,
        len(response_text),
    )
    return LLMPostprocessResult(
        status="pass",
        provider=profile.provider,
        model=model,
        output_text=response_text,
        error_code=None,
        error_detail=None,
        info_note=_merge_info_note(info_note, retry_note),
    )


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


def _http_json_post(
    *,
    url: str,
    payload: dict[str, Any],
    timeout_seconds: float,
    headers: dict[str, str],
    verify_tls: bool,
) -> Any:
    req_headers = {"Content-Type": "application/json"}
    req_headers.update(headers)
    data = json.dumps(payload).encode("utf-8")
    request = urlrequest.Request(url=url, method="POST", headers=req_headers, data=data)
    try:
        with _urlopen_with_tls_policy(request=request, timeout_seconds=timeout_seconds, verify_tls=verify_tls) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urlerror.HTTPError as exc:
        code = exc.code
        if code in {401, 403}:
            raise _LLMPostprocessException("auth_failed", f"Authentication failed with HTTP {code}.") from exc
        if code == 404:
            raise _LLMPostprocessException("api_mismatch", "Endpoint path not found (HTTP 404).") from exc
        raise _LLMPostprocessException("server_error", f"Server returned HTTP {code}.") from exc
    except urlerror.URLError as exc:
        reason = str(exc.reason).lower()
        if "timed out" in reason:
            raise _LLMPostprocessException("timeout", "Connection timed out.") from exc
        if "name or service not known" in reason or "getaddrinfo" in reason:
            raise _LLMPostprocessException("dns_failure", "DNS resolution failed.") from exc
        if "connection refused" in reason or "failed to establish a new connection" in reason:
            raise _LLMPostprocessException("tcp_unreachable", "TCP connection was refused by endpoint.") from exc
        raise _LLMPostprocessException("tcp_unreachable", "Unable to reach endpoint over TCP.") from exc
    except TimeoutError as exc:
        raise _LLMPostprocessException("timeout", "Connection timed out.") from exc
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise _LLMPostprocessException("api_mismatch", "Endpoint did not return valid JSON.") from exc


def _stream_ollama_with_timeout_retry(
    *,
    url: str,
    payload: dict[str, Any],
    timeout_seconds: float,
    headers: dict[str, str],
    verify_tls: bool,
    on_output_chunk: OutputChunkCallback,
) -> tuple[str, str | None]:
    def _runner(current_timeout: float) -> str:
        return _stream_ollama_response(
            url=url,
            payload=payload,
            timeout_seconds=current_timeout,
            headers=headers,
            verify_tls=verify_tls,
            on_output_chunk=on_output_chunk,
        )

    return _stream_with_timeout_retry(
        run_stream=_runner,
        timeout_seconds=timeout_seconds,
        log_url=url,
    )


def _stream_openai_with_timeout_retry(
    *,
    url: str,
    payload: dict[str, Any],
    timeout_seconds: float,
    headers: dict[str, str],
    verify_tls: bool,
    on_output_chunk: OutputChunkCallback,
) -> tuple[str, str | None]:
    def _runner(current_timeout: float) -> str:
        return _stream_openai_response(
            url=url,
            payload=payload,
            timeout_seconds=current_timeout,
            headers=headers,
            verify_tls=verify_tls,
            on_output_chunk=on_output_chunk,
        )

    return _stream_with_timeout_retry(
        run_stream=_runner,
        timeout_seconds=timeout_seconds,
        log_url=url,
    )


def _stream_with_timeout_retry(
    *,
    run_stream: Callable[[float], str],
    timeout_seconds: float,
    log_url: str,
) -> tuple[str, str | None]:
    try:
        return run_stream(timeout_seconds), None
    except _LLMPostprocessException as exc:
        if exc.code != "timeout":
            raise
        if (exc.partial_output or "").strip():
            raise
        retry_timeout = max(_POSTPROCESS_TIMEOUT_RETRY_SECONDS, float(timeout_seconds) * 2.0)
        if retry_timeout <= float(timeout_seconds) + 0.1:
            raise
        LOGGER.info(
            "llm.run.timeout.retry provider_url=%s timeout=%.1f retry_timeout=%.1f",
            log_url,
            timeout_seconds,
            retry_timeout,
        )
        return run_stream(retry_timeout), f"Retried after timeout with {retry_timeout:.1f}s timeout."


def _stream_ollama_response(
    *,
    url: str,
    payload: dict[str, Any],
    timeout_seconds: float,
    headers: dict[str, str],
    verify_tls: bool,
    on_output_chunk: OutputChunkCallback,
) -> str:
    req_headers = {"Content-Type": "application/json"}
    req_headers.update(headers)
    data = json.dumps(payload).encode("utf-8")
    request = urlrequest.Request(url=url, method="POST", headers=req_headers, data=data)
    chunks: list[str] = []
    try:
        with _urlopen_with_tls_policy(request=request, timeout_seconds=timeout_seconds, verify_tls=verify_tls) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    part = json.loads(line)
                except json.JSONDecodeError:
                    continue
                piece = part.get("response")
                if isinstance(piece, str) and piece:
                    chunks.append(piece)
                    on_output_chunk(piece)
                if bool(part.get("done")):
                    break
    except urlerror.HTTPError as exc:
        code = exc.code
        if code in {401, 403}:
            err = _LLMPostprocessException("auth_failed", f"Authentication failed with HTTP {code}.")
            err.partial_output = "".join(chunks)
            raise err from exc
        if code == 404:
            err = _LLMPostprocessException("api_mismatch", "Endpoint path not found (HTTP 404).")
            err.partial_output = "".join(chunks)
            raise err from exc
        err = _LLMPostprocessException("server_error", f"Server returned HTTP {code}.")
        err.partial_output = "".join(chunks)
        raise err from exc
    except urlerror.URLError as exc:
        reason = str(exc.reason).lower()
        if "timed out" in reason:
            err = _LLMPostprocessException("timeout", "Connection timed out.")
            err.partial_output = "".join(chunks)
            raise err from exc
        if "name or service not known" in reason or "getaddrinfo" in reason:
            err = _LLMPostprocessException("dns_failure", "DNS resolution failed.")
            err.partial_output = "".join(chunks)
            raise err from exc
        if "connection refused" in reason or "failed to establish a new connection" in reason:
            err = _LLMPostprocessException("tcp_unreachable", "TCP connection was refused by endpoint.")
            err.partial_output = "".join(chunks)
            raise err from exc
        err = _LLMPostprocessException("tcp_unreachable", "Unable to reach endpoint over TCP.")
        err.partial_output = "".join(chunks)
        raise err from exc
    except TimeoutError as exc:
        err = _LLMPostprocessException("timeout", "Connection timed out.")
        err.partial_output = "".join(chunks)
        raise err from exc
    return "".join(chunks).strip()


def _stream_openai_response(
    *,
    url: str,
    payload: dict[str, Any],
    timeout_seconds: float,
    headers: dict[str, str],
    verify_tls: bool,
    on_output_chunk: OutputChunkCallback,
) -> str:
    req_headers = {"Content-Type": "application/json"}
    req_headers.update(headers)
    data = json.dumps(payload).encode("utf-8")
    request = urlrequest.Request(url=url, method="POST", headers=req_headers, data=data)
    chunks: list[str] = []
    try:
        with _urlopen_with_tls_policy(request=request, timeout_seconds=timeout_seconds, verify_tls=verify_tls) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data_part = line[5:].strip()
                if not data_part:
                    continue
                if data_part == "[DONE]":
                    break
                try:
                    event_payload = json.loads(data_part)
                except json.JSONDecodeError:
                    continue
                piece = _extract_openai_stream_delta(event_payload)
                if piece:
                    chunks.append(piece)
                    on_output_chunk(piece)
    except urlerror.HTTPError as exc:
        code = exc.code
        if code in {401, 403}:
            err = _LLMPostprocessException("auth_failed", f"Authentication failed with HTTP {code}.")
            err.partial_output = "".join(chunks)
            raise err from exc
        if code == 404:
            err = _LLMPostprocessException("api_mismatch", "Endpoint path not found (HTTP 404).")
            err.partial_output = "".join(chunks)
            raise err from exc
        err = _LLMPostprocessException("server_error", f"Server returned HTTP {code}.")
        err.partial_output = "".join(chunks)
        raise err from exc
    except urlerror.URLError as exc:
        reason = str(exc.reason).lower()
        if "timed out" in reason:
            err = _LLMPostprocessException("timeout", "Connection timed out.")
            err.partial_output = "".join(chunks)
            raise err from exc
        if "name or service not known" in reason or "getaddrinfo" in reason:
            err = _LLMPostprocessException("dns_failure", "DNS resolution failed.")
            err.partial_output = "".join(chunks)
            raise err from exc
        if "connection refused" in reason or "failed to establish a new connection" in reason:
            err = _LLMPostprocessException("tcp_unreachable", "TCP connection was refused by endpoint.")
            err.partial_output = "".join(chunks)
            raise err from exc
        err = _LLMPostprocessException("tcp_unreachable", "Unable to reach endpoint over TCP.")
        err.partial_output = "".join(chunks)
        raise err from exc
    except TimeoutError as exc:
        err = _LLMPostprocessException("timeout", "Connection timed out.")
        err.partial_output = "".join(chunks)
        raise err from exc
    return "".join(chunks).strip()


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


def _post_json_with_timeout_retry(
    *,
    url: str,
    payload: dict[str, Any],
    timeout_seconds: float,
    headers: dict[str, str],
    verify_tls: bool,
) -> tuple[Any, str | None]:
    try:
        return (
            _http_json_post(
                url=url,
                payload=payload,
                timeout_seconds=timeout_seconds,
                headers=headers,
                verify_tls=verify_tls,
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
        )
        return result, f"Retried after timeout with {retry_timeout:.1f}s timeout."


def _urlopen_with_tls_policy(
    *,
    request: urlrequest.Request,
    timeout_seconds: float,
    verify_tls: bool,
):
    if request.full_url.lower().startswith("https://"):
        if verify_tls:
            context = ssl.create_default_context()
        else:
            context = ssl._create_unverified_context()
        return urlrequest.urlopen(request, timeout=timeout_seconds, context=context)
    return urlrequest.urlopen(request, timeout=timeout_seconds)


def _auth_headers(api_key: str | None) -> dict[str, str]:
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


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
