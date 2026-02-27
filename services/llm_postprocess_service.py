"""LLM post-processing execution service."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

from services.llm_connection_service import LLMConnectionProfile
from services.prompt_template_service import PromptTemplate


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMPostprocessRequest:
    transcript_text: str
    ocr_text: str
    notes_text: str
    selected_model: str | None


@dataclass(frozen=True)
class LLMPostprocessResult:
    status: str
    provider: str
    model: str | None
    output_text: str
    error_code: str | None
    error_detail: str | None


def run_llm_postprocess(
    profile: LLMConnectionProfile,
    template: PromptTemplate,
    request: LLMPostprocessRequest,
) -> LLMPostprocessResult:
    """Run LLM post-processing and return model output or error details."""
    transcript_text = (request.transcript_text or "").strip()
    if not transcript_text:
        return LLMPostprocessResult(
            status="fail",
            provider=profile.provider,
            model=request.selected_model or profile.default_model,
            output_text="",
            error_code="missing_transcript",
            error_detail="Transcript text is required for post-processing.",
        )

    model = (request.selected_model or profile.default_model or "").strip() or None
    if model is None:
        return LLMPostprocessResult(
            status="fail",
            provider=profile.provider,
            model=None,
            output_text="",
            error_code="missing_model",
            error_detail="No model is selected/configured for this profile.",
        )

    user_payload = _build_user_payload(template=template, request=request)
    LOGGER.info(
        "llm.run.start provider=%s scope=%s model=%s template_id=%s input_chars=%d",
        profile.provider,
        profile.scope,
        model,
        template.id,
        len(transcript_text),
    )
    if profile.provider == "ollama":
        return _run_ollama(profile=profile, template=template, model=model, user_payload=user_payload)
    return _run_openai_compatible(profile=profile, template=template, model=model, user_payload=user_payload)


def _run_ollama(
    *,
    profile: LLMConnectionProfile,
    template: PromptTemplate,
    model: str,
    user_payload: str,
) -> LLMPostprocessResult:
    url = f"{profile.base_url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "system": template.system_prompt,
        "prompt": user_payload,
        "stream": False,
    }
    try:
        result = _http_json_post(url=url, payload=payload, timeout_seconds=profile.timeout_seconds, headers={})
    except _LLMPostprocessException as exc:
        return _fail(profile=profile, model=model, code=exc.code, detail=str(exc))

    response_text = ""
    if isinstance(result, dict):
        response_text = str(result.get("response") or "").strip()
    if not response_text:
        return _fail(profile=profile, model=model, code="empty_response", detail="Model returned an empty response.")
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
    )


def _run_openai_compatible(
    *,
    profile: LLMConnectionProfile,
    template: PromptTemplate,
    model: str,
    user_payload: str,
) -> LLMPostprocessResult:
    url = f"{profile.base_url.rstrip('/')}/v1/chat/completions"
    headers = _auth_headers(profile.api_key)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": template.system_prompt},
            {"role": "user", "content": user_payload},
        ],
        "temperature": 0.2,
    }
    try:
        result = _http_json_post(url=url, payload=payload, timeout_seconds=profile.timeout_seconds, headers=headers)
    except _LLMPostprocessException as exc:
        return _fail(profile=profile, model=model, code=exc.code, detail=str(exc))

    response_text = _extract_openai_content(result)
    if not response_text:
        return _fail(profile=profile, model=model, code="empty_response", detail="Model returned an empty response.")
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
    )


def _build_user_payload(*, template: PromptTemplate, request: LLMPostprocessRequest) -> str:
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
    sections.append("")
    sections.append("Keep output concise and faithful to provided context.")
    return "\n".join(sections).strip()


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
    return ""


def _http_json_post(
    *,
    url: str,
    payload: dict[str, Any],
    timeout_seconds: float,
    headers: dict[str, str],
) -> Any:
    req_headers = {"Content-Type": "application/json"}
    req_headers.update(headers)
    data = json.dumps(payload).encode("utf-8")
    request = urlrequest.Request(url=url, method="POST", headers=req_headers, data=data)
    try:
        with urlrequest.urlopen(request, timeout=timeout_seconds) as response:
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


def _auth_headers(api_key: str | None) -> dict[str, str]:
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def _fail(*, profile: LLMConnectionProfile, model: str | None, code: str, detail: str) -> LLMPostprocessResult:
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
        output_text="",
        error_code=code,
        error_detail=detail,
    )


class _LLMPostprocessException(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
