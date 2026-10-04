"""LLM connection profile parsing and diagnostics."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import ipaddress
import json
import logging
import os
import socket
import ssl
from typing import Any
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest


LOGGER = logging.getLogger(__name__)

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
_SUPPORTED_PROVIDERS = {"ollama", "openai_compatible", "lm_studio"}
_SUPPORTED_SCOPES = {"local", "lan"}
_DEFAULT_ALLOWED_CIDRS = ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
_SCAN_MAX_HOSTS_DEFAULT = 256


@dataclass(frozen=True)
class LocalNetworkInfo:
    interface_name: str
    ip_address: str
    network_cidr: str
    scan_cidr: str
    is_private: bool
    is_loopback: bool


@dataclass(frozen=True)
class DiscoveredLLMInstance:
    provider: str
    base_url: str
    host: str
    port: int
    detected_models: tuple[str, ...]
    scope_hint: str
    network_cidr: str
    detail: str


@dataclass(frozen=True)
class LLMConnectionProfile:
    name: str
    provider: str
    scope: str
    base_url: str
    api_key: str | None
    default_model: str | None
    timeout_seconds: float
    verify_tls: bool
    allowed_cidrs: tuple[str, ...]
    enabled: bool
    allow_concurrent_with_local_transcription: bool


@dataclass(frozen=True)
class ConnectionStageResult:
    stage: str
    status: str
    code: str | None
    detail: str
    suggestions: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConnectionTestResult:
    status: str
    provider: str
    base_url: str
    selected_model: str | None
    detected_models: tuple[str, ...]
    loaded_model: str | None
    failure_code: str | None
    failure_detail: str | None
    stages: tuple[ConnectionStageResult, ...]


def load_llm_profiles(raw_profiles: list[dict[str, object]]) -> list[LLMConnectionProfile]:
    """Normalize raw config profile dictionaries to typed profiles."""
    profiles: list[LLMConnectionProfile] = []
    for idx, raw in enumerate(raw_profiles):
        profile = _parse_profile(raw, idx=idx)
        if profile is not None:
            profiles.append(profile)
    return profiles


def get_enabled_llm_profiles(raw_profiles: list[dict[str, object]]) -> list[LLMConnectionProfile]:
    """Return enabled normalized profiles only."""
    return [profile for profile in load_llm_profiles(raw_profiles) if profile.enabled]


def discover_local_networks(*, include_non_private: bool = False, include_loopback: bool = False) -> list[LocalNetworkInfo]:
    """Detect local IPv4 interfaces and return normalized network details."""
    discovered: list[LocalNetworkInfo] = []
    seen_keys: set[tuple[str, str]] = set()

    try:
        import psutil  # type: ignore

        if_addrs = psutil.net_if_addrs()
    except Exception:
        if_addrs = {}

    if if_addrs:
        for interface_name, addresses in if_addrs.items():
            for addr in addresses:
                if getattr(addr, "family", None) != socket.AF_INET:
                    continue
                ip_text = str(getattr(addr, "address", "") or "").strip()
                mask_text = str(getattr(addr, "netmask", "") or "").strip()
                info = _build_local_network_info(interface_name=interface_name, ip_text=ip_text, mask_text=mask_text)
                if info is None:
                    continue
                if not include_loopback and info.is_loopback:
                    continue
                if not include_non_private and not info.is_private and not info.is_loopback:
                    continue
                key = (info.interface_name, info.ip_address)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                discovered.append(info)

    # Fallback when psutil is unavailable or returns no entries.
    if not discovered:
        host = socket.gethostname()
        for ip_text in _resolve_host_ips(host):
            if ip_text.version != 4:
                continue
            if not include_loopback and ip_text.is_loopback:
                continue
            if not include_non_private and not ip_text.is_private and not ip_text.is_loopback:
                continue
            info = _build_local_network_info(
                interface_name="host",
                ip_text=str(ip_text),
                mask_text="255.255.255.0",
            )
            if info is None:
                continue
            key = (info.interface_name, info.ip_address)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            discovered.append(info)

    discovered.sort(key=lambda item: (item.interface_name.lower(), item.ip_address))
    return discovered


def scan_lan_for_llm_instances(
    network_cidrs: list[str] | tuple[str, ...],
    *,
    include_ollama: bool = True,
    include_openai_compatible: bool = True,
    include_lm_studio: bool = True,
    timeout_seconds: float = 0.35,
    max_hosts_per_network: int = _SCAN_MAX_HOSTS_DEFAULT,
) -> list[DiscoveredLLMInstance]:
    """Scan selected LAN subnets for potential Ollama/LM Studio/OpenAI-compatible endpoints."""
    scan_targets: list[tuple[str, str, int, str]] = []
    cidr_by_host: dict[str, str] = {}
    for cidr in network_cidrs:
        network = _safe_parse_network(cidr)
        if network is None or network.version != 4:
            continue
        hosts = list(network.hosts())
        if len(hosts) > max_hosts_per_network:
            hosts = hosts[:max_hosts_per_network]
        for host_ip in hosts:
            host = str(host_ip)
            cidr_by_host[host] = str(network)
            if include_ollama:
                scan_targets.append(("ollama", host, 11434, "/api/tags"))
            if include_lm_studio:
                scan_targets.append(("lm_studio", host, 1234, "/v1/models"))
            if include_openai_compatible:
                if not include_lm_studio:
                    scan_targets.append(("openai_compatible", host, 1234, "/v1/models"))
                scan_targets.append(("openai_compatible", host, 8000, "/v1/models"))
                scan_targets.append(("openai_compatible", host, 8080, "/v1/models"))

    discovered: list[DiscoveredLLMInstance] = []
    seen_urls: set[tuple[str, str]] = set()
    if not scan_targets:
        return discovered

    with ThreadPoolExecutor(max_workers=64) as pool:
        futures = [
            pool.submit(
                _probe_scan_target,
                provider=provider,
                host=host,
                port=port,
                path=path,
                timeout_seconds=timeout_seconds,
                network_cidr=cidr_by_host.get(host, ""),
            )
            for provider, host, port, path in scan_targets
        ]
        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue
            key = (result.provider, result.base_url)
            if key in seen_urls:
                continue
            seen_urls.add(key)
            discovered.append(result)

    discovered.sort(key=lambda item: (item.provider, item.host, item.port))
    return discovered


def run_connection_test(profile: LLMConnectionProfile) -> ConnectionTestResult:
    """Run staged diagnostics for an LLM profile and return pass/fail detail."""
    LOGGER.info(
        "llm.test.start provider=%s scope=%s base_url=%s timeout=%.1f",
        profile.provider,
        profile.scope,
        profile.base_url,
        profile.timeout_seconds,
    )
    stages: list[ConnectionStageResult] = []

    parsed = _parse_base_url(profile.base_url)
    if parsed is None:
        return _fail_result(
            profile=profile,
            stages=stages,
            stage="validate_url",
            code="invalid_url",
            detail="Base URL must be an absolute http/https URL.",
        )
    stages.append(ConnectionStageResult(stage="validate_url", status="pass", code=None, detail="URL is valid."))
    policy_error = _check_scope_policy(profile=profile, parsed_url=parsed)
    if policy_error is not None:
        return _fail_result(
            profile=profile,
            stages=stages,
            stage="scope_policy",
            code=policy_error,
            detail=_failure_detail(policy_error),
        )
    stages.append(ConnectionStageResult(stage="scope_policy", status="pass", code=None, detail="Scope policy passed."))
    tls_policy_error = _check_tls_policy(profile=profile, parsed_url=parsed)
    if tls_policy_error is not None:
        return _fail_result(
            profile=profile,
            stages=stages,
            stage="tls_policy",
            code=tls_policy_error,
            detail=_failure_detail(tls_policy_error),
        )
    stages.append(ConnectionStageResult(stage="tls_policy", status="pass", code=None, detail="TLS policy passed."))

    if profile.provider == "ollama":
        return _test_ollama(profile=profile, stages=stages)
    if profile.provider == "lm_studio":
        return _test_openai_compatible(profile=profile, stages=stages)
    return _test_openai_compatible(profile=profile, stages=stages)


def get_failure_suggestions(code: str) -> tuple[str, ...]:
    """Return user-facing suggestions for a failure code."""
    mapping: dict[str, tuple[str, ...]] = {
        "invalid_url": (
            "Use a full URL like http://127.0.0.1:11434 or http://192.168.1.20:1234/v1.",
            "Check for typos and unsupported schemes.",
        ),
        "policy_blocked_non_local": (
            "Switch profile scope to LAN if this endpoint is not local.",
            "Use localhost/127.0.0.1 when scope is local.",
        ),
        "policy_blocked_non_lan": (
            "Use an IP in private LAN ranges (10.x, 172.16-31.x, 192.168.x).",
            "Update allowed CIDRs if your network uses a custom private range.",
            "If the endpoint is localhost/127.0.0.1, switch scope to local.",
        ),
        "policy_loopback_with_lan": (
            "LAN scope does not allow loopback addresses.",
            "Use scope local for localhost/127.0.0.1 endpoints.",
        ),
        "dns_failure": (
            "Verify hostname spelling and local DNS resolution.",
            "Try using a direct LAN IP address instead of hostname.",
        ),
        "policy_tls_verification_required": (
            "Disable TLS verification only for localhost/loopback development endpoints.",
            "For LAN HTTPS endpoints, trust the certificate instead of bypassing validation.",
            "Use plain http:// on private LAN only if that matches your deployment and risk tolerance.",
        ),
        "tcp_unreachable": (
            "Confirm the LLM server is running and listening on the configured port.",
            "Check firewall rules on the host and client machines.",
        ),
        "timeout": (
            "Increase connection timeout in profile settings.",
            "Check LAN latency and endpoint responsiveness.",
        ),
        "tls_error": (
            "Use http:// for local/LAN endpoints unless TLS is configured.",
            "If using TLS, verify certificate trust and hostname.",
        ),
        "auth_failed": (
            "Check API key/token and auth header configuration.",
            "Confirm the server accepts your configured auth mode.",
        ),
        "api_mismatch": (
            "Verify provider type matches endpoint API shape.",
            "For LM Studio/OpenAI-compatible endpoints, include /v1 path when required.",
        ),
        "no_models_available": (
            "Load or pull at least one model in the LLM server.",
            "Refresh models list after loading a model.",
        ),
        "model_not_found": (
            "Choose a model from detected list or load the configured default model.",
            "Ensure model names match exactly (including tags).",
        ),
        "inference_failed": (
            "Try a smaller model or reduce server load.",
            "Check server logs for model runtime errors.",
        ),
    }
    return mapping.get(code, ("Check endpoint settings and server logs.",))


def _test_ollama(profile: LLMConnectionProfile, stages: list[ConnectionStageResult]) -> ConnectionTestResult:
    tags_url = _join_url(profile.base_url, "/api/tags")
    try:
        tags_payload = _http_json_get(
            tags_url,
            timeout_seconds=profile.timeout_seconds,
            verify_tls=profile.verify_tls,
        )
    except _ConnectionException as exc:
        return _fail_result(profile=profile, stages=stages, stage="reachability", code=exc.code, detail=str(exc))

    stages.append(ConnectionStageResult(stage="reachability", status="pass", code=None, detail="Endpoint reachable."))
    if not isinstance(tags_payload, dict):
        return _fail_result(
            profile=profile,
            stages=stages,
            stage="model_discovery",
            code="api_mismatch",
            detail="Unexpected response shape for Ollama /api/tags.",
        )

    models = _extract_ollama_models(tags_payload)
    if not models:
        return _fail_result(
            profile=profile,
            stages=stages,
            stage="model_discovery",
            code="no_models_available",
            detail="No models were returned by Ollama.",
        )
    stages.append(
        ConnectionStageResult(
            stage="model_discovery",
            status="pass",
            code=None,
            detail=f"Detected {len(models)} model(s).",
        )
    )

    selected_model = profile.default_model or models[0]
    if selected_model not in models:
        return _fail_result(
            profile=profile,
            stages=stages,
            stage="model_selection",
            code="model_not_found",
            detail=f"Configured model '{selected_model}' was not found on endpoint.",
            detected_models=tuple(models),
        )

    loaded_model = _get_ollama_loaded_model(profile)
    smoke_url = _join_url(profile.base_url, "/api/generate")
    payload = {"model": selected_model, "prompt": "Connection test. Reply with: OK", "stream": False}
    try:
        smoke_payload = _http_json_post(
            smoke_url,
            payload=payload,
            timeout_seconds=profile.timeout_seconds,
            verify_tls=profile.verify_tls,
        )
    except _ConnectionException as exc:
        return _fail_result(
            profile=profile,
            stages=stages,
            stage="inference_smoke",
            code=exc.code,
            detail=str(exc),
            selected_model=selected_model,
            detected_models=tuple(models),
            loaded_model=loaded_model,
        )
    if not isinstance(smoke_payload, dict):
        return _fail_result(
            profile=profile,
            stages=stages,
            stage="inference_smoke",
            code="api_mismatch",
            detail="Unexpected response shape for Ollama /api/generate.",
            selected_model=selected_model,
            detected_models=tuple(models),
            loaded_model=loaded_model,
        )

    stages.append(
        ConnectionStageResult(
            stage="inference_smoke",
            status="pass",
            code=None,
            detail=f"Smoke inference succeeded using model '{selected_model}'.",
        )
    )
    result = ConnectionTestResult(
        status="pass",
        provider=profile.provider,
        base_url=profile.base_url,
        selected_model=selected_model,
        detected_models=tuple(models),
        loaded_model=loaded_model,
        failure_code=None,
        failure_detail=None,
        stages=tuple(stages),
    )
    LOGGER.info(
        "llm.test.result status=pass provider=%s base_url=%s models=%d selected=%s loaded=%s",
        profile.provider,
        profile.base_url,
        len(models),
        selected_model,
        loaded_model or "unknown",
    )
    return result


def _test_openai_compatible(profile: LLMConnectionProfile, stages: list[ConnectionStageResult]) -> ConnectionTestResult:
    models_url = _join_url(profile.base_url, "/v1/models")
    headers = _auth_headers(profile.api_key)
    try:
        models_payload = _http_json_get(
            models_url,
            timeout_seconds=profile.timeout_seconds,
            headers=headers,
            verify_tls=profile.verify_tls,
        )
    except _ConnectionException as exc:
        return _fail_result(profile=profile, stages=stages, stage="reachability", code=exc.code, detail=str(exc))

    stages.append(ConnectionStageResult(stage="reachability", status="pass", code=None, detail="Endpoint reachable."))
    models = _extract_openai_models(models_payload)
    if not models:
        return _fail_result(
            profile=profile,
            stages=stages,
            stage="model_discovery",
            code="no_models_available",
            detail="No models were returned by the OpenAI-compatible endpoint.",
        )
    stages.append(
        ConnectionStageResult(
            stage="model_discovery",
            status="pass",
            code=None,
            detail=f"Detected {len(models)} model(s).",
        )
    )

    selected_model = profile.default_model or models[0]
    if selected_model not in models:
        return _fail_result(
            profile=profile,
            stages=stages,
            stage="model_selection",
            code="model_not_found",
            detail=f"Configured model '{selected_model}' was not found on endpoint.",
            detected_models=tuple(models),
        )

    completions_url = _join_url(profile.base_url, "/v1/chat/completions")
    payload = {
        "model": selected_model,
        "messages": [{"role": "user", "content": "Connection test. Reply with: OK"}],
        "temperature": 0,
        "max_tokens": 8,
    }
    try:
        smoke_payload = _http_json_post(
            completions_url,
            payload=payload,
            timeout_seconds=profile.timeout_seconds,
            headers=headers,
            verify_tls=profile.verify_tls,
        )
    except _ConnectionException as exc:
        return _fail_result(
            profile=profile,
            stages=stages,
            stage="inference_smoke",
            code=exc.code,
            detail=str(exc),
            selected_model=selected_model,
            detected_models=tuple(models),
        )
    if not isinstance(smoke_payload, dict):
        return _fail_result(
            profile=profile,
            stages=stages,
            stage="inference_smoke",
            code="api_mismatch",
            detail="Unexpected response shape for /v1/chat/completions.",
            selected_model=selected_model,
            detected_models=tuple(models),
        )

    stages.append(
        ConnectionStageResult(
            stage="inference_smoke",
            status="pass",
            code=None,
            detail=f"Smoke inference succeeded using model '{selected_model}'.",
        )
    )
    result = ConnectionTestResult(
        status="pass",
        provider=profile.provider,
        base_url=profile.base_url,
        selected_model=selected_model,
        detected_models=tuple(models),
        loaded_model=None,
        failure_code=None,
        failure_detail=None,
        stages=tuple(stages),
    )
    LOGGER.info(
        "llm.test.result status=pass provider=%s base_url=%s models=%d selected=%s loaded=unknown",
        profile.provider,
        profile.base_url,
        len(models),
        selected_model,
    )
    return result


def _get_ollama_loaded_model(profile: LLMConnectionProfile) -> str | None:
    ps_url = _join_url(profile.base_url, "/api/ps")
    try:
        payload = _http_json_get(
            ps_url,
            timeout_seconds=profile.timeout_seconds,
            verify_tls=profile.verify_tls,
        )
    except _ConnectionException:
        return None
    if not isinstance(payload, dict):
        return None
    raw_models = payload.get("models")
    if not isinstance(raw_models, list):
        return None
    for item in raw_models:
        if isinstance(item, dict):
            candidate = item.get("model") or item.get("name")
            text = _as_optional_str(candidate)
            if text:
                return text
    return None


def _fail_result(
    *,
    profile: LLMConnectionProfile,
    stages: list[ConnectionStageResult],
    stage: str,
    code: str,
    detail: str,
    selected_model: str | None = None,
    detected_models: tuple[str, ...] = (),
    loaded_model: str | None = None,
) -> ConnectionTestResult:
    stage_result = ConnectionStageResult(
        stage=stage,
        status="fail",
        code=code,
        detail=detail,
        suggestions=get_failure_suggestions(code),
    )
    stages.append(stage_result)
    LOGGER.info(
        "llm.test.result status=fail provider=%s base_url=%s code=%s stage=%s",
        profile.provider,
        profile.base_url,
        code,
        stage,
    )
    return ConnectionTestResult(
        status="fail",
        provider=profile.provider,
        base_url=profile.base_url,
        selected_model=selected_model,
        detected_models=detected_models,
        loaded_model=loaded_model,
        failure_code=code,
        failure_detail=detail,
        stages=tuple(stages),
    )


def _parse_profile(raw: dict[str, object], *, idx: int) -> LLMConnectionProfile | None:
    name = _as_optional_str(raw.get("name")) or f"profile-{idx + 1}"
    provider = (_as_optional_str(raw.get("provider")) or "ollama").lower()
    scope = (_as_optional_str(raw.get("scope")) or "local").lower()
    base_url = _as_optional_str(raw.get("base_url")) or ""
    if provider not in _SUPPORTED_PROVIDERS:
        LOGGER.warning("Skipping unknown LLM provider '%s' for profile '%s'", provider, name)
        return None
    if scope not in _SUPPORTED_SCOPES:
        LOGGER.warning("Skipping unknown LLM scope '%s' for profile '%s'", scope, name)
        return None
    if not base_url:
        LOGGER.warning("Skipping profile '%s' with empty base_url", name)
        return None
    base_url = _normalize_base_url_for_profile(provider, base_url)
    timeout_seconds = _as_float(raw.get("timeout_seconds"), default=8.0, min_value=1.0, max_value=120.0)
    return LLMConnectionProfile(
        name=name,
        provider=provider,
        scope=scope,
        base_url=base_url.rstrip("/"),
        api_key=_resolve_profile_api_key(raw),
        default_model=_as_optional_str(raw.get("default_model")),
        timeout_seconds=timeout_seconds,
        verify_tls=_as_bool(raw.get("verify_tls"), default=True),
        allowed_cidrs=_as_cidr_tuple(raw.get("allowed_cidrs"), default=_DEFAULT_ALLOWED_CIDRS),
        enabled=_as_bool(raw.get("enabled"), default=True),
        allow_concurrent_with_local_transcription=_as_bool(
            raw.get("allow_concurrent_with_local_transcription"),
            default=(scope == "lan"),
        ),
    )


def evaluate_profile_scope_policy(profile: LLMConnectionProfile) -> tuple[bool, str | None, str | None]:
    """
    Validate base URL shape and scope policy for runtime use (without network calls).
    """
    parsed = _parse_base_url(profile.base_url)
    if parsed is None:
        return False, "invalid_url", "Base URL must be an absolute http/https URL."
    policy_error = _check_scope_policy(profile=profile, parsed_url=parsed)
    if policy_error is not None:
        return False, policy_error, _failure_detail(policy_error)
    tls_policy_error = _check_tls_policy(profile=profile, parsed_url=parsed)
    if tls_policy_error is not None:
        return False, tls_policy_error, _failure_detail(tls_policy_error)
    return True, None, None


def _normalize_base_url_for_profile(provider: str, base_url: str) -> str:
    """
    Defensive wrapper around URL normalization used during profile parsing.

    Some stale/hot-reload runtimes can surface NameError for helper symbols.
    Keep parsing resilient so the UI can continue and report validation errors
    instead of crashing.
    """
    normalizer = globals().get("_normalize_base_url")
    if callable(normalizer):
        return normalizer(provider, base_url)

    LOGGER.warning("LLM base URL normalizer helper missing at runtime; using fallback parser.")
    text = str(base_url or "").strip()
    try:
        parsed = urlparse.urlparse(text)
    except Exception:
        return text
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return text
    if provider in {"openai_compatible", "lm_studio"} and parsed.path.rstrip("/") == "/v1":
        return urlparse.urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))
    return text


def _check_scope_policy(*, profile: LLMConnectionProfile, parsed_url: urlparse.ParseResult) -> str | None:
    host = (parsed_url.hostname or "").strip().lower()
    if not host:
        return "invalid_url"
    try:
        ip_obj = ipaddress.ip_address(host)
    except ValueError:
        ip_obj = None

    if profile.scope == "local":
        if host in _LOCAL_HOSTS:
            return None
        if ip_obj is not None and ip_obj.is_loopback:
            return None
        return "policy_blocked_non_local"

    # LAN scope
    if host in _LOCAL_HOSTS:
        return "policy_loopback_with_lan"
    if ip_obj is not None and ip_obj.is_loopback:
        return "policy_loopback_with_lan"
    candidate_ips = _resolve_host_ips(host)
    if not candidate_ips:
        # If DNS fails we treat as DNS issue rather than policy block.
        return "dns_failure"
    cidrs = _parse_allowed_networks(profile.allowed_cidrs)
    for candidate in candidate_ips:
        if candidate.is_loopback:
            continue
        if candidate.is_private and _ip_in_any_network(candidate, cidrs):
            return None
    return "policy_blocked_non_lan"


def _check_tls_policy(*, profile: LLMConnectionProfile, parsed_url: urlparse.ParseResult) -> str | None:
    if profile.verify_tls or parsed_url.scheme.lower() != "https":
        return None
    host = (parsed_url.hostname or "").strip().lower()
    if not host:
        return "invalid_url"
    if host in _LOCAL_HOSTS:
        return None
    try:
        ip_obj = ipaddress.ip_address(host)
    except ValueError:
        ip_obj = None
    if ip_obj is not None and ip_obj.is_loopback:
        return None
    candidate_ips = _resolve_host_ips(host)
    if any(candidate.is_loopback for candidate in candidate_ips):
        return None
    return "policy_tls_verification_required"


def _resolve_host_ips(host: str) -> list[ipaddress._BaseAddress]:
    try:
        import socket

        infos = socket.getaddrinfo(host, None)
    except Exception:
        return []
    resolved: list[ipaddress._BaseAddress] = []
    for info in infos:
        try:
            addr = info[4][0]
            ip_obj = ipaddress.ip_address(addr)
        except Exception:
            continue
        if ip_obj not in resolved:
            resolved.append(ip_obj)
    return resolved


def _build_local_network_info(*, interface_name: str, ip_text: str, mask_text: str) -> LocalNetworkInfo | None:
    if not ip_text:
        return None
    try:
        ip_obj = ipaddress.ip_address(ip_text)
    except ValueError:
        return None
    if ip_obj.version != 4:
        return None
    if not mask_text:
        mask_text = "255.255.255.0"
    try:
        interface = ipaddress.ip_interface(f"{ip_text}/{mask_text}")
    except ValueError:
        try:
            interface = ipaddress.ip_interface(f"{ip_text}/24")
        except ValueError:
            return None
    network = interface.network
    scan_network = network
    if isinstance(scan_network, ipaddress.IPv4Network) and scan_network.prefixlen < 24:
        try:
            scan_network = ipaddress.ip_network(f"{ip_text}/24", strict=False)
        except ValueError:
            scan_network = network
    return LocalNetworkInfo(
        interface_name=interface_name,
        ip_address=str(interface.ip),
        network_cidr=str(network),
        scan_cidr=str(scan_network),
        is_private=interface.ip.is_private,
        is_loopback=interface.ip.is_loopback,
    )


def _safe_parse_network(cidr: str) -> ipaddress._BaseNetwork | None:
    text = str(cidr or "").strip()
    if not text:
        return None
    try:
        return ipaddress.ip_network(text, strict=False)
    except ValueError:
        return None


def _probe_scan_target(
    *,
    provider: str,
    host: str,
    port: int,
    path: str,
    timeout_seconds: float,
    network_cidr: str,
) -> DiscoveredLLMInstance | None:
    base_url = f"http://{host}:{port}"
    probe_url = _join_url(base_url, path)
    try:
        payload = _http_json_get(probe_url, timeout_seconds=max(0.1, timeout_seconds), headers={})
    except Exception:
        return None

    detected_models: list[str] = []
    detail = ""
    if provider == "ollama":
        detected_models = _extract_ollama_models(payload)
        if not detected_models and not isinstance(payload, dict):
            return None
        detail = f"Ollama endpoint responded at {base_url}."
    else:
        detected_models = _extract_openai_models(payload)
        if not detected_models and not isinstance(payload, dict):
            return None
        if provider == "lm_studio":
            detail = f"LM Studio-compatible endpoint responded at {base_url}/v1."
        else:
            detail = f"OpenAI-compatible endpoint responded at {base_url}/v1."

    scope_hint = "local" if host in _LOCAL_HOSTS else "lan"
    return DiscoveredLLMInstance(
        provider=provider,
        base_url=base_url,
        host=host,
        port=port,
        detected_models=tuple(detected_models),
        scope_hint=scope_hint,
        network_cidr=network_cidr,
        detail=detail,
    )


def _parse_allowed_networks(cidrs: tuple[str, ...]) -> tuple[ipaddress._BaseNetwork, ...]:
    nets: list[ipaddress._BaseNetwork] = []
    for cidr in cidrs:
        try:
            nets.append(ipaddress.ip_network(cidr, strict=False))
        except ValueError:
            continue
    return tuple(nets)


def _ip_in_any_network(ip_obj: ipaddress._BaseAddress, networks: tuple[ipaddress._BaseNetwork, ...]) -> bool:
    if not networks:
        return ip_obj.is_private
    return any(ip_obj in net for net in networks)


def _parse_base_url(value: str) -> urlparse.ParseResult | None:
    try:
        parsed = urlparse.urlparse(value)
    except Exception:
        return None
    if parsed.scheme not in {"http", "https"}:
        return None
    if not parsed.netloc:
        return None
    return parsed


def _normalize_base_url(provider: str, base_url: str) -> str:
    text = str(base_url or "").strip()
    parsed = _parse_base_url(text)
    if parsed is None:
        return text
    if provider in {"openai_compatible", "lm_studio"} and parsed.path.rstrip("/") == "/v1":
        rebuilt = urlparse.urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))
        return rebuilt
    return text


def _join_url(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    suffix = path if path.startswith("/") else f"/{path}"
    return f"{base}{suffix}"


def _http_json_get(
    url: str,
    *,
    timeout_seconds: float,
    headers: dict[str, str] | None = None,
    verify_tls: bool = True,
) -> Any:
    request = urlrequest.Request(url=url, method="GET", headers=headers or {})
    return _http_json_request(request=request, timeout_seconds=timeout_seconds, verify_tls=verify_tls)


def _http_json_post(
    url: str,
    *,
    payload: dict[str, Any],
    timeout_seconds: float,
    headers: dict[str, str] | None = None,
    verify_tls: bool = True,
) -> Any:
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    data = json.dumps(payload).encode("utf-8")
    request = urlrequest.Request(url=url, method="POST", headers=req_headers, data=data)
    return _http_json_request(request=request, timeout_seconds=timeout_seconds, verify_tls=verify_tls)


def _http_json_request(*, request: urlrequest.Request, timeout_seconds: float, verify_tls: bool) -> Any:
    context: ssl.SSLContext | None = None
    if request.full_url.lower().startswith("https://"):
        if verify_tls:
            context = ssl.create_default_context()
        else:
            context = ssl._create_unverified_context()
    try:
        if context is None:
            response_handle = urlrequest.urlopen(request, timeout=timeout_seconds)
        else:
            response_handle = urlrequest.urlopen(request, timeout=timeout_seconds, context=context)
        with response_handle as response:
            body = response.read().decode("utf-8", errors="replace")
    except urlerror.HTTPError as exc:
        code = exc.code
        if code in {401, 403}:
            raise _ConnectionException("auth_failed", f"Authentication failed with HTTP {code}.") from exc
        if code == 404:
            raise _ConnectionException("api_mismatch", "Endpoint path not found (HTTP 404).") from exc
        if 400 <= code < 500:
            raise _ConnectionException("api_mismatch", f"Unexpected client error HTTP {code}.") from exc
        raise _ConnectionException("inference_failed", f"Server returned HTTP {code}.") from exc
    except urlerror.URLError as exc:
        reason = str(exc.reason).lower()
        if "timed out" in reason:
            raise _ConnectionException("timeout", "Connection timed out.") from exc
        if "certificate" in reason or "ssl" in reason:
            raise _ConnectionException("tls_error", "TLS/certificate validation failed.") from exc
        if "name or service not known" in reason or "getaddrinfo" in reason:
            raise _ConnectionException("dns_failure", "DNS resolution failed.") from exc
        if "connection refused" in reason or "failed to establish a new connection" in reason:
            raise _ConnectionException("tcp_unreachable", "TCP connection was refused by endpoint.") from exc
        raise _ConnectionException("tcp_unreachable", "Unable to reach endpoint over TCP.") from exc
    except TimeoutError as exc:
        raise _ConnectionException("timeout", "Connection timed out.") from exc
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise _ConnectionException("api_mismatch", "Endpoint did not return valid JSON.") from exc


def _extract_ollama_models(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    raw_models = payload.get("models")
    if not isinstance(raw_models, list):
        return []
    models: list[str] = []
    for item in raw_models:
        if not isinstance(item, dict):
            continue
        candidate = item.get("model") or item.get("name")
        text = _as_optional_str(candidate)
        if text and text not in models:
            models.append(text)
    return models


def _extract_openai_models(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    raw_models = payload.get("data")
    if not isinstance(raw_models, list):
        return []
    models: list[str] = []
    for item in raw_models:
        if not isinstance(item, dict):
            continue
        candidate = _as_optional_str(item.get("id"))
        if candidate and candidate not in models:
            models.append(candidate)
    return models


def _auth_headers(api_key: str | None) -> dict[str, str]:
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def _resolve_profile_api_key(raw: dict[str, object]) -> str | None:
    runtime_key = _as_optional_str(raw.get("api_key_runtime"))
    if runtime_key:
        return runtime_key
    api_key = _as_optional_str(raw.get("api_key"))
    if api_key and api_key.lower().startswith("env:"):
        env_name = api_key[4:].strip()
        if not env_name:
            return None
        env_value = os.environ.get(env_name)
        return _as_optional_str(env_value)
    env_name = _as_optional_str(raw.get("api_key_env"))
    if env_name:
        env_value = os.environ.get(env_name)
        return _as_optional_str(env_value)
    return api_key


def _failure_detail(code: str) -> str:
    details = {
        "invalid_url": "Base URL must be an absolute http/https URL.",
        "policy_blocked_non_local": "Local scope allows only localhost/loopback endpoints.",
        "policy_blocked_non_lan": "LAN scope allows only private network endpoints in allowed CIDRs.",
        "policy_loopback_with_lan": "LAN scope cannot use localhost/loopback endpoints. Use local scope instead.",
        "policy_tls_verification_required": (
            "HTTPS certificate verification can only be bypassed for localhost/loopback development endpoints."
        ),
        "dns_failure": "Host DNS resolution failed for endpoint.",
    }
    return details.get(code, "Connection test failed.")


def _as_optional_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _as_bool(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = _as_optional_str(value)
    if not text:
        return default
    normalized = text.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _as_float(value: object, *, default: float, min_value: float, max_value: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, parsed))


def _as_cidr_tuple(value: object, *, default: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(value, list):
        return default
    normalized: list[str] = []
    for item in value:
        text = _as_optional_str(item)
        if text and text not in normalized:
            normalized.append(text)
    return tuple(normalized or default)


class _ConnectionException(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
