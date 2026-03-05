"""Shared backend services for PyScribe frontends.

This module intentionally uses lazy exports so importing :mod:`services`
does not eagerly pull in heavy runtime dependencies (torch/pyannote/etc.).
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "BASE_MODEL_CHOICES": (".catalog_service", "BASE_MODEL_CHOICES"),
    "get_available_diarization_backends": (".catalog_service", "get_available_diarization_backends"),
    "get_backend_label": (".catalog_service", "get_backend_label"),
    "get_model_choices": (".catalog_service", "get_model_choices"),
    "AppConfig": (".config_service", "AppConfig"),
    "load_config": (".config_service", "load_config"),
    "save_config": (".config_service", "save_config"),
    "get_hf_token": (".hf_auth_service", "get_hf_token"),
    "save_hf_token": (".hf_auth_service", "save_hf_token"),
    "ensure_model_cached": (".model_download_service", "ensure_model_cached"),
    "estimate_model_download_size_bytes": (".model_download_service", "estimate_model_download_size_bytes"),
    "format_bytes": (".model_download_service", "format_bytes"),
    "is_model_cached": (".model_download_service", "is_model_cached"),
    "normalize_model_name": (".model_download_service", "normalize_model_name"),
    "resolve_repo_id": (".model_download_service", "resolve_repo_id"),
    "RuntimeInfo": (".model_service", "RuntimeInfo"),
    "detect_language": (".model_service", "detect_language"),
    "detect_runtime": (".model_service", "detect_runtime"),
    "load_model": (".model_service", "load_model"),
    "check_ocr_backend_ready": (".multimodal_service", "check_ocr_backend_ready"),
    "extract_text_from_images": (".multimodal_service", "extract_text_from_images"),
    "recommend_model": (".model_service", "recommend_model"),
    "open_folder": (".platform_service", "open_folder"),
    "ConnectionStageResult": (".llm_connection_service", "ConnectionStageResult"),
    "ConnectionTestResult": (".llm_connection_service", "ConnectionTestResult"),
    "DiscoveredLLMInstance": (".llm_connection_service", "DiscoveredLLMInstance"),
    "LLMConnectionProfile": (".llm_connection_service", "LLMConnectionProfile"),
    "LocalNetworkInfo": (".llm_connection_service", "LocalNetworkInfo"),
    "discover_local_networks": (".llm_connection_service", "discover_local_networks"),
    "get_enabled_llm_profiles": (".llm_connection_service", "get_enabled_llm_profiles"),
    "get_failure_suggestions": (".llm_connection_service", "get_failure_suggestions"),
    "load_llm_profiles": (".llm_connection_service", "load_llm_profiles"),
    "scan_lan_for_llm_instances": (".llm_connection_service", "scan_lan_for_llm_instances"),
    "test_connection": (".llm_connection_service", "test_connection"),
    "LLMPostprocessRequest": (".llm_postprocess_service", "LLMPostprocessRequest"),
    "LLMPreparedPayload": (".llm_postprocess_service", "LLMPreparedPayload"),
    "LLMPostprocessResult": (".llm_postprocess_service", "LLMPostprocessResult"),
    "build_llm_payload_preview": (".llm_postprocess_service", "build_llm_payload_preview"),
    "prepare_llm_postprocess_payload": (".llm_postprocess_service", "prepare_llm_postprocess_payload"),
    "run_llm_postprocess": (".llm_postprocess_service", "run_llm_postprocess"),
    "PromptTemplate": (".prompt_template_service", "PromptTemplate"),
    "create_user_prompt_template": (".prompt_template_service", "create_user_prompt_template"),
    "delete_user_prompt_template": (".prompt_template_service", "delete_user_prompt_template"),
    "get_default_prompt_template_id": (".prompt_template_service", "get_default_prompt_template_id"),
    "get_prompt_template": (".prompt_template_service", "get_prompt_template"),
    "load_prompt_templates": (".prompt_template_service", "load_prompt_templates"),
    "set_user_default_prompt_template": (".prompt_template_service", "set_user_default_prompt_template"),
    "update_user_prompt_template": (".prompt_template_service", "update_user_prompt_template"),
    "TranscriptionResult": (".transcription_service", "TranscriptionResult"),
    "transcribe_media_file": (".transcription_service", "transcribe_media_file"),
    "transcribe_prepared_audio": (".transcription_service", "transcribe_prepared_audio"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> object:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
