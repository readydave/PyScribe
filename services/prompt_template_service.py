"""Prompt template loading and validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)
PROMPTS_ROOT = Path(__file__).resolve().parent.parent / "assets" / "prompts"
DEFAULT_PROMPT_INDEX_PATH = PROMPTS_ROOT / "index.yaml"
_ALLOWED_OUTPUT_FORMATS = {"markdown", "json"}


@dataclass(frozen=True)
class PromptTemplate:
    id: str
    name: str
    version: int
    description: str
    tags: tuple[str, ...]
    output_format: str
    enabled: bool
    built_in: bool
    system_prompt: str
    user_prompt_scaffold: str
    source_path: str


def load_prompt_templates(
    index_path: Path = DEFAULT_PROMPT_INDEX_PATH,
    *,
    include_disabled: bool = False,
) -> tuple[list[PromptTemplate], str | None]:
    """Load prompt templates from index and template files."""
    index = _load_yaml_mapping(index_path)
    if not index:
        return [], None

    templates: list[PromptTemplate] = []
    default_template_id = _as_optional_id(index.get("default_template_id"))
    seen_ids: set[str] = set()
    raw_templates = index.get("templates")
    if not isinstance(raw_templates, list):
        LOGGER.warning("Prompt index missing templates list: %s", index_path)
        return [], default_template_id

    for entry in raw_templates:
        if not isinstance(entry, dict):
            continue
        template_id = _as_optional_id(entry.get("id"))
        file_name = _as_optional_str(entry.get("file"))
        if not template_id or not file_name:
            continue
        if template_id in seen_ids:
            LOGGER.warning("Duplicate prompt template id '%s' in %s", template_id, index_path)
            continue
        seen_ids.add(template_id)

        built_in = bool(entry.get("built_in", True))
        entry_enabled = bool(entry.get("enabled", True))
        template_path = (index_path.parent / file_name).resolve()
        template_payload = _load_yaml_mapping(template_path)
        if not template_payload:
            continue
        template = _parse_template(
            template_id=template_id,
            payload=template_payload,
            source_path=template_path,
            built_in=built_in,
            enabled=entry_enabled,
        )
        if template is None:
            continue
        if include_disabled or template.enabled:
            templates.append(template)

    if default_template_id and not any(t.id == default_template_id for t in templates):
        LOGGER.warning("Default prompt template '%s' not found/enabled; using first available.", default_template_id)
        default_template_id = templates[0].id if templates else None

    if not default_template_id and templates:
        default_template_id = templates[0].id

    return templates, default_template_id


def get_prompt_template(
    template_id: str,
    index_path: Path = DEFAULT_PROMPT_INDEX_PATH,
    *,
    include_disabled: bool = False,
) -> PromptTemplate | None:
    """Return a specific prompt template by id, if present."""
    templates, _ = load_prompt_templates(index_path=index_path, include_disabled=include_disabled)
    target_id = _as_optional_id(template_id)
    if not target_id:
        return None
    for template in templates:
        if template.id == target_id:
            return template
    return None


def get_default_prompt_template_id(index_path: Path = DEFAULT_PROMPT_INDEX_PATH) -> str | None:
    """Return the effective default prompt template id."""
    _, default_template_id = load_prompt_templates(index_path=index_path, include_disabled=False)
    return default_template_id


def _parse_template(
    *,
    template_id: str,
    payload: dict[str, Any],
    source_path: Path,
    built_in: bool,
    enabled: bool,
) -> PromptTemplate | None:
    payload_id = _as_optional_id(payload.get("id")) or template_id
    if payload_id != template_id:
        LOGGER.warning(
            "Prompt template id mismatch in %s: index='%s', file='%s'",
            source_path,
            template_id,
            payload_id,
        )
        return None

    name = _as_optional_str(payload.get("name"))
    system_prompt = _as_optional_str(payload.get("system_prompt"))
    user_prompt_scaffold = _as_optional_str(payload.get("user_prompt_scaffold"))
    if not name or not system_prompt or not user_prompt_scaffold:
        LOGGER.warning("Prompt template missing required fields in %s", source_path)
        return None

    version = _as_positive_int(payload.get("version"), default=1)
    description = _as_optional_str(payload.get("description")) or ""
    output_format = _as_output_format(payload.get("output_format"))
    tags = _as_tags(payload.get("tags"))
    effective_enabled = enabled and bool(payload.get("enabled", True))
    return PromptTemplate(
        id=template_id,
        name=name,
        version=version,
        description=description,
        tags=tags,
        output_format=output_format,
        enabled=effective_enabled,
        built_in=built_in,
        system_prompt=system_prompt,
        user_prompt_scaffold=user_prompt_scaffold,
        source_path=str(source_path),
    )


def _load_yaml_mapping(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        LOGGER.warning("Prompt YAML not found: %s", path)
        return None
    try:
        import yaml  # local import keeps optional dependency local to this service
    except Exception as exc:  # pragma: no cover - dependency error path
        LOGGER.warning("Unable to import yaml package: %s", exc)
        return None
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning("Failed to parse YAML %s: %s", path, exc)
        return None
    if not isinstance(payload, dict):
        LOGGER.warning("YAML payload is not a mapping: %s", path)
        return None
    return payload


def _as_optional_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _as_optional_id(value: object) -> str | None:
    text = _as_optional_str(value)
    if not text:
        return None
    return text.lower()


def _as_positive_int(value: object, *, default: int) -> int:
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and value.isdigit() and int(value) > 0:
        return int(value)
    return default


def _as_output_format(value: object) -> str:
    normalized = _as_optional_str(value)
    if normalized and normalized.lower() in _ALLOWED_OUTPUT_FORMATS:
        return normalized.lower()
    return "markdown"


def _as_tags(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    tags: list[str] = []
    for item in value:
        text = _as_optional_str(item)
        if not text:
            continue
        normalized = text.lower()
        if normalized not in tags:
            tags.append(normalized)
    return tuple(tags)
