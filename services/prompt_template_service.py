"""Prompt template loading, validation, and user-template CRUD utilities."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Any


LOGGER = logging.getLogger(__name__)
BUILTIN_PROMPTS_ROOT = Path(__file__).resolve().parent.parent / "assets" / "prompts"
DEFAULT_PROMPT_INDEX_PATH = BUILTIN_PROMPTS_ROOT / "index.yaml"
USER_PROMPTS_ROOT = Path.home() / ".pyscribe" / "prompts"
USER_PROMPT_INDEX_PATH = USER_PROMPTS_ROOT / "index.yaml"
_ALLOWED_OUTPUT_FORMATS = {"markdown", "json"}
_TEMPLATE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{1,63}$")


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
    include_user_templates: bool = True,
    user_index_path: Path = USER_PROMPT_INDEX_PATH,
) -> tuple[list[PromptTemplate], str | None]:
    """Load built-in templates and optional user templates."""
    builtins, built_in_default = _load_templates_from_index(
        index_path=index_path,
        include_disabled=include_disabled,
        warn_if_missing=True,
    )
    templates: list[PromptTemplate] = list(builtins)
    default_template_id = built_in_default
    seen_ids = {template.id for template in templates}

    if include_user_templates:
        user_templates, user_default = _load_templates_from_index(
            index_path=user_index_path,
            include_disabled=include_disabled,
            warn_if_missing=False,
        )
        for template in user_templates:
            if template.id in seen_ids:
                LOGGER.warning("Skipping duplicate template id '%s' from user templates.", template.id)
                continue
            seen_ids.add(template.id)
            templates.append(template)
        if user_default and any(template.id == user_default for template in templates):
            default_template_id = user_default

    if default_template_id and not any(template.id == default_template_id for template in templates):
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
    include_user_templates: bool = True,
    user_index_path: Path = USER_PROMPT_INDEX_PATH,
) -> PromptTemplate | None:
    """Return a specific prompt template by id, if present."""
    templates, _ = load_prompt_templates(
        index_path=index_path,
        include_disabled=include_disabled,
        include_user_templates=include_user_templates,
        user_index_path=user_index_path,
    )
    target_id = _as_optional_id(template_id)
    if not target_id:
        return None
    for template in templates:
        if template.id == target_id:
            return template
    return None


def get_default_prompt_template_id(
    index_path: Path = DEFAULT_PROMPT_INDEX_PATH,
    *,
    include_user_templates: bool = True,
    user_index_path: Path = USER_PROMPT_INDEX_PATH,
) -> str | None:
    """Return the effective default prompt template id."""
    _, default_template_id = load_prompt_templates(
        index_path=index_path,
        include_disabled=False,
        include_user_templates=include_user_templates,
        user_index_path=user_index_path,
    )
    return default_template_id


def create_user_prompt_template(
    *,
    name: str,
    description: str,
    tags: list[str] | tuple[str, ...],
    output_format: str,
    system_prompt: str,
    user_prompt_scaffold: str,
    enabled: bool = True,
    template_id: str | None = None,
    user_index_path: Path = USER_PROMPT_INDEX_PATH,
    built_in_index_path: Path = DEFAULT_PROMPT_INDEX_PATH,
) -> PromptTemplate:
    """Create a user prompt template and return the normalized template."""
    base_id = _normalize_or_slug_id(template_id, name=name)
    built_in_ids = {template.id for template in _load_templates_from_index(
        index_path=built_in_index_path,
        include_disabled=True,
        warn_if_missing=True,
    )[0]}
    index_payload = _load_or_init_user_index(user_index_path)
    existing_ids = _collect_template_ids(index_payload)
    template_id_value = _make_unique_id(base_id, blocked_ids=built_in_ids | existing_ids)

    file_name = f"templates/{template_id_value}.yaml"
    template_path = user_index_path.parent / file_name
    template_payload = _build_template_payload(
        template_id=template_id_value,
        name=name,
        description=description,
        tags=tags,
        output_format=output_format,
        system_prompt=system_prompt,
        user_prompt_scaffold=user_prompt_scaffold,
        enabled=enabled,
        version=1,
    )
    _write_yaml_mapping(template_path, template_payload)

    raw_templates = index_payload.get("templates")
    if not isinstance(raw_templates, list):
        raw_templates = []
    raw_templates.append(
        {
            "id": template_id_value,
            "file": file_name,
            "built_in": False,
            "enabled": bool(enabled),
        }
    )
    index_payload["templates"] = raw_templates
    if not _as_optional_id(index_payload.get("default_template_id")):
        index_payload["default_template_id"] = template_id_value
    _write_yaml_mapping(user_index_path, index_payload)

    created = get_prompt_template(
        template_id_value,
        include_disabled=True,
        include_user_templates=True,
        user_index_path=user_index_path,
    )
    if created is None:
        raise ValueError(f"Failed to load created template '{template_id_value}'.")
    return created


def update_user_prompt_template(
    *,
    template_id: str,
    name: str,
    description: str,
    tags: list[str] | tuple[str, ...],
    output_format: str,
    system_prompt: str,
    user_prompt_scaffold: str,
    enabled: bool = True,
    user_index_path: Path = USER_PROMPT_INDEX_PATH,
) -> PromptTemplate:
    """Update an existing user template and return the normalized template."""
    target_id = _as_optional_id(template_id)
    if not target_id:
        raise ValueError("Template id is required.")

    index_payload = _load_or_init_user_index(user_index_path)
    raw_templates = index_payload.get("templates")
    if not isinstance(raw_templates, list):
        raw_templates = []

    entry: dict[str, Any] | None = None
    for item in raw_templates:
        if not isinstance(item, dict):
            continue
        if _as_optional_id(item.get("id")) == target_id:
            entry = item
            break
    if entry is None:
        raise ValueError(f"User template '{target_id}' was not found.")

    file_name = _as_optional_str(entry.get("file")) or f"templates/{target_id}.yaml"
    template_path = _resolve_index_relative_path(
        index_path=user_index_path,
        relative_path=file_name,
        require_within_root=True,
    )
    existing_payload = _load_yaml_mapping(template_path, warn_if_missing=False) or {}
    next_version = _as_positive_int(existing_payload.get("version"), default=1) + 1
    template_payload = _build_template_payload(
        template_id=target_id,
        name=name,
        description=description,
        tags=tags,
        output_format=output_format,
        system_prompt=system_prompt,
        user_prompt_scaffold=user_prompt_scaffold,
        enabled=enabled,
        version=next_version,
    )
    _write_yaml_mapping(template_path, template_payload)
    entry["enabled"] = bool(enabled)
    _write_yaml_mapping(user_index_path, index_payload)

    updated = get_prompt_template(
        target_id,
        include_disabled=True,
        include_user_templates=True,
        user_index_path=user_index_path,
    )
    if updated is None:
        raise ValueError(f"Failed to load updated template '{target_id}'.")
    return updated


def delete_user_prompt_template(
    template_id: str,
    *,
    user_index_path: Path = USER_PROMPT_INDEX_PATH,
) -> bool:
    """Delete a user template entry and its backing file."""
    target_id = _as_optional_id(template_id)
    if not target_id:
        return False
    index_payload = _load_or_init_user_index(user_index_path)
    raw_templates = index_payload.get("templates")
    if not isinstance(raw_templates, list):
        return False

    kept: list[dict[str, Any]] = []
    removed_file: str | None = None
    removed = False
    for item in raw_templates:
        if not isinstance(item, dict):
            continue
        current_id = _as_optional_id(item.get("id"))
        if current_id != target_id:
            kept.append(item)
            continue
        removed = True
        removed_file = _as_optional_str(item.get("file")) or f"templates/{target_id}.yaml"

    if not removed:
        return False
    index_payload["templates"] = kept
    default_id = _as_optional_id(index_payload.get("default_template_id"))
    if default_id == target_id:
        index_payload["default_template_id"] = _as_optional_id(kept[0].get("id")) if kept else None
    _write_yaml_mapping(user_index_path, index_payload)

    if removed_file:
        try:
            template_path = _resolve_index_relative_path(
                index_path=user_index_path,
                relative_path=removed_file,
                require_within_root=True,
            )
        except ValueError as exc:
            LOGGER.warning("Refusing to delete template file outside root: %s", exc)
            return True
        try:
            if template_path.exists():
                template_path.unlink()
        except OSError as exc:
            LOGGER.warning("Failed to remove template file '%s': %s", template_path, exc)
    return True


def set_user_default_prompt_template(
    template_id: str | None,
    *,
    user_index_path: Path = USER_PROMPT_INDEX_PATH,
) -> bool:
    """Set or clear the user-template default id in the user index."""
    index_payload = _load_or_init_user_index(user_index_path)
    raw_templates = index_payload.get("templates")
    if not isinstance(raw_templates, list):
        raw_templates = []
    target_id = _as_optional_id(template_id)
    if target_id and not any(_as_optional_id(item.get("id")) == target_id for item in raw_templates if isinstance(item, dict)):
        return False
    index_payload["default_template_id"] = target_id
    _write_yaml_mapping(user_index_path, index_payload)
    return True


def _load_templates_from_index(
    *,
    index_path: Path,
    include_disabled: bool,
    warn_if_missing: bool,
) -> tuple[list[PromptTemplate], str | None]:
    index = _load_yaml_mapping(index_path, warn_if_missing=warn_if_missing)
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
        try:
            template_path = _resolve_index_relative_path(
                index_path=index_path,
                relative_path=file_name,
                require_within_root=True,
            )
        except ValueError as exc:
            LOGGER.warning("Skipping prompt template '%s' with invalid file path: %s", template_id, exc)
            continue
        template_payload = _load_yaml_mapping(template_path, warn_if_missing=True)
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
    return templates, default_template_id


def _load_or_init_user_index(user_index_path: Path) -> dict[str, Any]:
    payload = _load_yaml_mapping(user_index_path, warn_if_missing=False)
    if payload is not None:
        if not isinstance(payload.get("templates"), list):
            payload["templates"] = []
        if "version" not in payload:
            payload["version"] = 1
        if "default_template_id" not in payload:
            payload["default_template_id"] = None
        return payload
    initialized = {"version": 1, "default_template_id": None, "templates": []}
    _write_yaml_mapping(user_index_path, initialized)
    return initialized


def _resolve_index_relative_path(*, index_path: Path, relative_path: str, require_within_root: bool) -> Path:
    root = index_path.parent.resolve()
    candidate = (root / relative_path).resolve()
    if require_within_root:
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"path escapes template root: {relative_path}") from exc
    return candidate


def _collect_template_ids(index_payload: dict[str, Any]) -> set[str]:
    raw_templates = index_payload.get("templates")
    if not isinstance(raw_templates, list):
        return set()
    ids: set[str] = set()
    for item in raw_templates:
        if not isinstance(item, dict):
            continue
        template_id = _as_optional_id(item.get("id"))
        if template_id:
            ids.add(template_id)
    return ids


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


def _normalize_or_slug_id(raw_id: str | None, *, name: str) -> str:
    if raw_id:
        normalized = _as_optional_id(raw_id)
        if not normalized or not _is_valid_template_id(normalized):
            raise ValueError("Template id must match pattern: lowercase letters/numbers/hyphen.")
        return normalized
    slug = _slugify(name)
    if not _is_valid_template_id(slug):
        raise ValueError("Template name did not produce a valid id. Provide an explicit template id.")
    return slug


def _is_valid_template_id(value: str) -> bool:
    return bool(_TEMPLATE_ID_PATTERN.match(value))


def _make_unique_id(base_id: str, *, blocked_ids: set[str]) -> str:
    if base_id not in blocked_ids:
        return base_id
    suffix = 2
    while True:
        candidate = f"{base_id}-{suffix}"
        if candidate not in blocked_ids and _is_valid_template_id(candidate):
            return candidate
        suffix += 1


def _slugify(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower())
    text = text.strip("-")
    if len(text) < 2:
        text = f"template-{text or 'custom'}"
    if len(text) > 64:
        text = text[:64].rstrip("-")
    return text or "template-custom"


def _build_template_payload(
    *,
    template_id: str,
    name: str,
    description: str,
    tags: list[str] | tuple[str, ...],
    output_format: str,
    system_prompt: str,
    user_prompt_scaffold: str,
    enabled: bool,
    version: int,
) -> dict[str, Any]:
    name_value = _as_optional_str(name)
    system_value = _as_optional_str(system_prompt)
    scaffold_value = _as_optional_str(user_prompt_scaffold)
    if not name_value or not system_value or not scaffold_value:
        raise ValueError("Name, system prompt, and user prompt scaffold are required.")
    return {
        "id": template_id,
        "name": name_value,
        "version": int(max(1, version)),
        "description": _as_optional_str(description) or "",
        "tags": list(_as_tags(list(tags))),
        "output_format": _as_output_format(output_format),
        "enabled": bool(enabled),
        "system_prompt": system_value,
        "user_prompt_scaffold": scaffold_value,
    }


def _load_yaml_mapping(path: Path, *, warn_if_missing: bool) -> dict[str, Any] | None:
    if not path.exists():
        if warn_if_missing:
            LOGGER.warning("Prompt YAML not found: %s", path)
        return None
    try:
        import yaml
    except Exception as exc:  # pragma: no cover
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


def _write_yaml_mapping(path: Path, payload: dict[str, Any]) -> None:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Unable to import yaml package: {exc}") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    dumped = yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)
    path.write_text(dumped, encoding="utf-8")


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
