"""Hugging Face token helpers for gated model access."""

from __future__ import annotations

import os

from huggingface_hub import HfFolder

_SESSION_HF_TOKEN: str | None = None


def get_hf_token() -> str | None:
    """Returns token from session memory, env, or saved Hugging Face auth cache."""
    if _SESSION_HF_TOKEN:
        return _SESSION_HF_TOKEN
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if env_token:
        return env_token.strip() or None
    token = HfFolder.get_token()
    if token:
        token = token.strip()
    return token or None


def save_hf_token(token: str, *, persist: bool = False) -> None:
    """Stores token for the current session and optionally persists it to the HF cache."""
    global _SESSION_HF_TOKEN
    value = (token or "").strip()
    if not value:
        raise ValueError("Token is empty.")
    _SESSION_HF_TOKEN = value
    if persist:
        HfFolder.save_token(value)


def clear_session_hf_token() -> None:
    """Clears any in-memory session token."""
    global _SESSION_HF_TOKEN
    _SESSION_HF_TOKEN = None
