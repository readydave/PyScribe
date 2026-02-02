"""Hugging Face token helpers for gated model access."""

from __future__ import annotations

import os

from huggingface_hub import HfFolder


def get_hf_token() -> str | None:
    """Returns token from env or saved Hugging Face auth cache."""
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if env_token:
        return env_token.strip() or None
    token = HfFolder.get_token()
    if token:
        token = token.strip()
    return token or None


def save_hf_token(token: str) -> None:
    """Saves token to Hugging Face cache and current process env."""
    value = (token or "").strip()
    if not value:
        raise ValueError("Token is empty.")
    HfFolder.save_token(value)
    os.environ["HF_TOKEN"] = value
