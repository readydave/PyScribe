# models.py
# Model tier metadata and helpers for PyScribe.

from __future__ import annotations

import os
from typing import Dict, List, Optional, Set

BADGES = {
    "FAST": "🟢",
    "BALANCED": "🟡",
    "PRO": "🔴",
}

TIER_ORDER = {"PRO": 0, "BALANCED": 1, "FAST": 2}


def _entry(name: str, tier: str, vram: float, wer: float, rt: float) -> Dict:
    return {
        "name": name,
        "tier": tier,
        "vram": vram,
        "wer": wer,
        "rt": rt,
    }


# Curated, confirmed-available models with ballpark VRAM (GB), WER (%), and speed (x real-time)
TIERS: Dict[str, Dict] = {
    # Fast tier
    "tiny": _entry("tiny", "FAST", 0.5, 24.0, 120),
    "base": _entry("base", "FAST", 0.8, 20.0, 100),
    "small": _entry("small", "FAST", 1.5, 16.5, 80),
    "small.en": _entry("small.en", "FAST", 1.5, 15.5, 85),
    # Balanced tier
    "medium": _entry("medium", "BALANCED", 3.0, 13.0, 50),
    "distil-whisper/distil-large-v3": _entry("distil-whisper/distil-large-v3", "BALANCED", 3.0, 12.5, 55),
    "deepdml/faster-whisper-large-v3-turbo-ct2": _entry("deepdml/faster-whisper-large-v3-turbo-ct2", "BALANCED", 4.0, 12.0, 48),
    # Pro tier
    "guillaumekln/whisper-large-v2-ct2": _entry("guillaumekln/whisper-large-v2-ct2", "PRO", 6.0, 11.5, 40),
    "guillaumekln/whisper-large-v3-ct2": _entry("guillaumekln/whisper-large-v3-ct2", "PRO", 6.5, 11.0, 35),
    # Use CTranslate2/faster-whisper exports for turbo; raw OpenAI checkpoints are not supported by faster-whisper.
}


def detect_vram_gb() -> float:
    """Best-effort GPU VRAM detection in GB (returns 0 for CPU-only)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        props = torch.cuda.get_device_properties(0)
        return round(props.total_memory / (1024 ** 3), 1)
    except Exception:
        return 0.0


def _guess_tier(name: str) -> str:
    lowered = name.lower()
    if any(key in lowered for key in ["tiny", "base", "small"]):
        return "FAST"
    if any(key in lowered for key in ["distil", "turbo", "medium"]):
        return "BALANCED"
    return "PRO"


def get_ranked_models(
    vram_gb: float,
    remote: Optional[List[str]] = None,
    cached_models: Optional[Set[str]] = None,
) -> List[Dict]:
    """
    Returns a ranked list of model dicts with labels, badges, and tooltip text.
    Each dict: name, tier, badge, label, tooltip, fits (bool), verified (bool).
    """
    cached_models = cached_models or set()
    model_map = dict(TIERS)  # only curated, known-good models

    entries: List[Dict] = []
    for name, meta in model_map.items():
        tier = meta["tier"]
        if vram_gb <= 0 and tier != "FAST":
            continue  # CPU-only: show only fast tier

        badge = BADGES.get(tier, "")
        fits = vram_gb >= meta.get("vram", 0)
        verified = name in cached_models
        hq = tier == "PRO"
        label = f"{badge}{' ★' if hq else ''} {name}"
        if verified:
            label = f"✅ {label}"

        tooltip = (
            f"Tier: {tier}{' (HQ accuracy)' if hq else ''} | WER ~{meta.get('wer')}% | "
            f"Speed {meta.get('rt')}x | VRAM {meta.get('vram')} GB"
        )

        entries.append(
            {
                "name": name,
                "tier": tier,
                "badge": badge,
                "hq": hq,
                "label": label,
                "tooltip": tooltip,
                "fits": fits,
                "verified": verified,
                "vram": meta.get("vram"),
                "wer": meta.get("wer"),
                "rt": meta.get("rt"),
            }
        )

    def sort_key(item):
        return (not item["fits"], TIER_ORDER.get(item["tier"], 9), item["name"])

    entries.sort(key=sort_key)
    return entries


def strip_badges(label: str) -> str:
    """Remove leading badge/verified markers and trailing annotations from a label."""
    if not label:
        return label
    cleaned = label
    for sym in ["✅", "🟢", "🟡", "🔴"]:
        if cleaned.startswith(sym):
            cleaned = cleaned[len(sym):].strip()
    if "• Verified" in cleaned:
        cleaned = cleaned.split("• Verified")[0].strip()
    return cleaned
