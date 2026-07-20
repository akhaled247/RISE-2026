"""Per-step cost extraction from Gymnasium info dicts."""

from __future__ import annotations


def cost_from_info(info: dict) -> float:
    """OpenAI: ``info.get('cost', 0)``."""
    return float(info.get("cost", 0))
