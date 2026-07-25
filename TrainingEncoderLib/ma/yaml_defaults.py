"""Load SafePO multi-agent marl_cfg YAML defaults from sibling repo."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

RISE_ROOT = Path(__file__).resolve().parents[2]
MARL_CFG_ROOT = (
    RISE_ROOT
    / "Safe-Policy-Optimization"
    / "safepo"
    / "multi_agent"
    / "marl_cfg"
)

ALGO_YAML_DIR: dict[str, str] = {
    "mappo": "mappo",
    "happo": "happo",
    "mappo_lag": "mappolag",
    "ippo": "ippo",
    "ippo_lag": "ippo_lag",
}

CLI_TO_CONFIG: dict[str, str] = {
    "task": "env_name",
    "total-steps": "num_env_steps",
    "num-envs": "n_rollout_threads",
    "cost-limit": "cost_limit",
    "entropy-coef": "entropy_coef",
    "episode-length": "episode_length",
    "learning-iters": "learning_iters",
    "save-model-freq": "save_interval",
    "seed": "seed",
}


def yaml_dir_for_algo(algo: str) -> str:
    if algo not in ALGO_YAML_DIR:
        raise ValueError(f"unknown MA algorithm: {algo!r}")
    return ALGO_YAML_DIR[algo]


def load_ma_yaml(algo: str) -> dict[str, Any]:
    """Load ``marl_cfg/{algo}/config.yaml`` from sibling Safe-Policy-Optimization."""
    yaml_dir = yaml_dir_for_algo(algo)
    path = MARL_CFG_ROOT / yaml_dir / "config.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"MA YAML not found: {path}\n"
            f"Expected Safe-Policy-Optimization clone at {RISE_ROOT / 'Safe-Policy-Optimization'}"
        )
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping in {path}")
    return data


def _is_missing(value: Any) -> bool:
    return value is None or value == ""


def flatten_ma_yaml(yaml_base: dict[str, Any]) -> dict[str, Any]:
    """Top-level YAML scalars; blank values fall back to ``mamujoco`` nested keys."""
    mamujoco = yaml_base.get("mamujoco")
    mamujoco = mamujoco if isinstance(mamujoco, dict) else {}
    flat: dict[str, Any] = {}
    for key, value in yaml_base.items():
        if isinstance(value, dict):
            continue
        if _is_missing(value) and key in mamujoco and not _is_missing(mamujoco[key]):
            value = mamujoco[key]
        if not _is_missing(value):
            flat[key] = value
    return flat


def merge_ma_config(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Shallow merge: ``overrides`` win when present and non-empty."""
    merged = dict(base)
    for key, value in overrides.items():
        if not _is_missing(value):
            merged[key] = value
    return merged


def merge_yaml_gaps(base: dict[str, Any], existing: dict[str, Any]) -> dict[str, Any]:
    """Fill missing keys in ``existing`` from YAML ``base`` defaults."""
    merged = dict(base)
    for key, value in existing.items():
        if not _is_missing(value):
            merged[key] = value
    return merged


def cli_flags_to_config_overrides(flags: dict[str, str | list[str]]) -> dict[str, Any]:
    """Map thin MA CLI flags to SafePO config.json field names."""
    overrides: dict[str, Any] = {}
    for cli_key, cfg_key in CLI_TO_CONFIG.items():
        val = flags.get(cli_key)
        if isinstance(val, str):
            overrides[cfg_key] = val
    return overrides
