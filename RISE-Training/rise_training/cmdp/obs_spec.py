"""Observation flatten helpers for SA train → MA deploy (paper protocol)."""

from __future__ import annotations

from typing import Any

import numpy as np

from rise_training.cmdp.flatten import DictFlattenWrapper


def probe_flatten_keys(env_id: str, *, sar_ltl_ordering: bool = False) -> list[str]:
    """Return sorted DictFlatten keys used during SA training on ``env_id``."""
    from rise_training.cmdp.factory import make_cmdp_env

    env = make_cmdp_env(
        env_id,
        normalize_obs=False,
        autoreset=False,
        training=False,
        sar_ltl_ordering=sar_ltl_ordering,
    )
    try:
        cur: Any = env
        while cur is not None:
            if isinstance(cur, DictFlattenWrapper):
                return list(cur.flatten_keys)
            cur = getattr(cur, "env", None)
        raise RuntimeError(f"No DictFlattenWrapper in env stack for {env_id!r}")
    finally:
        env.close()


def flatten_agent_obs(agent_obs: dict[str, Any], keys: list[str]) -> np.ndarray:
    """Concatenate per-agent dict obs in stable key order (matches DictFlattenWrapper)."""
    parts = [np.ravel(np.asarray(agent_obs[k], dtype=np.float32)) for k in keys]
    return np.concatenate(parts, axis=0).astype(np.float32)


def normalize_obs_vector(
    vec: np.ndarray,
    rms_state: dict[str, Any] | Any,
    *,
    clip_obs: float = 10.0,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Apply frozen RMS normalization (SafePO Normalizer / ObsNormalizeWrapper state)."""
    if hasattr(rms_state, "mean") and hasattr(rms_state, "var"):
        mean = np.asarray(rms_state.mean, dtype=np.float32)
        var = np.asarray(rms_state.var, dtype=np.float32)
    elif isinstance(rms_state, dict):
        mean = np.asarray(rms_state["mean"], dtype=np.float32)
        var = np.asarray(rms_state["var"], dtype=np.float32)
        clip_obs = float(rms_state.get("clip_obs", clip_obs))
        epsilon = float(rms_state.get("epsilon", epsilon))
    else:
        raise TypeError(f"Unsupported RMS state type: {type(rms_state)}")
    vec = np.asarray(vec, dtype=np.float32)
    out = (vec - mean) / np.sqrt(var + epsilon)
    return np.clip(out, -clip_obs, clip_obs).astype(np.float32)


def load_rms_from_pkl(norm_path: str) -> Any:
    """Load SafePO joblib Normalizer blob from a run directory."""
    import joblib

    state = joblib.load(norm_path)
    if isinstance(state, dict) and "Normalizer" in state:
        return state["Normalizer"]
    return state
