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


def infer_actor_obs_dim(state_dict: dict[str, Any]) -> int:
    """Input dimension of SafePO Actor MLP from checkpoint weights."""
    weight = state_dict.get("mean.0.weight")
    if weight is None:
        raise KeyError("Actor state_dict missing mean.0.weight")
    return int(weight.shape[1])


def probe_train_obs_dim(env_id: str, *, sar_ltl_ordering: bool = False) -> int:
    """Flat Box obs size used during SA training on ``env_id``."""
    from rise_training.cmdp.factory import make_cmdp_env

    env = make_cmdp_env(
        env_id,
        normalize_obs=False,
        autoreset=False,
        training=False,
        sar_ltl_ordering=sar_ltl_ordering,
    )
    try:
        return int(env.observation_space.shape[0])
    finally:
        env.close()


def _key_ravel_size(obs: dict[str, Any], key: str) -> int | str:
    if key not in obs:
        return "MISSING"
    return int(np.ravel(np.asarray(obs[key], dtype=np.float32)).size)


def assert_deploy_obs_compatible(
    flat_vec: np.ndarray,
    keys: list[str],
    expected_dim: int,
    *,
    train_env: str,
    eval_env: str,
    sar_ltl_ordering: bool = False,
) -> None:
    """Raise if deploy flatten length differs from trained actor input dim."""
    actual = int(len(flat_vec))
    if actual == expected_dim:
        return

    from rise_training.env_utils import make_env

    mismatches: list[str] = []
    train_obs: dict[str, Any] | None = None
    eval_obs: dict[str, Any] | None = None
    try:
        train_e = make_env(train_env, flat=True, sar_ltl_ordering=sar_ltl_ordering)
        train_obs, _ = train_e.reset(seed=0)
        train_e.close()
    except Exception as exc:
        mismatches.append(f"train probe failed ({train_env!r}): {exc}")
    try:
        eval_e = make_env(eval_env, flat=False, sar_ltl_ordering=sar_ltl_ordering)
        eval_obs0, _ = eval_e.reset(seed=0)
        agents = list(eval_e.unwrapped.possible_agents)
        eval_obs = eval_obs0[agents[0]]
        eval_e.close()
    except Exception as exc:
        mismatches.append(f"eval probe failed ({eval_env!r}): {exc}")

    if train_obs is not None and eval_obs is not None:
        for key in keys:
            s_train = _key_ravel_size(train_obs, key)
            s_eval = _key_ravel_size(eval_obs, key)
            if s_train != s_eval:
                mismatches.append(f"{key!r}: train={s_train} eval_agent_0={s_eval}")

    detail = "; ".join(mismatches) if mismatches else "no per-key diff (check flatten_keys order)"
    raise ValueError(
        f"Deploy obs dim {actual} != trained actor obs_dim {expected_dim} "
        f"(train={train_env!r}, eval={eval_env!r}). {detail}"
    )
