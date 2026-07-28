"""Algorithm registry for SafePO SpecRLBench backend."""

from __future__ import annotations

SUPPORTED_ALGOS = (
    "ppo",
    "trpo",
    "ppo_lag",
    "trpo_lag",
    "cpo",
)

SUPPORTED_MA_ALGOS = (
    "mappo",
    "happo",
    "mappo_lag",
    "mappolag",
    "ippo",
    "ippo_lag",
)

DEFERRED_ALGOS = {
    "sac": "SAC deferred — SafePO has no SAC; vendor later (plan 1C).",
    "sac_lag": "SAC-Lag deferred with SAC.",
}

# SpecRL name → SafePO multi_agent module stem / multi_agent_args algo key
MA_ALGO_MODULE = {
    "mappo": "mappo",
    "happo": "happo",
    "mappo_lag": "mappolag",
    "mappolag": "mappolag",
    "ippo": "ippo",
    "ippo_lag": "ippo_lag",
}


def resolve_algo(name: str) -> str:
    key = name.lower().replace("-", "_")
    if key in DEFERRED_ALGOS:
        raise NotImplementedError(DEFERRED_ALGOS[key])
    if key not in SUPPORTED_ALGOS:
        raise ValueError(f"Unknown algo {name!r}. Supported: {SUPPORTED_ALGOS}")
    return key


def resolve_ma_algo(name: str) -> str:
    key = name.lower().replace("-", "_")
    if key not in SUPPORTED_MA_ALGOS:
        raise ValueError(f"Unknown MA algo {name!r}. Supported: {SUPPORTED_MA_ALGOS}")
    return key
