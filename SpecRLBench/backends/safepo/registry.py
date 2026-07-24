"""Algorithm registry for SafePO SpecRLBench backend."""

from __future__ import annotations

SUPPORTED_ALGOS = (
    "ppo",
    "trpo",
    "ppo_lag",
    "trpo_lag",
    "cpo",
)

DEFERRED_ALGOS = {
    "sac": "SAC deferred — SafePO has no SAC; vendor later (plan 1C).",
    "sac_lag": "SAC-Lag deferred with SAC.",
}


def resolve_algo(name: str) -> str:
    key = name.lower().replace("-", "_")
    if key in DEFERRED_ALGOS:
        raise NotImplementedError(DEFERRED_ALGOS[key])
    if key not in SUPPORTED_ALGOS:
        raise ValueError(f"Unknown algo {name!r}. Supported: {SUPPORTED_ALGOS}")
    return key
