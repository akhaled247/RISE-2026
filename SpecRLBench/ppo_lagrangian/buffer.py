"""Shim: re-export Lag rollout buffers from shared ``lagrangian.on_policy``."""

from lagrangian.on_policy.buffer import (  # noqa: F401
    LagDictRolloutBuffer,
    LagDictRolloutBufferSamples,
    LagRolloutBuffer,
    LagRolloutBufferSamples,
    _compute_cost_gae,
)

__all__ = [
    "LagRolloutBuffer",
    "LagDictRolloutBuffer",
    "LagRolloutBufferSamples",
    "LagDictRolloutBufferSamples",
    "_compute_cost_gae",
]
