"""Shim: re-export Lag actor-critic policies from shared ``lagrangian.on_policy``."""

from lagrangian.on_policy.policy import (  # noqa: F401
    CostValueMixin,
    LagActorCriticPolicy,
    LagMultiInputActorCriticPolicy,
)

__all__ = [
    "CostValueMixin",
    "LagActorCriticPolicy",
    "LagMultiInputActorCriticPolicy",
]
