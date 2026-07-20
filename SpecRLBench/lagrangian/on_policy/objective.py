"""OpenAI objective-penalized Lag policy objective (on-policy)."""

from __future__ import annotations

import torch as th


def unclipped_surr(advantages: th.Tensor, ratio: th.Tensor) -> th.Tensor:
    """``mean(ratio * advantages)`` — never clip (cost surrogate / TRPO)."""
    return (ratio * advantages).mean()


def openai_lag_pi_objective(
    surr_adv: th.Tensor,
    surr_cost: th.Tensor,
    ent: th.Tensor | float,
    penalty: th.Tensor,
    ent_coef: float,
) -> th.Tensor:
    """``(surr_adv + ent_coef * ent - penalty * surr_cost) / (1 + penalty)``."""
    return (surr_adv + ent_coef * ent - penalty * surr_cost) / (1.0 + penalty)
