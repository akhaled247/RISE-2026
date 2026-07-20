"""Shared softplus Lagrangian dual (OpenAI PG EpCost update)."""

from __future__ import annotations

import numpy as np
import torch as th
from torch import nn
from torch.optim import Adam


class LagPenalty(nn.Module):
    """``penalty = softplus(penalty_param)``; dual on EpCost vs ``cost_lim``.

    Dual loss (minimize once per ``train()``)::

        -penalty_param * (ep_cost_mean - cost_lim)

    Init matches OpenAI safety-starter-agents::

        param_init = log(max(exp(penalty_init) - 1, 1e-8))
    """

    def __init__(self, penalty_init: float = 1.0, penalty_lr: float = 5e-2, device: th.device | str = "cpu"):
        super().__init__()
        device_t = th.device(device) if not isinstance(device, th.device) else device
        param_init = float(np.log(max(np.exp(penalty_init) - 1.0, 1e-8)))
        self.penalty_param = nn.Parameter(th.tensor(param_init, dtype=th.float32, device=device_t))
        self.optimizer = Adam([self.penalty_param], lr=penalty_lr)

    @property
    def penalty(self) -> th.Tensor:
        return th.nn.functional.softplus(self.penalty_param)

    def update(self, ep_cost_mean: float, cost_lim: float) -> float:
        """One Adam step on the EpCost dual; return scalar penalty for logging."""
        self.optimizer.zero_grad()
        (-self.penalty_param * (float(ep_cost_mean) - float(cost_lim))).backward()
        self.optimizer.step()
        return float(self.penalty.item())
