"""SAC policies with twin cost Q-networks (OpenAI Qc, SB3-shaped)."""

from __future__ import annotations

from typing import Any

import torch as th
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac.policies import (
    MultiInputPolicy as SacMultiInputPolicy,
)
from stable_baselines3.sac.policies import (
    SACPolicy,
)


class CostQMixin:
    """Twin Qc + target nets parallel to SB3 twin Qf."""

    cost_critic: Any
    cost_critic_target: Any

    def _build_cost_critics(self, lr_schedule: Schedule) -> None:
        # Separate features extractor (same as SB3 non-shared critic path)
        self.cost_critic = self.make_critic(features_extractor=None)  # type: ignore[attr-defined]
        self.cost_critic_target = self.make_critic(features_extractor=None)  # type: ignore[attr-defined]
        self.cost_critic_target.load_state_dict(self.cost_critic.state_dict())
        self.cost_critic.optimizer = self.optimizer_class(  # type: ignore[attr-defined]
            self.cost_critic.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,  # type: ignore[attr-defined]
        )
        self.cost_critic_target.set_training_mode(False)

    def cost_q_values(self, obs: th.Tensor, actions: th.Tensor) -> tuple[th.Tensor, ...]:
        """Twin Qc(s, a) as a tuple of tensors."""
        return self.cost_critic(obs, actions)

    def cost_q_mean(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """Mean over twin Qc (documented twin-mean aggregation)."""
        qs = th.cat(self.cost_critic(obs, actions), dim=1)
        return qs.mean(dim=1, keepdim=True)

    def cost_q_target_mean(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        qs = th.cat(self.cost_critic_target(obs, actions), dim=1)
        return qs.mean(dim=1, keepdim=True)

    def soft_update_cost_critic(self, tau: float) -> None:
        polyak_update(self.cost_critic.parameters(), self.cost_critic_target.parameters(), tau)


class LagSACPolicy(CostQMixin, SACPolicy):
    """Mlp SAC policy with twin cost critics."""

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
        self._build_cost_critics(lr_schedule)


class LagMultiInputSACPolicy(CostQMixin, SacMultiInputPolicy):
    """MultiInput SAC policy with twin cost critics (Dict obs / SAR)."""

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
        self._build_cost_critics(lr_schedule)
