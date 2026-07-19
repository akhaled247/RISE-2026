"""Lag-augmented SB3 actor-critic policies with a cost value head."""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from torch import nn


class CostValueMixin:
    """Adds ``cost_net`` parallel to SB3 ``value_net``."""

    cost_net: nn.Linear

    def _build_cost_net(self) -> None:
        assert hasattr(self, "mlp_extractor")
        self.cost_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)  # type: ignore[attr-defined]
        if getattr(self, "ortho_init", False):
            self.cost_net.apply(partial(self.init_weights, gain=1))  # type: ignore[attr-defined]

    def predict_cost_values(self, obs: PyTorchObs) -> th.Tensor:
        """Cost critic estimate V_c(s)."""
        features = super().extract_features(obs, self.vf_features_extractor)  # type: ignore[misc]
        latent_vf = self.mlp_extractor.forward_critic(features)  # type: ignore[attr-defined]
        return self.cost_net(latent_vf)

    def evaluate_actions(
        self, obs: PyTorchObs, actions: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor | None]:
        """Return values, cost_values, log_prob, entropy."""
        features = self.extract_features(obs)  # type: ignore[attr-defined]
        if self.share_features_extractor:  # type: ignore[attr-defined]
            latent_pi, latent_vf = self.mlp_extractor(features)  # type: ignore[attr-defined]
        else:
            pi_features, vf_features = features  # type: ignore[misc]
            latent_pi = self.mlp_extractor.forward_actor(pi_features)  # type: ignore[attr-defined]
            latent_vf = self.mlp_extractor.forward_critic(vf_features)  # type: ignore[attr-defined]
        distribution = self._get_action_dist_from_latent(latent_pi)  # type: ignore[attr-defined]
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)  # type: ignore[attr-defined]
        cost_values = self.cost_net(latent_vf)
        entropy = distribution.entropy()
        return values, cost_values, log_prob, entropy

    def pi_parameters(self) -> list[nn.Parameter]:
        """Actor parameters for dual-optimizer (openai) mode."""
        params: list[nn.Parameter] = list(self.action_net.parameters())  # type: ignore[attr-defined]
        if hasattr(self, "log_std") and isinstance(self.log_std, nn.Parameter):  # type: ignore[attr-defined]
            params.append(self.log_std)  # type: ignore[attr-defined]
        params.extend(self.mlp_extractor.policy_net.parameters())  # type: ignore[attr-defined]
        params.extend(self.pi_features_extractor.parameters())  # type: ignore[attr-defined]
        return params

    def vf_parameters(self) -> list[nn.Parameter]:
        """Reward + cost critic parameters for dual-optimizer (openai) mode."""
        params: list[nn.Parameter] = list(self.value_net.parameters())  # type: ignore[attr-defined]
        params.extend(self.cost_net.parameters())
        params.extend(self.mlp_extractor.value_net.parameters())  # type: ignore[attr-defined]
        params.extend(self.vf_features_extractor.parameters())  # type: ignore[attr-defined]
        return params


class LagActorCriticPolicy(CostValueMixin, ActorCriticPolicy):
    """MlpPolicy with cost critic."""

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
        self._build_cost_net()
        # Rebuild optimizer to include cost_net (Tier 3 / SB3 single-optimizer path)
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )  # type: ignore[call-arg]


class LagMultiInputActorCriticPolicy(CostValueMixin, MultiInputActorCriticPolicy):
    """MultiInputPolicy with cost critic (Dict obs / SAR)."""

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
        self._build_cost_net()
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )  # type: ignore[call-arg]
