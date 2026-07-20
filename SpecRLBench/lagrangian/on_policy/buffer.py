"""Lag rollout buffers: SB3 RolloutBuffer/DictRolloutBuffer + cost GAE."""

from __future__ import annotations

from typing import Generator, NamedTuple

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.vec_env import VecNormalize

EPS = 1e-8


class LagRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    old_cost_values: th.Tensor
    cost_advantages: th.Tensor
    cost_returns: th.Tensor


class LagDictRolloutBufferSamples(NamedTuple):
    observations: dict[str, th.Tensor]
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    old_cost_values: th.Tensor
    cost_advantages: th.Tensor
    cost_returns: th.Tensor


def _compute_cost_gae(
    costs: np.ndarray,
    cost_values: np.ndarray,
    episode_starts: np.ndarray,
    last_cost_values: np.ndarray,
    dones: np.ndarray,
    buffer_size: int,
    gamma: float,
    cost_gamma: float,
    cost_gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """OpenAI quirk: cost TD deltas use reward gamma; returns use cost_gamma."""
    advantages = np.zeros_like(costs)
    last_gae_lam = 0.0
    for step in reversed(range(buffer_size)):
        if step == buffer_size - 1:
            next_non_terminal = 1.0 - dones.astype(np.float32)
            next_values = last_cost_values
        else:
            next_non_terminal = 1.0 - episode_starts[step + 1]
            next_values = cost_values[step + 1]
        # TD delta uses reward gamma (OpenAI CPOBuffer quirk)
        delta = costs[step] + gamma * next_values * next_non_terminal - cost_values[step]
        last_gae_lam = delta + cost_gamma * cost_gae_lambda * next_non_terminal * last_gae_lam
        advantages[step] = last_gae_lam
    returns = advantages + cost_values
    return advantages, returns


class _LagCostBufferMixin:
    """Shared cost fields + GAE for Box and Dict rollout buffers."""

    costs: np.ndarray
    cost_values: np.ndarray
    cost_advantages: np.ndarray
    cost_returns: np.ndarray
    cost_gamma: float
    cost_gae_lambda: float
    normalize_advantage_once: bool

    def _init_lag_cost_fields(
        self,
        cost_gamma: float,
        cost_gae_lambda: float,
        normalize_advantage_once: bool,
    ) -> None:
        self.cost_gamma = cost_gamma
        self.cost_gae_lambda = cost_gae_lambda
        self.normalize_advantage_once = normalize_advantage_once

    def reset(self) -> None:
        super().reset()  # type: ignore[misc]
        self.costs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(  # type: ignore[override]
        self,
        obs,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        cost: np.ndarray | None = None,
        cost_value: th.Tensor | None = None,
    ) -> None:
        if cost is None:
            cost = np.zeros(self.n_envs, dtype=np.float32)
        if cost_value is None:
            cost_value = th.zeros(self.n_envs, device=value.device)
        self.costs[self.pos] = np.asarray(cost, dtype=np.float32)
        self.cost_values[self.pos] = cost_value.clone().cpu().numpy().flatten()
        super().add(obs, action, reward, episode_start, value, log_prob)  # type: ignore[misc]

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        super().compute_returns_and_advantage(last_values, dones)  # type: ignore[misc]
        # Advantage norm once on full buffer (OpenAI Lag); not SB3 per-minibatch.
        if self.normalize_advantage_once:
            adv = self.advantages
            self.advantages = (adv - adv.mean()) / (adv.std() + EPS)

    def compute_cost_returns_and_advantage(self, last_cost_values: th.Tensor, dones: np.ndarray) -> None:
        last_cv = last_cost_values.clone().cpu().numpy().flatten()
        self.cost_advantages, self.cost_returns = _compute_cost_gae(
            self.costs,
            self.cost_values,
            self.episode_starts,
            last_cv,
            dones,
            self.buffer_size,
            self.gamma,
            self.cost_gamma,
            self.cost_gae_lambda,
        )
        # Center cost advantages once (no std)
        self.cost_advantages = self.cost_advantages - self.cost_advantages.mean()

    def _prepare_generator_flatten(self) -> None:
        raise NotImplementedError

    def get(self, batch_size: int | None = None):  # type: ignore[override]
        assert self.full
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        if not self.generator_ready:
            self._prepare_generator_flatten()
            self.generator_ready = True
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs
        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size


class LagRolloutBuffer(_LagCostBufferMixin, RolloutBuffer):
    """Box-obs rollout buffer with cost fields."""

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        cost_gamma: float = 0.99,
        cost_gae_lambda: float = 0.97,
        normalize_advantage_once: bool = True,
    ):
        self._init_lag_cost_fields(cost_gamma, cost_gae_lambda, normalize_advantage_once)
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs)

    def _prepare_generator_flatten(self) -> None:
        for tensor in (
            "observations",
            "actions",
            "values",
            "log_probs",
            "advantages",
            "returns",
            "cost_values",
            "cost_advantages",
            "cost_returns",
        ):
            self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])

    def get(self, batch_size: int | None = None) -> Generator[LagRolloutBufferSamples, None, None]:  # type: ignore[override]
        return super().get(batch_size)  # type: ignore[return-value]

    def _get_samples(  # type: ignore[override]
        self,
        batch_inds: np.ndarray,
        env: VecNormalize | None = None,
    ) -> LagRolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds].astype(np.float32, copy=False),
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.cost_values[batch_inds].flatten(),
            self.cost_advantages[batch_inds].flatten(),
            self.cost_returns[batch_inds].flatten(),
        )
        return LagRolloutBufferSamples(*tuple(map(self.to_torch, data)))


class LagDictRolloutBuffer(_LagCostBufferMixin, DictRolloutBuffer):
    """Dict-obs rollout buffer with cost fields (SAR / MultiInput)."""

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        cost_gamma: float = 0.99,
        cost_gae_lambda: float = 0.97,
        normalize_advantage_once: bool = True,
    ):
        self._init_lag_cost_fields(cost_gamma, cost_gae_lambda, normalize_advantage_once)
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs)

    def _prepare_generator_flatten(self) -> None:
        for key, obs in self.observations.items():
            self.observations[key] = self.swap_and_flatten(obs)
        for tensor in (
            "actions",
            "values",
            "log_probs",
            "advantages",
            "returns",
            "cost_values",
            "cost_advantages",
            "cost_returns",
        ):
            self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])

    def get(  # type: ignore[override]
        self,
        batch_size: int | None = None,
    ) -> Generator[LagDictRolloutBufferSamples, None, None]:
        return super().get(batch_size)  # type: ignore[return-value]

    def _get_samples(  # type: ignore[override]
        self,
        batch_inds: np.ndarray,
        env: VecNormalize | None = None,
    ) -> LagDictRolloutBufferSamples:
        return LagDictRolloutBufferSamples(
            observations={key: self.to_torch(obs[batch_inds]) for key, obs in self.observations.items()},
            actions=self.to_torch(self.actions[batch_inds].astype(np.float32, copy=False)),
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
            old_cost_values=self.to_torch(self.cost_values[batch_inds].flatten()),
            cost_advantages=self.to_torch(self.cost_advantages[batch_inds].flatten()),
            cost_returns=self.to_torch(self.cost_returns[batch_inds].flatten()),
        )
