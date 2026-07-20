"""Lag replay buffers: SB3 ReplayBuffer/DictReplayBuffer + per-transition cost."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import torch as th
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize


class LagReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    costs: th.Tensor
    discounts: th.Tensor | None = None


class LagDictReplayBufferSamples(NamedTuple):
    observations: dict[str, th.Tensor]
    actions: th.Tensor
    next_observations: dict[str, th.Tensor]
    dones: th.Tensor
    rewards: th.Tensor
    costs: th.Tensor
    discounts: th.Tensor | None = None


class _LagReplayCostMixin:
    """Store raw ``info['cost']``; never VecNormalize costs."""

    costs: np.ndarray

    def _init_cost_storage(self) -> None:
        self.costs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)  # type: ignore[attr-defined]

    def _store_costs(self, infos: list[dict[str, Any]]) -> None:
        costs = np.array([float(info.get("cost", 0)) for info in infos], dtype=np.float32)
        self.costs[self.pos] = costs  # type: ignore[attr-defined]


class LagReplayBuffer(_LagReplayCostMixin, ReplayBuffer):
    """Box-obs replay buffer with a cost field."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._init_cost_storage()

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        self._store_costs(infos)
        super().add(obs, next_obs, action, reward, done, infos)

    def _get_samples(  # type: ignore[override]
        self,
        batch_inds: np.ndarray,
        env: VecNormalize | None = None,
    ) -> LagReplayBufferSamples:
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env
            )
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        obs = self._normalize_obs(self.observations[batch_inds, env_indices, :], env)
        actions = self.actions[batch_inds, env_indices, :]
        dones = (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(
            -1, 1
        )
        rewards = self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env)
        costs = self.costs[batch_inds, env_indices].reshape(-1, 1)

        return LagReplayBufferSamples(
            observations=self.to_torch(obs),
            actions=self.to_torch(actions),
            next_observations=self.to_torch(next_obs),
            dones=self.to_torch(dones),
            rewards=self.to_torch(rewards),
            costs=self.to_torch(costs),
            discounts=None,
        )


class LagDictReplayBuffer(_LagReplayCostMixin, DictReplayBuffer):
    """Dict-obs replay buffer with a cost field."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._init_cost_storage()

    def add(  # type: ignore[override]
        self,
        obs: dict[str, np.ndarray],
        next_obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        self._store_costs(infos)
        super().add(obs, next_obs, action, reward, done, infos)

    def _get_samples(  # type: ignore[override]
        self,
        batch_inds: np.ndarray,
        env: VecNormalize | None = None,
    ) -> LagDictReplayBufferSamples:
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        obs_ = self._normalize_obs(
            {key: obs[batch_inds, env_indices, :] for key, obs in self.observations.items()}, env
        )
        next_obs_ = self._normalize_obs(
            {key: obs[batch_inds, env_indices, :] for key, obs in self.next_observations.items()}, env
        )
        assert isinstance(obs_, dict)
        assert isinstance(next_obs_, dict)

        costs = self.costs[batch_inds, env_indices].reshape(-1, 1)
        return LagDictReplayBufferSamples(
            observations={key: self.to_torch(o) for key, o in obs_.items()},
            actions=self.to_torch(self.actions[batch_inds, env_indices]),
            next_observations={key: self.to_torch(o) for key, o in next_obs_.items()},
            dones=self.to_torch(
                self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])
            ).reshape(-1, 1),
            rewards=self.to_torch(
                self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env)
            ),
            costs=self.to_torch(costs),
            discounts=None,
        )
