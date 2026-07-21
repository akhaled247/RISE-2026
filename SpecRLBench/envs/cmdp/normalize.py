"""Running observation normalization (obs only; never reward/cost)."""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np
from gymnasium.core import ActType, ObsType


class RunningMeanStd:
    """Welford running mean/variance."""

    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: float
    ) -> None:
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total
        self.mean = new_mean
        self.var = m2 / total
        self.count = total


class ObsNormalizeWrapper(gymnasium.Wrapper):
    """Normalize observations with running RMS; clip like SB3 VecNormalize."""

    def __init__(
        self,
        env: gymnasium.Env,
        clip_obs: float = 10.0,
        epsilon: float = 1e-8,
        training: bool = True,
    ):
        super().__init__(env)
        shape = env.observation_space.shape
        assert shape is not None and len(shape) == 1
        self.obs_rms = RunningMeanStd(shape=shape)
        self.clip_obs = clip_obs
        self.epsilon = epsilon
        self.training = training

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if self.training:
            self.obs_rms.update(obs)
        mean = self.obs_rms.mean.astype(np.float32)
        var = self.obs_rms.var.astype(np.float32)
        out = (obs - mean) / np.sqrt(var + self.epsilon)
        return np.clip(out, -self.clip_obs, self.clip_obs).astype(np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        return self._normalize(obs), info

    def step(self, action: ActType):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        if "final_observation" in info:
            info = dict(info)
            info["final_observation"] = self._normalize(
                np.asarray(info["final_observation"], dtype=np.float32)
            )
        return self._normalize(obs), reward, cost, terminated, truncated, info

    def get_rms_state(self) -> dict[str, Any]:
        return {
            "mean": self.obs_rms.mean.copy(),
            "var": self.obs_rms.var.copy(),
            "count": float(self.obs_rms.count),
            "clip_obs": self.clip_obs,
            "epsilon": self.epsilon,
        }

    def set_rms_state(self, state: dict[str, Any]) -> None:
        self.obs_rms.mean = np.asarray(state["mean"], dtype=np.float64)
        self.obs_rms.var = np.asarray(state["var"], dtype=np.float64)
        self.obs_rms.count = float(state["count"])
        self.clip_obs = float(state.get("clip_obs", self.clip_obs))
        self.epsilon = float(state.get("epsilon", self.epsilon))
