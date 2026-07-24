"""Running statistics for RND observation and intrinsic-return normalization."""

from __future__ import annotations

import numpy as np


class RunningMeanStd:
    """Welford running mean/variance."""

    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = float(x.shape[0])
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total
        self.mean = new_mean
        self.var = m2 / total
        self.count = total


class RNDRunningStats:
    """Observation RMS + discounted intrinsic-return RMS (Burda et al.)."""

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        n_envs: int,
        gamma_int: float = 0.99,
        epsilon: float = 1e-8,
        rms_epsilon: float = 1e-4,
        obs_clip: float = 5.0,
        reward_clip: float = 5.0,
        obs_norm: bool = True,
        return_norm: bool = True,
    ) -> None:
        self.gamma_int = gamma_int
        self.epsilon = epsilon
        self.obs_clip = obs_clip
        self.reward_clip = reward_clip
        self.obs_norm = obs_norm
        self.return_norm = return_norm
        self.n_envs = n_envs
        self.obs_rms = RunningMeanStd(shape=obs_shape, epsilon=rms_epsilon)
        self.int_ret_rms = RunningMeanStd(shape=(), epsilon=rms_epsilon)
        self.int_returns = np.zeros(n_envs, dtype=np.float64)

    def reset_returns(self, n_envs: int | None = None) -> None:
        if n_envs is not None:
            self.n_envs = n_envs
        self.int_returns = np.zeros(self.n_envs, dtype=np.float64)

    def normalize_obs(self, obs: np.ndarray, update: bool = True) -> np.ndarray:
        if not self.obs_norm:
            return obs.astype(np.float32, copy=False)
        if update:
            self.obs_rms.update(obs)
        normed = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)
        return np.clip(normed, -self.obs_clip, self.obs_clip).astype(np.float32)

    def normalize_intrinsic_reward(
        self,
        r_int: np.ndarray,
        dones: np.ndarray,
        update: bool = True,
    ) -> np.ndarray:
        r_int = np.asarray(r_int, dtype=np.float64).reshape(-1)
        dones = np.asarray(dones, dtype=np.float64).reshape(-1)
        assert r_int.shape[0] == self.n_envs
        self.int_returns = self.gamma_int * self.int_returns * (1.0 - dones) + r_int
        if update and self.return_norm:
            self.int_ret_rms.update(self.int_returns)
        if not self.return_norm:
            return r_int.astype(np.float32)
        scale = np.sqrt(self.int_ret_rms.var + self.epsilon)
        r_norm = r_int / scale
        return np.clip(r_norm, -self.reward_clip, self.reward_clip).astype(np.float32)
