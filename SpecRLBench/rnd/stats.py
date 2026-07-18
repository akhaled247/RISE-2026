"""Running statistics for RND observation and intrinsic-return normalization."""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.running_mean_std import RunningMeanStd


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

        self.obs_rms = RunningMeanStd(epsilon=rms_epsilon, shape=obs_shape)
        self.int_ret_rms = RunningMeanStd(epsilon=rms_epsilon, shape=())
        self.int_returns = np.zeros(n_envs, dtype=np.float64)

    def reset_returns(self, n_envs: int | None = None) -> None:
        if n_envs is not None:
            self.n_envs = n_envs
        self.int_returns = np.zeros(self.n_envs, dtype=np.float64)

    def normalize_obs(self, obs: np.ndarray, update: bool = True) -> np.ndarray:
        """Normalize flat obs ``(n_envs, dim)``; optionally update RMS."""
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
        """Normalize raw intrinsic rewards by running std of discounted returns.

        Does **not** subtract the mean (paper convention for intrinsic rewards).
        """
        r_int = np.asarray(r_int, dtype=np.float64).reshape(-1)
        dones = np.asarray(dones, dtype=np.float64).reshape(-1)
        assert r_int.shape[0] == self.n_envs

        # Discounted intrinsic returns; reset on episode end after update
        self.int_returns = self.gamma_int * self.int_returns * (1.0 - dones) + r_int
        if update and self.return_norm:
            self.int_ret_rms.update(self.int_returns)

        if not self.return_norm:
            return r_int.astype(np.float32)

        scale = np.sqrt(self.int_ret_rms.var + self.epsilon)
        r_norm = r_int / scale
        return np.clip(r_norm, -self.reward_clip, self.reward_clip).astype(np.float32)

    def get_state(self) -> dict:
        return {
            "obs_rms_mean": self.obs_rms.mean.copy(),
            "obs_rms_var": self.obs_rms.var.copy(),
            "obs_rms_count": float(self.obs_rms.count),
            "int_ret_rms_mean": float(self.int_ret_rms.mean),
            "int_ret_rms_var": float(self.int_ret_rms.var),
            "int_ret_rms_count": float(self.int_ret_rms.count),
            "int_returns": self.int_returns.copy(),
            "n_envs": self.n_envs,
        }

    def set_state(self, state: dict) -> None:
        self.obs_rms.mean = np.asarray(state["obs_rms_mean"], dtype=np.float64)
        self.obs_rms.var = np.asarray(state["obs_rms_var"], dtype=np.float64)
        self.obs_rms.count = float(state["obs_rms_count"])
        self.int_ret_rms.mean = np.asarray(state["int_ret_rms_mean"], dtype=np.float64)
        self.int_ret_rms.var = np.asarray(state["int_ret_rms_var"], dtype=np.float64)
        self.int_ret_rms.count = float(state["int_ret_rms_count"])
        self.n_envs = int(state["n_envs"])
        self.int_returns = np.asarray(state["int_returns"], dtype=np.float64)
        if self.int_returns.shape[0] != self.n_envs:
            self.int_returns = np.zeros(self.n_envs, dtype=np.float64)
