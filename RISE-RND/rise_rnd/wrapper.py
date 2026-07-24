"""Safety 6-tuple env wrapper that shapes reward with RND intrinsic bonus."""

from __future__ import annotations

from typing import Any

import numpy as np

from rise_rnd.config import RNDConfig
from rise_rnd.module import RNDModule


def _to_numpy1d(x: Any, n_envs: int) -> np.ndarray:
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.shape[0] != n_envs:
        raise ValueError(f"Expected batch size {n_envs}, got {arr.shape[0]}")
    return arr


def _to_bool1d(x: Any, n_envs: int) -> np.ndarray:
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    arr = np.asarray(x, dtype=np.bool_).reshape(-1)
    if arr.shape[0] != n_envs:
        raise ValueError(f"Expected batch size {n_envs}, got {arr.shape[0]}")
    return arr


class SafetyRNDWrapper:
    """Wrap Safety 6-tuple envs; add intrinsic reward; train predictor per epoch.

    SafePO sees only shaped scalar rewards. Predictor updates happen inside the
    wrapper after ``local_steps_per_epoch`` vector steps (one SafePO epoch).
    """

    def __init__(
        self,
        env: Any,
        *,
        config: RNDConfig | None = None,
        device: str = "cpu",
        local_steps_per_epoch: int,
        num_envs: int,
        training: bool = True,
    ) -> None:
        self.env = env
        self.config = config or RNDConfig()
        self.device = device
        self.local_steps_per_epoch = int(local_steps_per_epoch)
        self.num_envs = int(num_envs)
        self.training = training
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._step_in_epoch = 0
        self._epoch_buffer: list[np.ndarray] = []
        self.rnd: RNDModule | None = None
        if self.config.use_rnd and self.config.intrinsic_reward_coef != 0.0:
            self.rnd = RNDModule(
                self.observation_space,
                n_envs=self.num_envs,
                config=self.config,
                device=device,
            )

    @property
    def obs_rms(self):
        return getattr(self.env, "obs_rms", None)

    @obs_rms.setter
    def obs_rms(self, value) -> None:
        if hasattr(self.env, "obs_rms"):
            self.env.obs_rms = value

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def _maybe_flush_epoch(self) -> None:
        if self.rnd is None or not self.training:
            self._epoch_buffer.clear()
            self._step_in_epoch = 0
            return
        if self._step_in_epoch >= self.local_steps_per_epoch and self._epoch_buffer:
            buf = np.concatenate(self._epoch_buffer, axis=0)
            self.rnd.update_predictor_from_buffer(buf)
            self._epoch_buffer.clear()
            self._step_in_epoch = 0

    def reset(self, seed=None, options=None):
        self._step_in_epoch = 0
        self._epoch_buffer.clear()
        if self.rnd is not None:
            self.rnd.reset_returns(self.num_envs)
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        n = self.num_envs
        ext = _to_numpy1d(reward, n)
        term = _to_bool1d(terminated, n)
        trunc = _to_bool1d(truncated, n)
        dones = np.logical_or(term, trunc).astype(np.float32)

        if self.rnd is not None and self.config.use_rnd:
            rnd_obs, _raw, r_norm = self.rnd.compute_intrinsic_reward(
                obs, dones, n_envs=n, training=self.training
            )
            shaped = self.rnd.combine_rewards(ext, r_norm)
            if self.training:
                self._epoch_buffer.append(rnd_obs.astype(np.float32, copy=False))
                self._step_in_epoch += 1
                if self._step_in_epoch >= self.local_steps_per_epoch:
                    self._maybe_flush_epoch()
        else:
            shaped = ext

        # Preserve original dtype/container/shape (SafePO single-env uses batch dim 1)
        if hasattr(reward, "cpu"):
            import torch

            shaped_out = torch.as_tensor(shaped, dtype=reward.dtype, device=reward.device)
        else:
            orig = np.asarray(reward, dtype=np.float32)
            if orig.ndim == 0:
                shaped_out = float(shaped[0])
            else:
                shaped_out = shaped

        return obs, shaped_out, cost, terminated, truncated, info

    def close(self):
        return self.env.close()
