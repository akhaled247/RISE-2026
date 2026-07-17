"""Rollout storage for RND predictor training inputs and diagnostics."""

from __future__ import annotations

import numpy as np
import torch as th
from stable_baselines3.common.utils import get_device


class RNDStorage:
    """Stores flattened RND observations and reward diagnostics for one rollout."""

    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        input_dim: int,
        device: th.device | str = "auto",
    ) -> None:
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.input_dim = input_dim
        self.device = get_device(device)
        self.pos = 0
        self.full = False
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.rnd_obs = np.zeros((self.n_steps, self.n_envs, self.input_dim), dtype=np.float32)
        self.raw_intrinsic = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.norm_intrinsic = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.extrinsic = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.combined = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.pos = 0
        self.full = False
        self.generator_ready = False

    def add(
        self,
        rnd_obs: np.ndarray,
        raw_intrinsic: np.ndarray,
        norm_intrinsic: np.ndarray,
        extrinsic: np.ndarray,
        combined: np.ndarray,
    ) -> None:
        self.rnd_obs[self.pos] = np.asarray(rnd_obs, dtype=np.float32)
        self.raw_intrinsic[self.pos] = np.asarray(raw_intrinsic, dtype=np.float32)
        self.norm_intrinsic[self.pos] = np.asarray(norm_intrinsic, dtype=np.float32)
        self.extrinsic[self.pos] = np.asarray(extrinsic, dtype=np.float32)
        self.combined[self.pos] = np.asarray(combined, dtype=np.float32)
        self.pos += 1
        if self.pos == self.n_steps:
            self.full = True

    def _prepare(self) -> None:
        if self.generator_ready:
            return
        assert self.full, "RNDStorage must be full before sampling"
        # Flatten (n_steps, n_envs, ...) -> (n_steps * n_envs, ...)
        self.rnd_obs = self.rnd_obs.swapaxes(0, 1).reshape(-1, self.input_dim)
        self.raw_intrinsic = self.raw_intrinsic.swapaxes(0, 1).reshape(-1)
        self.norm_intrinsic = self.norm_intrinsic.swapaxes(0, 1).reshape(-1)
        self.extrinsic = self.extrinsic.swapaxes(0, 1).reshape(-1)
        self.combined = self.combined.swapaxes(0, 1).reshape(-1)
        self.generator_ready = True

    def get_rnd_obs_batch(self, batch_inds: np.ndarray) -> th.Tensor:
        self._prepare()
        return th.as_tensor(self.rnd_obs[batch_inds], device=self.device, dtype=th.float32)

    def summary(self) -> dict[str, float]:
        """Aggregate diagnostics over the current (pre-flatten or flattened) buffer."""
        if self.generator_ready:
            raw = self.raw_intrinsic
            norm = self.norm_intrinsic
            ext = self.extrinsic
            comb = self.combined
        else:
            n = max(self.pos, 1)
            raw = self.raw_intrinsic[:n].ravel()
            norm = self.norm_intrinsic[:n].ravel()
            ext = self.extrinsic[:n].ravel()
            comb = self.combined[:n].ravel()
        return {
            "raw_mean": float(np.mean(raw)),
            "raw_std": float(np.std(raw)),
            "raw_min": float(np.min(raw)),
            "raw_max": float(np.max(raw)),
            "norm_mean": float(np.mean(norm)),
            "norm_std": float(np.std(norm)),
            "norm_min": float(np.min(norm)),
            "norm_max": float(np.max(norm)),
            "extrinsic_mean": float(np.mean(ext)),
            "combined_mean": float(np.mean(comb)),
            "contribution_mean": float(np.mean(comb - ext)),
        }
