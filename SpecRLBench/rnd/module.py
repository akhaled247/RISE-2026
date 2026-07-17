"""RND module: adapters, networks, stats, intrinsic reward, aux loss."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn

from rnd.config import RNDConfig
from rnd.networks import RNDModel
from rnd.obs_adapter import RNDObsAdapter
from rnd.stats import RNDRunningStats


class RNDModule(nn.Module):
    """Owns target/predictor, shared obs RMS, and OpenAI reward/aux helpers."""

    def __init__(
        self,
        observation_space: spaces.Space,
        n_envs: int,
        config: RNDConfig | None = None,
        device: th.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.config = (config or RNDConfig()).resolve()
        self.device = th.device(device)
        self.n_envs = n_envs
        self.adapter = RNDObsAdapter(
            observation_space,
            obs_key=self.config.obs_key,
            obs_keys=self.config.obs_keys,
        )

        image_shape = self._infer_image_shape(observation_space)
        self.image_shape = image_shape
        if image_shape is not None:
            # OpenAI: RMS over last channel only, shape HxWx1
            c, h, w = image_shape
            obs_rms_shape = (h, w, 1) if c >= 1 else (h, w)
            input_dim = None
        else:
            obs_rms_shape = (self.adapter.input_dim,)
            input_dim = self.adapter.input_dim

        self.stats = RNDRunningStats(
            obs_shape=obs_rms_shape,
            n_envs=n_envs,
            gamma=self.config.gamma,
            epsilon=self.config.epsilon,
            rms_epsilon=self.config.rms_epsilon,
            clip_obs=self.config.clip_obs,
            obs_norm=self.config.obs_norm,
            return_norm=self.config.return_norm,
        )
        self.model = RNDModel(
            input_dim=input_dim,
            image_shape=(1, image_shape[1], image_shape[2]) if image_shape else None,
            rep_size=self.config.rnd_rep_size,
            enlargement=self.config.enlargement,
        ).to(self.device)

    @staticmethod
    def _infer_image_shape(space: spaces.Space) -> tuple[int, int, int] | None:
        if not isinstance(space, spaces.Box) or len(space.shape) != 3:
            return None
        shape = space.shape
        if shape[0] in (1, 3, 4):
            return shape[0], shape[1], shape[2]
        if shape[-1] in (1, 3, 4):
            return shape[2], shape[0], shape[1]
        return None

    def prep_rnd_input(self, obs: Any, update_rms: bool = False) -> th.Tensor:
        """Normalize and tensorize observations for RND (OpenAI last-channel path)."""
        if self.image_shape is not None:
            arr = np.asarray(obs, dtype=np.float32)
            if arr.ndim == 3:
                arr = arr[None, ...]
            if arr.shape[-1] in (1, 3, 4):
                last = arr[..., -1:]
            else:
                last = arr[:, -1:, :, :].transpose(0, 2, 3, 1)
            if update_rms:
                self.stats.update_obs_rms(last)
            normed = self.stats.normalize_obs(last, update=False)
            x = th.as_tensor(normed, device=self.device, dtype=th.float32)
            if x.shape[-1] == 1:
                x = x.permute(0, 3, 1, 2).contiguous()
            return x

        if isinstance(obs, np.ndarray) and obs.ndim == 2 and obs.shape[-1] == self.adapter.input_dim:
            flat = obs.astype(np.float32, copy=False)
        else:
            flat = self.adapter.to_numpy(obs)
        if update_rms:
            self.stats.update_obs_rms(flat)
        normed = self.stats.normalize_obs(flat, update=False)
        return th.as_tensor(normed, device=self.device, dtype=th.float32)

    @th.no_grad()
    def compute_intrinsic_rewards(
        self,
        next_obs: np.ndarray,
        update_rms: bool = False,
    ) -> np.ndarray:
        """OpenAI ``int_rew``: mean squared feature error per sample.

        ``next_obs`` shape ``(n_envs, n_steps, ...)`` or ``(n_envs, ...)``.
        """
        next_obs = np.asarray(next_obs)
        flat_leading = next_obs.shape[0] * (next_obs.shape[1] if next_obs.ndim > len(self.adapter.observation_space.shape) + 1 or (self.image_shape is None and next_obs.ndim == 3) else 1)
        # Flatten env×time for batching
        if self.image_shape is None:
            if next_obs.ndim == 3:  # (E, T, D)
                e, t, d = next_obs.shape
                batch = next_obs.reshape(e * t, d)
                x = self.prep_rnd_input(batch, update_rms=update_rms)
                err = self.model.prediction_error(x).cpu().numpy().astype(np.float32)
                return err.reshape(e, t)
            x = self.prep_rnd_input(next_obs, update_rms=update_rms)
            return self.model.prediction_error(x).cpu().numpy().astype(np.float32)

        # Image: (E, T, H, W, C) or (E, T, C, H, W)
        if next_obs.ndim == 5:
            e, t = next_obs.shape[:2]
            batch = next_obs.reshape(e * t, *next_obs.shape[2:])
            x = self.prep_rnd_input(batch, update_rms=update_rms)
            err = self.model.prediction_error(x).cpu().numpy().astype(np.float32)
            return err.reshape(e, t)
        x = self.prep_rnd_input(next_obs, update_rms=update_rms)
        return self.model.prediction_error(x).cpu().numpy().astype(np.float32)

    def aux_loss(self, next_obs_batch: th.Tensor | np.ndarray) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        if isinstance(next_obs_batch, np.ndarray):
            x = self.prep_rnd_input(next_obs_batch, update_rms=False)
        else:
            x = next_obs_batch
        return self.model.aux_loss(
            x,
            proportion=self.config.proportion_of_exp_used_for_predictor_update,
        )

    def normalize_intrinsic_rewards(self, rews_int: np.ndarray) -> np.ndarray:
        return self.stats.normalize_intrinsic_rewards(rews_int)

    # --- legacy helpers kept for older unit tests ---
    @th.no_grad()
    def compute_intrinsic_reward(
        self,
        obs: Any,
        dones: np.ndarray,
        training: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Legacy single-step API (not used by OpenAI-faithful rollouts)."""
        flat = self.adapter.to_numpy(obs)
        rnd_obs = self.stats.normalize_obs(flat, update=training)
        x = th.as_tensor(rnd_obs, device=self.device, dtype=th.float32)
        r_raw = self.model.prediction_error(x).cpu().numpy().astype(np.float32)
        # Approximate online norm via forward filter one-step
        r_norm = self.stats.normalize_intrinsic_rewards(r_raw.reshape(self.n_envs, 1)).reshape(-1)
        return rnd_obs, r_raw, r_norm

    def combine_rewards(
        self,
        extrinsic: np.ndarray,
        intrinsic_norm: np.ndarray,
    ) -> np.ndarray:
        """Legacy combined-reward helper (not used in faithful path)."""
        beta = self.config.int_coeff if self.config.use_rnd else 0.0
        return (
            np.asarray(extrinsic, dtype=np.float32)
            + float(beta) * np.asarray(intrinsic_norm, dtype=np.float32)
        )

    def train_predictor(self, rnd_obs_batch: th.Tensor) -> dict[str, float]:
        """Legacy separate Adam step — unused in faithful joint optimization."""
        loss, feat_var, max_feat = self.aux_loss(rnd_obs_batch)
        return {
            "predictor_loss": float(loss.item()),
            "predictor_grad_norm": 0.0,
            "prediction_error_mean": float(loss.item()),
            "target_feature_norm": float(max_feat.item()),
            "predictor_feature_norm": 0.0,
            "feat_var": float(feat_var.item()),
        }

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "stats": self.stats.get_state(),
        }

    def set_extra_state(self, state: dict[str, Any]) -> None:
        if "config" in state:
            self.config = RNDConfig.from_dict(state["config"])
        if "stats" in state:
            self.stats.set_state(state["stats"])
