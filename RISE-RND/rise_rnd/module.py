"""RND module: networks, stats, reward computation, predictor training."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn

from rise_rnd.config import RNDConfig
from rise_rnd.networks import RNDModel
from rise_rnd.stats import RNDRunningStats


def _obs_to_batch(obs: Any, n_envs: int) -> np.ndarray:
    """Flatten Box obs to ``(n_envs, dim)`` float32."""
    arr = np.asarray(obs, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr.reshape(arr.shape[0], -1)


class RNDModule(nn.Module):
    """Owns target/predictor nets, running stats, and reward helpers."""

    def __init__(
        self,
        observation_space: spaces.Space,
        n_envs: int,
        config: RNDConfig | None = None,
        device: th.device | str = "cpu",
    ) -> None:
        super().__init__()
        if not isinstance(observation_space, spaces.Box):
            raise TypeError(
                f"RISE-RND expects flat Box observations, got {type(observation_space)}"
            )
        self.config = config or RNDConfig()
        self.device = th.device(device)
        self.input_dim = int(np.prod(observation_space.shape))
        self.stats = RNDRunningStats(
            obs_shape=(self.input_dim,),
            n_envs=n_envs,
            gamma_int=self.config.gamma_int,
            epsilon=self.config.epsilon,
            rms_epsilon=self.config.rms_epsilon,
            obs_clip=self.config.obs_clip,
            reward_clip=self.config.reward_clip,
            obs_norm=self.config.obs_norm,
            return_norm=self.config.return_norm,
        )
        self.model = RNDModel(
            input_dim=self.input_dim,
            feature_dim=self.config.feature_dim,
            target_net_arch=self.config.target_net_arch,
            predictor_net_arch=self.config.predictor_net_arch,
            activation=self.config.activation,
        ).to(self.device)
        self.optimizer = th.optim.Adam(
            self.model.predictor.parameters(),
            lr=self.config.predictor_learning_rate,
        )
        self._n_updates = 0

    def reset_returns(self, n_envs: int | None = None) -> None:
        self.stats.reset_returns(n_envs)

    @th.no_grad()
    def compute_intrinsic_reward(
        self,
        obs: Any,
        dones: np.ndarray,
        *,
        n_envs: int,
        training: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        flat = _obs_to_batch(obs, n_envs)
        rnd_obs = self.stats.normalize_obs(flat, update=training)
        x = th.as_tensor(rnd_obs, device=self.device, dtype=th.float32)
        r_raw = self.model.prediction_error(x).cpu().numpy().astype(np.float32)
        r_norm = self.stats.normalize_intrinsic_reward(r_raw, dones, update=training)
        return rnd_obs, r_raw, r_norm

    def combine_rewards(
        self,
        extrinsic: np.ndarray,
        intrinsic_norm: np.ndarray,
    ) -> np.ndarray:
        beta = self.config.intrinsic_reward_coef if self.config.use_rnd else 0.0
        return (
            np.asarray(extrinsic, dtype=np.float32)
            + float(beta) * np.asarray(intrinsic_norm, dtype=np.float32)
        )

    def train_predictor(self, rnd_obs_batch: th.Tensor) -> dict[str, float]:
        self.model.predictor.train()
        self.model.freeze_target()
        pred, tgt = self.model(rnd_obs_batch)
        per_sample = 0.5 * ((pred - tgt) ** 2).sum(dim=-1)
        loss = per_sample.mean()
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = float(
            nn.utils.clip_grad_norm_(
                self.model.predictor.parameters(),
                self.config.max_grad_norm,
            )
        )
        self.optimizer.step()
        self._n_updates += 1
        with th.no_grad():
            err = float(per_sample.mean().item())
            tgt_norm = float(tgt.norm(dim=-1).mean().item())
            pred_norm = float(pred.norm(dim=-1).mean().item())
        return {
            "predictor_loss": float(loss.item()),
            "predictor_grad_norm": grad_norm,
            "prediction_error_mean": err,
            "target_feature_norm": tgt_norm,
            "predictor_feature_norm": pred_norm,
        }

    def update_predictor_from_buffer(self, rnd_obs: np.ndarray) -> dict[str, float]:
        """Train predictor on buffered epoch transitions (RLeXplore-style update)."""
        if rnd_obs.size == 0 or not self.config.use_rnd:
            return {}
        n = rnd_obs.shape[0]
        keep = max(1, int(n * float(self.config.keep_proportion)))
        if keep < n:
            idx = np.random.choice(n, size=keep, replace=False)
            rnd_obs = rnd_obs[idx]
        batch_size = min(self.config.predictor_batch_size, rnd_obs.shape[0])
        n_steps = max(1, self.config.predictor_epochs * (rnd_obs.shape[0] // batch_size))
        metrics: dict[str, float] = {}
        for _ in range(n_steps):
            batch_inds = np.random.randint(0, rnd_obs.shape[0], size=batch_size)
            batch = th.as_tensor(rnd_obs[batch_inds], device=self.device, dtype=th.float32)
            metrics = self.train_predictor(batch)
        return metrics
