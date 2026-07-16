"""RND module: adapters, networks, stats, reward computation, predictor training."""

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
    """Owns target/predictor nets, running stats, and reward helpers."""

    def __init__(
        self,
        observation_space: spaces.Space,
        n_envs: int,
        config: RNDConfig | None = None,
        device: th.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.config = config or RNDConfig()
        self.device = th.device(device)
        self.adapter = RNDObsAdapter(observation_space, obs_key=self.config.obs_key)
        self.stats = RNDRunningStats(
            obs_shape=(self.adapter.input_dim,),
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
            input_dim=self.adapter.input_dim,
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

    def reset(self, n_envs: int | None = None) -> None:
        self.stats.reset_returns(n_envs)

    @th.no_grad()
    def compute_intrinsic_reward(
        self,
        obs: Any,
        dones: np.ndarray,
        training: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute raw and normalized intrinsic rewards from next observations.

        Returns:
            rnd_obs_norm: normalized flattened obs ``(n_envs, dim)``
            r_int_raw: raw prediction error ``(n_envs,)``
            r_int_norm: normalized intrinsic reward ``(n_envs,)``
        """
        flat = self.adapter.to_numpy(obs)
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
        return (np.asarray(extrinsic, dtype=np.float32)
                + float(beta) * np.asarray(intrinsic_norm, dtype=np.float32))

    def train_predictor(self, rnd_obs_batch: th.Tensor) -> dict[str, float]:
        """One Adam step on predictor. Target stays frozen."""
        self.model.predictor.train()
        self.model.freeze_target()
        loss = self.model.predictor_loss(rnd_obs_batch)
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
            pred, tgt = self.model(rnd_obs_batch)
            err = 0.5 * ((pred - tgt) ** 2).sum(dim=-1).mean().item()
        return {
            "predictor_loss": float(loss.item()),
            "predictor_grad_norm": grad_norm,
            "prediction_error_mean": float(err),
            "target_feature_norm": float(tgt.norm(dim=-1).mean().item()),
            "predictor_feature_norm": float(pred.norm(dim=-1).mean().item()),
        }

    def get_extra_state(self) -> dict[str, Any]:
        """Pickle-friendly extras (RMS, counters). Torch nets saved via state_dict."""
        return {
            "config": self.config.to_dict(),
            "stats": self.stats.get_state(),
            "n_updates": self._n_updates,
        }

    def set_extra_state(self, state: dict[str, Any]) -> None:
        if "config" in state:
            self.config = RNDConfig.from_dict(state["config"])
        if "stats" in state:
            self.stats.set_state(state["stats"])
        self._n_updates = int(state.get("n_updates", 0))
