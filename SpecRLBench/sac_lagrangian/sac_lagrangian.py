"""SAC-Lagrangian: SB3 SAC + OpenAI Qc actor term + EpCost dual (intentional).

Dual update matches PPO/TRPO EpCost softplus dual — **not** OpenAI SAC
``beta * (cost_constraint - qc)``. Actor still uses ``+ penalty * Qc``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, TypeVar

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import polyak_update
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

from lagrangian.cost import cost_from_info
from lagrangian.dual import LagPenalty
from lagrangian.metrics import LOG_COST_Q_LOSS, LOG_EP_COST, LOG_PENALTY, LOG_QC_PI
from lagrangian.off_policy.buffer import (
    LagDictReplayBuffer,
    LagDictReplayBufferSamples,
    LagReplayBuffer,
    LagReplayBufferSamples,
)
from lagrangian.off_policy.policy import LagMultiInputSACPolicy, LagSACPolicy
from lagrangian.on_policy.collect import EpCostTracker

SelfSACLag = TypeVar("SelfSACLag", bound="SACLag")


class SACLag(SAC):
    """SB3 SAC + twin Qc + EpCost Lag dual."""

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": LagSACPolicy,
        "MultiInputPolicy": LagMultiInputSACPolicy,
    }

    def __init__(
        self,
        policy: str | type[BasePolicy],
        env: GymEnv | str,
        learning_rate: float | Schedule = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int | tuple[int, str] = 1,
        gradient_steps: int = 1,
        action_noise: Any = None,
        replay_buffer_class: type[ReplayBuffer] | None = None,
        replay_buffer_kwargs: dict[str, Any] | None = None,
        optimize_memory_usage: bool = False,
        ent_coef: str | float = "auto",
        target_update_interval: int = 1,
        target_entropy: str | float = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "auto",
        _init_setup_model: bool = True,
        # --- Lag ---
        cost_lim: float = 25.0,
        penalty_init: float = 1.0,
        penalty_lr: float = 5e-2,
        cost_gamma: float = 0.99,  # unused (API parity); kept for shared kwargs tables
        cost_gae_lambda: float = 0.97,  # unused (API parity)
        cost_fn: Callable[[dict], float] | None = None,
    ):
        self.cost_lim = cost_lim
        self.penalty_init = penalty_init
        self.penalty_lr = penalty_lr
        self.cost_gamma = cost_gamma
        self.cost_gae_lambda = cost_gae_lambda
        self.cost_fn = cost_fn or cost_from_info

        self.lag_penalty: LagPenalty | None = None
        self.penalty_param: nn.Parameter | None = None
        self.penalty_optimizer: Adam | None = None
        self._ep_cost_tracker = EpCostTracker()
        self._last_ep_cost_mean = 0.0

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            ent_coef=ent_coef,
            target_update_interval=target_update_interval,
            target_entropy=target_entropy,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

    @property
    def penalty(self) -> th.Tensor:
        assert self.lag_penalty is not None
        return self.lag_penalty.penalty

    def _setup_model(self) -> None:
        if self.replay_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.replay_buffer_class = LagDictReplayBuffer
            else:
                self.replay_buffer_class = LagReplayBuffer

        super()._setup_model()

        self._ep_cost_tracker.reset_envs(self.n_envs)
        self.lag_penalty = LagPenalty(self.penalty_init, self.penalty_lr, self.device)
        self.penalty_param = self.lag_penalty.penalty_param
        self.penalty_optimizer = self.lag_penalty.optimizer
        assert isinstance(self.policy, (LagSACPolicy, LagMultiInputSACPolicy))

    def _excluded_save_params(self) -> list[str]:
        excluded = super()._excluded_save_params()
        excluded.extend(["penalty_optimizer", "lag_penalty", "_ep_cost_tracker"])
        return excluded

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts, torch_vars = super()._get_torch_save_params()
        state_dicts = list(state_dicts) + ["penalty_optimizer", "policy.cost_critic.optimizer"]
        torch_vars = list(torch_vars) + ["penalty_param"]
        return state_dicts, torch_vars

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: np.ndarray | dict[str, np.ndarray],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        # Ensure cost field for Lag replay (custom cost_fn supported)
        infos = [dict(info) for info in infos]
        costs = np.zeros(len(infos), dtype=np.float32)
        for i, info in enumerate(infos):
            c = float(self.cost_fn(info))
            infos[i]["cost"] = c
            costs[i] = c
        self._ep_cost_tracker.on_step(costs, dones)
        super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        assert self.lag_penalty is not None
        assert isinstance(self.policy, (LagSACPolicy, LagMultiInputSACPolicy))

        self.policy.set_training_mode(True)

        optimizers = [self.actor.optimizer, self.critic.optimizer, self.policy.cost_critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        # EpCost dual once per train() (intentional ≠ OpenAI Qc cost_constraint dual)
        self._last_ep_cost_mean = self._ep_cost_tracker.consume_mean()
        self.logger.record(LOG_EP_COST, self._last_ep_cost_mean)
        self.lag_penalty.update(self._last_ep_cost_mean, self.cost_lim)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses, cost_q_losses, qc_pis = [], [], [], []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            assert isinstance(replay_data, (LagReplayBufferSamples, LagDictReplayBufferSamples))
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

                # Qc target (no entropy); twin-mean aggregation
                next_qc = self.policy.cost_q_target_mean(replay_data.next_observations, next_actions)
                target_qc = replay_data.costs + (1 - replay_data.dones) * discounts * next_qc

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            current_qc_values = self.policy.cost_q_values(replay_data.observations, replay_data.actions)
            cost_q_loss = 0.5 * sum(F.mse_loss(qc, target_qc) for qc in current_qc_values)
            assert isinstance(cost_q_loss, th.Tensor)
            cost_q_losses.append(cost_q_loss.item())

            self.policy.cost_critic.optimizer.zero_grad()
            cost_q_loss.backward()
            self.policy.cost_critic.optimizer.step()

            # Actor: α logπ - min Qf + penalty * Qc  (OpenAI actor term; EpCost dual)
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            qc_pi = self.policy.cost_q_mean(replay_data.observations, actions_pi)
            pen = self.penalty.detach()
            actor_loss = (ent_coef * log_prob - min_qf_pi + pen * qc_pi).mean()
            actor_losses.append(actor_loss.item())
            qc_pis.append(float(qc_pi.mean().item()))

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                self.policy.soft_update_cost_critic(self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record(LOG_COST_Q_LOSS, np.mean(cost_q_losses) if cost_q_losses else 0.0)
        self.logger.record(LOG_QC_PI, np.mean(qc_pis) if qc_pis else 0.0)
        self.logger.record(LOG_PENALTY, float(self.penalty.item()))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def learn(
        self: SelfSACLag,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "SACLagrangian",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfSACLag:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
