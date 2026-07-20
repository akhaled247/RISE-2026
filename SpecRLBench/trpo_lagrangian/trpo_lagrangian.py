"""TRPO-Lagrangian: sb3_contrib TRPO + OpenAI objective-penalized Lag."""

from __future__ import annotations

import copy
from collections.abc import Callable
from functools import partial
from typing import Any, ClassVar, TypeVar

import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib import TRPO
from sb3_contrib.common.utils import conjugate_gradient_solver
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.distributions import kl_divergence
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

from lagrangian.cost import cost_from_info
from lagrangian.dual import LagPenalty
from lagrangian.metrics import LOG_COST_VALUE_LOSS, LOG_EP_COST, LOG_PENALTY, LOG_SURR_COST
from lagrangian.on_policy.buffer import (
    LagDictRolloutBuffer,
    LagDictRolloutBufferSamples,
    LagRolloutBuffer,
    LagRolloutBufferSamples,
)
from lagrangian.on_policy.collect import EpCostTracker
from lagrangian.on_policy.objective import openai_lag_pi_objective, unclipped_surr
from lagrangian.on_policy.policy import LagActorCriticPolicy, LagMultiInputActorCriticPolicy

SelfTRPOLag = TypeVar("SelfTRPOLag", bound="TRPOLag")


class TRPOLag(TRPO):
    """SB3-contrib TRPO + OpenAI Lag objective (EpCost dual, unclipped surr_cost)."""

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": LagActorCriticPolicy,
        "MultiInputPolicy": LagMultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: str | type[ActorCriticPolicy],
        env: GymEnv | str,
        learning_rate: float | Schedule = 1e-3,
        n_steps: int = 2048,
        batch_size: int = 128,
        gamma: float = 0.99,
        cg_max_steps: int = 15,
        cg_damping: float = 0.1,
        line_search_shrinking_factor: float = 0.8,
        line_search_max_iter: int = 10,
        n_critic_updates: int = 10,
        gae_lambda: float = 0.95,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: type[RolloutBuffer] | None = None,
        rollout_buffer_kwargs: dict[str, Any] | None = None,
        normalize_advantage: bool = False,
        target_kl: float = 0.01,
        sub_sampling_factor: int = 1,
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
        cost_gamma: float = 0.99,
        cost_gae_lambda: float = 0.97,
        cost_fn: Callable[[dict], float] | None = None,
        ent_coef: float = 0.0,
    ):
        self.cost_lim = cost_lim
        self.penalty_init = penalty_init
        self.penalty_lr = penalty_lr
        self.cost_gamma = cost_gamma
        self.cost_gae_lambda = cost_gae_lambda
        self.cost_fn = cost_fn or cost_from_info
        # Stored until after super(); TRPO forces ent_coef=0.0 in OnPolicyAlgorithm
        self._lag_ent_coef = ent_coef

        self.lag_penalty: LagPenalty | None = None
        self.penalty_param: nn.Parameter | None = None
        self.penalty_optimizer: Adam | None = None
        self._ep_cost_tracker = EpCostTracker()
        self._last_ep_cost_mean = 0.0

        buf_kwargs = dict(rollout_buffer_kwargs or {})
        buf_kwargs.setdefault("cost_gamma", cost_gamma)
        buf_kwargs.setdefault("cost_gae_lambda", cost_gae_lambda)
        # OpenAI-style once-per-rollout reward adv norm (like PPOLag openai mode)
        buf_kwargs.setdefault("normalize_advantage_once", True)
        # Buffer does once-norm; disable SB3 per-batch re-norm by default
        normalize_advantage = False

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            gamma=gamma,
            cg_max_steps=cg_max_steps,
            cg_damping=cg_damping,
            line_search_shrinking_factor=line_search_shrinking_factor,
            line_search_max_iter=line_search_max_iter,
            n_critic_updates=n_critic_updates,
            gae_lambda=gae_lambda,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=buf_kwargs,
            normalize_advantage=normalize_advantage,
            target_kl=target_kl,
            sub_sampling_factor=sub_sampling_factor,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )
        # Lag entropy term inside openai_lag_pi_objective (TRPO parent zeros ent_coef)
        self.ent_coef = self._lag_ent_coef

    @property
    def penalty(self) -> th.Tensor:
        assert self.lag_penalty is not None
        return self.lag_penalty.penalty

    def _setup_model(self) -> None:
        if self.rollout_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.rollout_buffer_class = LagDictRolloutBuffer
            else:
                self.rollout_buffer_class = LagRolloutBuffer

        super()._setup_model()

        self._ep_cost_tracker.reset_envs(self.n_envs)
        self.lag_penalty = LagPenalty(self.penalty_init, self.penalty_lr, self.device)
        self.penalty_param = self.lag_penalty.penalty_param
        self.penalty_optimizer = self.lag_penalty.optimizer
        assert isinstance(self.policy, (LagActorCriticPolicy, LagMultiInputActorCriticPolicy))

    def _excluded_save_params(self) -> list[str]:
        excluded = super()._excluded_save_params()
        excluded.extend(["penalty_optimizer", "lag_penalty", "_ep_cost_tracker"])
        return excluded

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts, torch_vars = super()._get_torch_save_params()
        state_dicts = list(state_dicts) + ["penalty_optimizer"]
        torch_vars = list(torch_vars) + ["penalty_param"]
        return state_dicts, torch_vars

    def _compute_actor_grad(
        self, kl_div: th.Tensor, policy_objective: th.Tensor
    ) -> tuple[list[nn.Parameter], th.Tensor, th.Tensor, list[tuple[int, ...]]]:
        """Like SB3 TRPO, but also skip cost critic params (``cost`` in name)."""
        policy_objective_gradients_list = []
        grad_kl_list = []
        grad_shape: list[tuple[int, ...]] = []
        actor_params: list[nn.Parameter] = []

        for name, param in self.policy.named_parameters():
            if "value" in name or "cost" in name:
                continue

            kl_param_grad, *_ = th.autograd.grad(
                kl_div,
                param,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
                only_inputs=True,
            )
            if kl_param_grad is not None:
                policy_objective_grad, *_ = th.autograd.grad(
                    policy_objective, param, retain_graph=True, only_inputs=True
                )
                grad_shape.append(kl_param_grad.shape)
                grad_kl_list.append(kl_param_grad.reshape(-1))
                policy_objective_gradients_list.append(policy_objective_grad.reshape(-1))
                actor_params.append(param)

        policy_objective_gradients = th.cat(policy_objective_gradients_list)
        grad_kl = th.cat(grad_kl_list)
        return actor_params, policy_objective_gradients, grad_kl, grad_shape

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        assert self._last_obs is not None
        assert isinstance(rollout_buffer, (LagRolloutBuffer, LagDictRolloutBuffer))
        assert isinstance(self.policy, (LagActorCriticPolicy, LagMultiInputActorCriticPolicy))
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        self._ep_cost_tracker.begin_window()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)  # type: ignore[arg-type]
                actions, values, log_probs = self.policy(obs_tensor)
                cost_values = self.policy.predict_cost_values(obs_tensor)
            actions = actions.cpu().numpy()

            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs

            costs = np.array([self.cost_fn(infos[i]) for i in range(env.num_envs)], dtype=np.float32)
            self._ep_cost_tracker.on_step(costs, dones)

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * float(terminal_value)

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
                cost=costs,
                cost_value=cost_values,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]
            last_cost_values = self.policy.predict_cost_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        rollout_buffer.compute_cost_returns_and_advantage(last_cost_values=last_cost_values, dones=dones)

        self._last_ep_cost_mean = self._ep_cost_tracker.finalize_mean()
        self.logger.record(LOG_EP_COST, self._last_ep_cost_mean)
        self.logger.record(LOG_PENALTY, float(self.penalty.item()))

        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    @staticmethod
    def _subsample_lag(
        data: LagRolloutBufferSamples | LagDictRolloutBufferSamples,
        factor: int,
    ) -> LagRolloutBufferSamples | LagDictRolloutBufferSamples:
        sl = slice(None, None, factor)
        if isinstance(data, LagDictRolloutBufferSamples):
            return LagDictRolloutBufferSamples(
                observations={k: v[sl] for k, v in data.observations.items()},
                actions=data.actions[sl],
                old_values=data.old_values[sl],
                old_log_prob=data.old_log_prob[sl],
                advantages=data.advantages[sl],
                returns=data.returns[sl],
                old_cost_values=data.old_cost_values[sl],
                cost_advantages=data.cost_advantages[sl],
                cost_returns=data.cost_returns[sl],
            )
        return LagRolloutBufferSamples(
            observations=data.observations[sl],
            actions=data.actions[sl],
            old_values=data.old_values[sl],
            old_log_prob=data.old_log_prob[sl],
            advantages=data.advantages[sl],
            returns=data.returns[sl],
            old_cost_values=data.old_cost_values[sl],
            cost_advantages=data.cost_advantages[sl],
            cost_returns=data.cost_returns[sl],
        )

    def train(self) -> None:
        assert self.lag_penalty is not None
        assert isinstance(self.policy, (LagActorCriticPolicy, LagMultiInputActorCriticPolicy))

        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        # EpCost dual once per train()
        self.lag_penalty.update(self._last_ep_cost_mean, self.cost_lim)

        policy_objective_values: list[float] = []
        kl_divergences: list[float] = []
        line_search_results: list[bool] = []
        value_losses: list[float] = []
        cost_value_losses: list[float] = []
        surr_costs: list[float] = []
        actor_params: list[nn.Parameter] = []

        for rollout_data in self.rollout_buffer.get(batch_size=None):
            assert isinstance(rollout_data, (LagRolloutBufferSamples, LagDictRolloutBufferSamples))
            if self.sub_sampling_factor > 1:
                rollout_data = self._subsample_lag(rollout_data, self.sub_sampling_factor)

            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                actions = rollout_data.actions.long().flatten()

            with th.no_grad():
                old_distribution = copy.copy(self.policy.get_distribution(rollout_data.observations))

            distribution = self.policy.get_distribution(rollout_data.observations)
            log_prob = distribution.log_prob(actions)

            advantages = rollout_data.advantages
            cost_advantages = rollout_data.cost_advantages
            if self.normalize_advantage and len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (rollout_data.advantages.std() + 1e-8)

            ratio = th.exp(log_prob - rollout_data.old_log_prob)
            surr_adv = unclipped_surr(advantages, ratio)
            surr_cost = unclipped_surr(cost_advantages, ratio)
            pen = self.penalty.detach()
            policy_objective = openai_lag_pi_objective(surr_adv, surr_cost, 0.0, pen, self.ent_coef)
            surr_costs.append(float(surr_cost.item()))

            kl_div = kl_divergence(distribution, old_distribution).mean()

            self.policy.optimizer.zero_grad()
            actor_params, policy_objective_gradients, grad_kl, grad_shape = self._compute_actor_grad(
                kl_div, policy_objective
            )

            hessian_vector_product_fn = partial(self.hessian_vector_product, actor_params, grad_kl)
            search_direction = conjugate_gradient_solver(
                hessian_vector_product_fn,
                policy_objective_gradients,
                max_iter=self.cg_max_steps,
            )

            line_search_max_step_size = 2 * self.target_kl
            line_search_max_step_size /= float(
                th.matmul(search_direction, hessian_vector_product_fn(search_direction, retain_graph=False))
            )
            line_search_max_step_size = np.sqrt(line_search_max_step_size)

            line_search_backtrack_coeff = 1.0
            original_actor_params = [param.detach().clone() for param in actor_params]

            is_line_search_success = False
            with th.no_grad():
                for _ in range(self.line_search_max_iter):
                    start_idx = 0
                    for param, original_param, shape in zip(
                        actor_params, original_actor_params, grad_shape, strict=True
                    ):
                        n_params = param.numel()
                        param.data = (
                            original_param.data
                            + line_search_backtrack_coeff
                            * line_search_max_step_size
                            * search_direction[start_idx : (start_idx + n_params)].view(shape)
                        )
                        start_idx += n_params

                    distribution = self.policy.get_distribution(rollout_data.observations)
                    log_prob = distribution.log_prob(actions)
                    ratio = th.exp(log_prob - rollout_data.old_log_prob)
                    new_surr_adv = unclipped_surr(advantages, ratio)
                    new_surr_cost = unclipped_surr(cost_advantages, ratio)
                    new_policy_objective = openai_lag_pi_objective(
                        new_surr_adv, new_surr_cost, 0.0, pen, self.ent_coef
                    )
                    kl_div = kl_divergence(distribution, old_distribution).mean()

                    if (kl_div < self.target_kl) and (new_policy_objective > policy_objective):
                        is_line_search_success = True
                        break

                    line_search_backtrack_coeff *= self.line_search_shrinking_factor

                line_search_results.append(is_line_search_success)

                if not is_line_search_success:
                    for param, original_param in zip(actor_params, original_actor_params, strict=True):
                        param.data = original_param.data.clone()
                    policy_objective_values.append(policy_objective.item())
                    kl_divergences.append(0.0)
                else:
                    policy_objective_values.append(new_policy_objective.item())
                    kl_divergences.append(kl_div.item())

        # Critic: V + Vc
        for _ in range(self.n_critic_updates):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                assert isinstance(rollout_data, (LagRolloutBufferSamples, LagDictRolloutBufferSamples))
                values_pred = self.policy.predict_values(rollout_data.observations)
                cost_values_pred = self.policy.predict_cost_values(rollout_data.observations)
                value_loss = F.mse_loss(rollout_data.returns, values_pred.flatten())
                cost_value_loss = F.mse_loss(rollout_data.cost_returns, cost_values_pred.flatten())
                value_losses.append(value_loss.item())
                cost_value_losses.append(cost_value_loss.item())

                self.policy.optimizer.zero_grad()
                (value_loss + cost_value_loss).backward()
                for param in actor_params:
                    param.grad = None
                self.policy.optimizer.step()

        self._n_updates += 1
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        self.logger.record("train/policy_objective", np.mean(policy_objective_values))
        self.logger.record("train/value_loss", np.mean(value_losses) if value_losses else 0.0)
        self.logger.record(LOG_COST_VALUE_LOSS, np.mean(cost_value_losses) if cost_value_losses else 0.0)
        self.logger.record(LOG_SURR_COST, np.mean(surr_costs) if surr_costs else 0.0)
        self.logger.record("train/kl_divergence_loss", np.mean(kl_divergences) if kl_divergences else 0.0)
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/is_line_search_success", np.mean(line_search_results) if line_search_results else 0.0)
        self.logger.record(LOG_PENALTY, float(self.penalty.item()))
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

    def learn(
        self: SelfTRPOLag,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "TRPOLagrangian",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfTRPOLag:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
