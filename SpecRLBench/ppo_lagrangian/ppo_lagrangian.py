"""PPO-Lagrangian as SB3 PPO subclass (Tier 2 openai parity + Tier 3 hook).

objective_penalized=True, learn_penalty=True, penalty_param_loss=True.
SB3-style minibatch PPO updates + OpenAI objective-penalized Lagrangian.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, Literal, TypeVar

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

from ppo_lagrangian.buffer import LagDictRolloutBuffer, LagRolloutBuffer
from ppo_lagrangian.policy import LagActorCriticPolicy, LagMultiInputActorCriticPolicy

SelfPPOLagrangian = TypeVar("SelfPPOLagrangian", bound="PPOLagrangian")


def _cost_from_info(info: dict) -> float:
    """OpenAI: info.get('cost', 0)."""
    return float(info.get("cost", 0))


class PPOLagrangian(PPO):
    """SB3 PPO + OpenAI objective-penalized Lagrangian (dual critics).

    ``lag_mode="openai"`` (default, Tier 2): once-per-rollout adv norm, dual
    pi/vf optimizers, entropy inside Lag objective only.

    ``lag_mode="sb3"`` (Tier 3): SB3 per-minibatch adv norm, single optimizer,
    ``vf_coef * (v_loss + vc_loss)`` packed with Lag policy loss.
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": LagActorCriticPolicy,
        "MultiInputPolicy": LagMultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: str | type[ActorCriticPolicy],
        env: GymEnv | str,
        learning_rate: float | Schedule = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.97,
        clip_range: float | Schedule = 0.2,
        clip_range_vf: None | float | Schedule = None,
        normalize_advantage: bool = False,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: type[RolloutBuffer] | None = None,
        rollout_buffer_kwargs: dict[str, Any] | None = None,
        target_kl: float | None = 0.01,
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
        vf_lr: float = 1e-3,
        cost_fn: Callable[[dict], float] | None = None,
        lag_mode: Literal["openai", "sb3"] = "openai",
    ):
        self.cost_lim = cost_lim
        self.penalty_init = penalty_init
        self.penalty_lr = penalty_lr
        self.cost_gamma = cost_gamma
        self.cost_gae_lambda = cost_gae_lambda
        self.vf_lr = vf_lr
        self.cost_fn = cost_fn or _cost_from_info
        self.lag_mode = lag_mode

        # Tier 2: buffer does once-norm; Tier 3: SB3 per-mb normalize_advantage
        if lag_mode == "openai":
            normalize_advantage = False
        elif lag_mode == "sb3":
            normalize_advantage = True

        self.penalty_param: nn.Parameter | None = None
        self.penalty_optimizer: Adam | None = None
        self.pi_optimizer: Adam | None = None
        self.vf_optimizer: Adam | None = None
        self._rollout_ep_costs: list[float] = []
        self._ep_cost = np.zeros(1, dtype=np.float64)
        self._last_ep_cost_mean = 0.0

        buf_kwargs = dict(rollout_buffer_kwargs or {})
        buf_kwargs.setdefault("cost_gamma", cost_gamma)
        buf_kwargs.setdefault("cost_gae_lambda", cost_gae_lambda)
        buf_kwargs.setdefault("normalize_advantage_once", lag_mode == "openai")

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=buf_kwargs,
            target_kl=target_kl,
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
        assert self.penalty_param is not None
        return th.nn.functional.softplus(self.penalty_param)

    def _setup_model(self) -> None:
        # Choose Lag buffer by obs space (same rule as SB3 OnPolicyAlgorithm)
        if self.rollout_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.rollout_buffer_class = LagDictRolloutBuffer
            else:
                self.rollout_buffer_class = LagRolloutBuffer

        super()._setup_model()

        n_envs = self.n_envs
        self._ep_cost = np.zeros(n_envs, dtype=np.float64)

        param_init = float(np.log(max(np.exp(self.penalty_init) - 1.0, 1e-8)))
        self.penalty_param = nn.Parameter(th.tensor(param_init, dtype=th.float32, device=self.device))
        self.penalty_optimizer = Adam([self.penalty_param], lr=self.penalty_lr)

        assert isinstance(self.policy, (LagActorCriticPolicy, LagMultiInputActorCriticPolicy))
        if self.lag_mode == "openai":
            self.pi_optimizer = Adam(self.policy.pi_parameters(), lr=self.lr_schedule(1))
            self.vf_optimizer = Adam(self.policy.vf_parameters(), lr=self.vf_lr)

    def _excluded_save_params(self) -> list[str]:
        excluded = super()._excluded_save_params()
        excluded.extend(["pi_optimizer", "vf_optimizer", "penalty_optimizer", "_rollout_ep_costs"])
        return excluded

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts, torch_vars = super()._get_torch_save_params()
        state_dicts = list(state_dicts)
        # penalty_param is an nn.Parameter; save via named attr
        torch_vars = list(torch_vars) + ["penalty_param"]
        if self.lag_mode == "openai":
            state_dicts.extend(["pi_optimizer", "vf_optimizer", "penalty_optimizer"])
        else:
            state_dicts.append("penalty_optimizer")
        return state_dicts, torch_vars

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
        self._rollout_ep_costs = []
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
            self._ep_cost += costs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            # Timeout bootstrap reward + cost
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                        terminal_cvalue = self.policy.predict_cost_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * float(terminal_value)
                    # Cost bootstrap on truncate (keep array for buffer; GAE uses last_cost_values at end)
                    _ = terminal_cvalue
                if done:
                    self._rollout_ep_costs.append(float(self._ep_cost[idx]))
                    self._ep_cost[idx] = 0.0

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

        self._last_ep_cost_mean = (
            float(np.mean(self._rollout_ep_costs)) if self._rollout_ep_costs else 0.0
        )
        self.logger.record("rollout/ep_cost_mean", self._last_ep_cost_mean)
        self.logger.record("train/penalty", float(self.penalty.item()))

        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    def train(self) -> None:
        # SB3-style minibatch PPO updates + OpenAI objective-penalized Lagrangian.
        assert self.penalty_param is not None and self.penalty_optimizer is not None
        assert isinstance(self.policy, (LagActorCriticPolicy, LagMultiInputActorCriticPolicy))

        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        if self.lag_mode == "openai" and self.pi_optimizer is not None:
            # Keep pi LR in sync with schedule
            for pg in self.pi_optimizer.param_groups:
                pg["lr"] = self.lr_schedule(self._current_progress_remaining)

        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]

        # Penalty update once per rollout (before policy/value minibatch training)
        self.penalty_optimizer.zero_grad()
        (-self.penalty_param * (self._last_ep_cost_mean - self.cost_lim)).backward()
        self.penalty_optimizer.step()

        pg_losses: list[float] = []
        value_losses: list[float] = []
        cost_value_losses: list[float] = []
        entropy_losses: list[float] = []
        approx_kl_divs: list[float] = []
        clip_fractions: list[float] = []
        surr_costs: list[float] = []
        last_loss = 0.0
        continue_training = True

        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()

                values, cost_values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()
                cost_values = cost_values.flatten()

                advantages = rollout_data.advantages
                cost_advantages = rollout_data.cost_advantages
                # Tier 3: SB3 per-minibatch reward adv re-norm
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                # Clipped surrogate (same form as OpenAI / SB3)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                surr_adv = th.min(policy_loss_1, policy_loss_2).mean()
                surr_cost = (ratio * cost_advantages).mean()
                pen = self.penalty.detach()

                if entropy is None:
                    ent = -log_prob.mean()
                else:
                    ent = entropy.mean()
                # Entropy only inside Lag objective (not a second SB3 entropy term)
                pi_obj = (surr_adv + self.ent_coef * ent - pen * surr_cost) / (1.0 + pen)
                policy_loss = -pi_obj
                entropy_loss = float((-ent).item())

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl = th.mean((th.exp(log_ratio) - 1) - log_ratio).item()
                    approx_kl_divs.append(approx_kl)
                    clip_fractions.append(th.mean((th.abs(ratio - 1) > clip_range).float()).item())

                if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at epoch {epoch} due to reaching max kl: {approx_kl:.2f}")
                    break

                v_loss = F.mse_loss(rollout_data.returns, values)
                vc_loss = F.mse_loss(rollout_data.cost_returns, cost_values)

                if self.lag_mode == "openai":
                    assert self.pi_optimizer is not None and self.vf_optimizer is not None
                    self.pi_optimizer.zero_grad()
                    policy_loss.backward()
                    th.nn.utils.clip_grad_norm_(self.policy.pi_parameters(), self.max_grad_norm)
                    self.pi_optimizer.step()

                    # Fresh value forward (separate optimizer; avoid shared-graph with pi)
                    values2 = self.policy.predict_values(rollout_data.observations).flatten()
                    cost_values2 = self.policy.predict_cost_values(rollout_data.observations).flatten()
                    v_loss = F.mse_loss(rollout_data.returns, values2)
                    vc_loss = F.mse_loss(rollout_data.cost_returns, cost_values2)
                    self.vf_optimizer.zero_grad()
                    (v_loss + vc_loss).backward()
                    th.nn.utils.clip_grad_norm_(self.policy.vf_parameters(), self.max_grad_norm)
                    self.vf_optimizer.step()
                    last_loss = float(policy_loss.item() + v_loss.item() + vc_loss.item())
                else:
                    # Tier 3: single optimizer, packed loss
                    loss = policy_loss + self.vf_coef * (v_loss + vc_loss)
                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()
                    last_loss = float(loss.item())

                pg_losses.append(float(policy_loss.item()))
                value_losses.append(float(v_loss.item()))
                cost_value_losses.append(float(vc_loss.item()))
                entropy_losses.append(entropy_loss)
                surr_costs.append(float(surr_cost.item()))

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )
        self.logger.record("train/entropy_loss", np.mean(entropy_losses) if entropy_losses else 0.0)
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses) if pg_losses else 0.0)
        self.logger.record("train/value_loss", np.mean(value_losses) if value_losses else 0.0)
        self.logger.record("train/cost_value_loss", np.mean(cost_value_losses) if cost_value_losses else 0.0)
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs) if approx_kl_divs else 0.0)
        self.logger.record("train/clip_fraction", np.mean(clip_fractions) if clip_fractions else 0.0)
        self.logger.record("train/surr_cost", np.mean(surr_costs) if surr_costs else 0.0)
        self.logger.record("train/loss", last_loss)
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/penalty", float(self.penalty.item()))
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)

    def learn(
        self: SelfPPOLagrangian,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPOLagrangian",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfPPOLagrangian:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    @classmethod
    def load(  # type: ignore[override]
        cls,
        path: str | Path,
        env: GymEnv | None = None,
        device: th.device | str = "auto",
        custom_objects: dict[str, Any] | None = None,
        force_reset: bool = True,
        **kwargs: Any,
    ) -> PPOLagrangian:
        path = Path(path)
        # Legacy .pt unsupported after Tier-2 SB3 refactor
        if path.suffix in {".pt", ".pth"} or (
            not path.exists()
            and not path.with_suffix(".zip").exists()
            and Path(str(path) + ".pt").exists()
        ):
            raise ValueError(
                "Legacy PPOLagrangian .pt checkpoints are unsupported after the SB3 Tier-2 "
                "refactor. Retrain and save as SB3 .zip (PPOLagrangian.save)."
            )
        return super().load(  # type: ignore[return-value]
            path=path,
            env=env,
            device=device,
            custom_objects=custom_objects,
            force_reset=force_reset,
            **kwargs,
        )
