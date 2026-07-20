"""RND-augmented PPO built on Stable-Baselines3 PPO."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from rnd.config import RNDConfig
from rnd.module import RNDModule
from rnd.storage import RNDStorage


class RND(PPO):
    """PPO with Random Network Distillation intrinsic rewards.

    Intrinsic rewards are computed from ``s_{t+1}`` after each env step and
    added to extrinsic rewards before ``RolloutBuffer.add``. The value function
    therefore predicts combined returns. Monitor / ``rollout/ep_rew_mean`` still
    reflects extrinsic env rewards only.

    Set ``use_rnd=False`` or ``intrinsic_reward_coef=0`` for PPO-parity behavior
    (predictor still initializes but contributes zero reward).
    """

    def __init__(
        self,
        policy: str | type[ActorCriticPolicy],
        env: GymEnv | str,
        learning_rate: float | Schedule = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float | Schedule = 0.2,
        clip_range_vf: None | float | Schedule = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: type[RolloutBuffer] | None = None,
        rollout_buffer_kwargs: dict[str, Any] | None = None,
        target_kl: float | None = None,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "auto",
        _init_setup_model: bool = True,
        # --- RND ---
        rnd_config: RNDConfig | dict[str, Any] | None = None,
        use_rnd: bool = True,
        intrinsic_reward_coef: float | None = None,
    ):
        if isinstance(rnd_config, dict):
            self.rnd_config = RNDConfig.from_dict(rnd_config)
        elif isinstance(rnd_config, RNDConfig):
            self.rnd_config = rnd_config
        else:
            self.rnd_config = RNDConfig()

        self.rnd_config.use_rnd = use_rnd
        if intrinsic_reward_coef is not None:
            self.rnd_config.intrinsic_reward_coef = intrinsic_reward_coef

        self.rnd: RNDModule | None = None
        self.rnd_storage: RNDStorage | None = None
        self.rnd_extra_state: dict[str, Any] | None = None
        self._last_rnd_train_metrics: dict[str, float] = {}

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
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

    def _setup_model(self) -> None:
        super()._setup_model()
        self._setup_rnd()

    def _setup_rnd(self) -> None:
        assert self.env is not None
        self.rnd = RNDModule(
            observation_space=self.observation_space,
            n_envs=self.n_envs,
            config=self.rnd_config,
            device=self.device,
        )
        self.rnd_storage = RNDStorage(
            n_steps=self.n_steps,
            n_envs=self.n_envs,
            input_dim=self.rnd.adapter.input_dim,
            device=self.device,
        )
        # Restored from checkpoint pickle after __dict__.update during load()
        if self.rnd_extra_state is not None:
            self.rnd.set_extra_state(self.rnd_extra_state)

    def _excluded_save_params(self) -> list[str]:
        excluded = super()._excluded_save_params()
        # Torch modules / optimizers saved via state_dict; storage is transient
        excluded.extend(["rnd", "rnd_storage", "_last_rnd_train_metrics"])
        return excluded

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts, torch_vars = super()._get_torch_save_params()
        # Save RND model + predictor optimizer as state dicts
        state_dicts = list(state_dicts) + ["rnd.model", "rnd.optimizer"]
        return state_dicts, torch_vars

    def save(self, path, exclude=None, include=None) -> None:  # type: ignore[override]
        # Persist RND extra state (RMS, counters, config) into picklable attrs
        if self.rnd is not None:
            self.rnd_extra_state = self.rnd.get_extra_state()
            self.rnd_config = self.rnd.config
        super().save(path, exclude=exclude, include=include)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"
        assert self.rnd is not None and self.rnd_storage is not None
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        self.rnd_storage.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)  # type: ignore[arg-type]
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            extrinsic_rewards = np.asarray(rewards, dtype=np.float32).copy()

            self.num_timesteps += env.num_envs

            # RND intrinsic reward from next observation
            if self.rnd_config.use_rnd and self.rnd_config.intrinsic_reward_coef != 0.0:
                rnd_obs, r_int_raw, r_int_norm = self.rnd.compute_intrinsic_reward(
                    new_obs, dones, training=True
                )
                combined_rewards = self.rnd.combine_rewards(extrinsic_rewards, r_int_norm)
            else:
                # Still advance RND storage shapes; zero intrinsic
                flat = self.rnd.adapter.to_numpy(new_obs)
                rnd_obs = self.rnd.stats.normalize_obs(flat, update=False)
                r_int_raw = np.zeros(env.num_envs, dtype=np.float32)
                r_int_norm = np.zeros(env.num_envs, dtype=np.float32)
                combined_rewards = extrinsic_rewards.copy()

            intrinsic_rewards = r_int_norm
            rewards = combined_rewards  # buffer + timeout bootstrap use combined

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            # Timeout bootstrap on combined reward (value predicts total return)
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self.rnd_storage.add(
                rnd_obs=rnd_obs,
                raw_intrinsic=r_int_raw,
                norm_intrinsic=r_int_norm,
                extrinsic=extrinsic_rewards,
                combined=rewards,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        # Log RND rollout diagnostics (extrinsic Monitor metrics untouched)
        summary = self.rnd_storage.summary()
        self.logger.record("rnd/reward_raw_mean", summary["raw_mean"])
        self.logger.record("rnd/reward_raw_std", summary["raw_std"])
        self.logger.record("rnd/reward_raw_min", summary["raw_min"])
        self.logger.record("rnd/reward_raw_max", summary["raw_max"])
        self.logger.record("rnd/reward_norm_mean", summary["norm_mean"])
        self.logger.record("rnd/reward_norm_std", summary["norm_std"])
        self.logger.record("rnd/reward_norm_min", summary["norm_min"])
        self.logger.record("rnd/reward_norm_max", summary["norm_max"])
        self.logger.record("rnd/reward_coef", float(self.rnd_config.intrinsic_reward_coef))
        self.logger.record("rnd/reward_contribution_mean", summary["contribution_mean"])
        self.logger.record("rollout/extrinsic_reward_mean", summary["extrinsic_mean"])
        self.logger.record("rollout/combined_reward_mean", summary["combined_mean"])
        self.logger.record("rollout/intrinsic_reward_mean", summary["norm_mean"])
        self.logger.record("rnd/obs_rms_mean_abs", float(np.mean(np.abs(self.rnd.stats.obs_rms.mean))))
        self.logger.record("rnd/obs_rms_var_mean", float(np.mean(self.rnd.stats.obs_rms.var)))
        self.logger.record("rnd/int_return_rms_var", float(self.rnd.stats.int_ret_rms.var))
        self.logger.record("rnd/int_return_rms_count", float(self.rnd.stats.int_ret_rms.count))
        self.logger.record("rnd/int_return_mean", float(np.mean(self.rnd.stats.int_returns)))
        self.logger.record("rnd/int_return_std", float(np.std(self.rnd.stats.int_returns)))

        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    def train(self) -> None:
        """SB3 PPO update, then RND predictor updates. SB3 owns train/* logs."""
        assert self.rnd is not None and self.rnd_storage is not None

        super().train()

        if not self.rnd_config.use_rnd:
            return

        # Full n_epochs * n_minibatches even if SB3 early-stopped on target_kl.
        n_minibatches = (self.n_steps * self.n_envs) // self.batch_size
        n_rnd_steps = self.n_epochs * n_minibatches
        total = self.n_steps * self.n_envs
        batch = self.batch_size

        rnd_losses: list[float] = []
        rnd_grad_norms: list[float] = []
        rnd_errors: list[float] = []

        for _ in range(n_rnd_steps):
            batch_inds = np.random.randint(0, total, size=batch)
            rnd_batch = self.rnd_storage.get_rnd_obs_batch(batch_inds)
            metrics = self.rnd.train_predictor(rnd_batch)
            rnd_losses.append(metrics["predictor_loss"])
            rnd_grad_norms.append(metrics["predictor_grad_norm"])
            rnd_errors.append(metrics["prediction_error_mean"])
            self._last_rnd_train_metrics = metrics

        if rnd_losses:
            self.logger.record("rnd/predictor_loss", float(np.mean(rnd_losses)))
            self.logger.record("rnd/predictor_grad_norm", float(np.mean(rnd_grad_norms)))
            self.logger.record("rnd/prediction_error_mean", float(np.mean(rnd_errors)))
            self.logger.record("train/rnd_learning_rate", float(self.rnd_config.predictor_learning_rate))
            self.logger.record("train/rnd_updates", int(self.rnd._n_updates))
            if self._last_rnd_train_metrics:
                self.logger.record(
                    "rnd/target_feature_norm",
                    self._last_rnd_train_metrics.get("target_feature_norm", 0.0),
                )
                self.logger.record(
                    "rnd/predictor_feature_norm",
                    self._last_rnd_train_metrics.get("predictor_feature_norm", 0.0),
                )
