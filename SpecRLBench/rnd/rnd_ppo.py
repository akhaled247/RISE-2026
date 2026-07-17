"""OpenAI-faithful PPO + RND with SB3-like public API."""

from __future__ import annotations

import os
import time
import warnings
from collections import deque
from pathlib import Path
from typing import Any, Callable, ClassVar, TypeVar

import numpy as np
import torch as th
from gymnasium import spaces
from rnd.config import RNDConfig
from rnd.module import RNDModule
from rnd.networks import DualValuePolicy
from rnd.obs_adapter import RNDObsAdapter
from rnd.storage import InteractionStorage

try:
    from stable_baselines3.common.logger import configure as sb3_configure
    from stable_baselines3.common.save_util import (
        load_from_zip_file,
        recursive_getattr,
        recursive_setattr,
        save_to_zip_file,
    )
    from stable_baselines3.common.utils import get_device, set_random_seed
    from stable_baselines3.common.vec_env import VecEnv, VecNormalize
except ImportError:  # pragma: no cover
    sb3_configure = None  # type: ignore[assignment]
    save_to_zip_file = None  # type: ignore[assignment]
    load_from_zip_file = None  # type: ignore[assignment]
    recursive_getattr = None  # type: ignore[assignment]
    recursive_setattr = None  # type: ignore[assignment]
    get_device = None  # type: ignore[assignment]
    set_random_seed = None  # type: ignore[assignment]
    VecEnv = Any  # type: ignore[misc, assignment]
    VecNormalize = Any  # type: ignore[misc, assignment]

SelfPPORND = TypeVar("SelfPPORND", bound="PPORND")


def _explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    vary = np.var(y_true)
    if vary == 0:
        return float("nan")
    return float(1.0 - np.var(y_true - y_pred) / vary)


def _obs_to_storage_array(obs: Any, adapter: RNDObsAdapter, is_image: bool) -> np.ndarray:
    if is_image:
        arr = np.asarray(obs, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[None, ...]
        return arr
    return adapter.to_numpy(obs)


class PPORND:
    """PPO + Random Network Distillation (OpenAI behavioral fidelity).

    Public surface mirrors Stable-Baselines3 ``PPO`` enough for existing scripts::

        model = PPORND("MultiInputPolicy", env, ...)
        model.learn(total_timesteps=...)
        model.predict(obs)
        model.save(path)
        model = PPORND.load(path, env=env)

    Internals follow ``openai/random-network-distillation``:
    dual value heads, separate int/ext GAE, combined advantages,
    RND aux loss in the same optimizer, env-slice minibatches.
    """

    policy_aliases: ClassVar[dict[str, str]] = {
        "MlpPolicy": "mlp",
        "CnnPolicy": "cnn",
        "MultiInputPolicy": "multi",
    }

    def __init__(
        self,
        policy: str | type,
        env: Any,
        learning_rate: float | Callable[[float], float] = 1e-4,
        n_steps: int = 128,
        batch_size: int = 256,
        n_epochs: int = 4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float | Callable[[float], float] = 0.1,
        clip_range_vf: None | float = None,
        normalize_advantage: bool = False,
        ent_coef: float = 0.001,
        vf_coef: float = 1.0,
        max_grad_norm: float | None = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: float | None = None,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "auto",
        _init_setup_model: bool = True,
        # RND
        rnd_config: RNDConfig | dict[str, Any] | None = None,
        use_rnd: bool = True,
        intrinsic_reward_coef: float | None = None,
        # OpenAI extras exposed optionally
        gamma_ext: float | None = None,
        int_coeff: float | None = None,
        ext_coeff: float | None = None,
        nminibatches: int | None = None,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            warnings.warn(f"Ignoring unsupported kwargs: {sorted(kwargs)}", stacklevel=2)

        if isinstance(rnd_config, dict):
            self.rnd_config = RNDConfig.from_dict(rnd_config)
        elif isinstance(rnd_config, RNDConfig):
            self.rnd_config = rnd_config.resolve()
        else:
            self.rnd_config = RNDConfig().resolve()

        self.rnd_config.use_rnd = use_rnd
        if intrinsic_reward_coef is not None:
            self.rnd_config.intrinsic_reward_coef = intrinsic_reward_coef
            self.rnd_config.int_coeff = float(intrinsic_reward_coef)
        if int_coeff is not None:
            self.rnd_config.int_coeff = float(int_coeff)
        if ext_coeff is not None:
            self.rnd_config.ext_coeff = float(ext_coeff)
        if gamma_ext is not None:
            self.rnd_config.gamma_ext = float(gamma_ext)
        self.rnd_config.gamma = float(gamma)
        self.rnd_config.lam = float(gae_lambda)
        if nminibatches is not None:
            self.rnd_config.nminibatches = int(nminibatches)
        self.rnd_config.resolve()

        self.policy_name = policy if isinstance(policy, str) else getattr(policy, "__name__", "custom")
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.n_envs = int(getattr(env, "num_envs", 1))
        self.learning_rate = learning_rate
        self.n_steps = int(n_steps)
        self.batch_size = int(batch_size)
        self.n_epochs = int(n_epochs)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = bool(normalize_advantage)
        self.ent_coef = float(ent_coef)
        self.vf_coef = float(vf_coef)
        # OpenAI run_atari uses max_grad_norm=0.0 → no clipping
        self.max_grad_norm = (
            float(self.rnd_config.max_grad_norm)
            if max_grad_norm is None
            else float(max_grad_norm)
        )
        self.use_sde = use_sde
        self.sde_sample_freq = sde_sample_freq
        self.target_kl = target_kl
        self.tensorboard_log = tensorboard_log
        self.policy_kwargs = policy_kwargs or {}
        self.verbose = verbose
        self.seed = seed
        self.device = get_device(device) if get_device is not None else th.device(
            "cuda" if th.cuda.is_available() and device != "cpu" else "cpu"
        )

        self.num_timesteps = 0
        self._n_updates = 0
        self._current_progress_remaining = 1.0
        self._episode_rewards: deque[float] = deque(maxlen=100)
        self._episode_lengths: deque[float] = deque(maxlen=100)
        self.ep_info_buffer: deque[dict[str, Any]] = deque(maxlen=100)
        self._last_obs: Any = None
        self._last_episode_starts: np.ndarray | None = None
        self._prev_rews: np.ndarray | None = None

        self.policy: DualValuePolicy | None = None
        self.rnd: RNDModule | None = None
        self.storage: InteractionStorage | None = None
        self.optimizer: th.optim.Optimizer | None = None
        self.logger: Any = None
        self.rnd_extra_state: dict[str, Any] | None = None

        # Map batch_size → nminibatches (OpenAI env slices)
        total = self.n_envs * self.n_steps
        if self.rnd_config.nminibatches is None:
            nm = max(1, total // max(self.batch_size, 1))
            # Prefer divisor of n_envs
            while self.n_envs % nm != 0 and nm > 1:
                nm -= 1
            if self.n_envs % nm != 0:
                nm = 1
                warnings.warn(
                    f"batch_size={self.batch_size} incompatible with n_envs={self.n_envs}; "
                    f"using nminibatches=1 (OpenAI env-slice constraint).",
                    stacklevel=2,
                )
            self.nminibatches = nm
        else:
            self.nminibatches = int(self.rnd_config.nminibatches)
            if self.n_envs % self.nminibatches != 0:
                raise ValueError(
                    f"n_envs={self.n_envs} must be divisible by nminibatches={self.nminibatches}"
                )

        if seed is not None and set_random_seed is not None:
            set_random_seed(seed)

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        adapter = RNDObsAdapter(
            self.observation_space,
            obs_key=self.rnd_config.obs_key,
            obs_keys=self.rnd_config.obs_keys,
        )
        self.policy = DualValuePolicy(
            observation_space=self.observation_space,
            action_space=self.action_space,
            adapter=adapter,
            **{k: v for k, v in self.policy_kwargs.items() if k in ("features_dim", "net_arch", "log_std_init")},
        ).to(self.device)

        self.rnd = RNDModule(
            observation_space=self.observation_space,
            n_envs=self.n_envs,
            config=self.rnd_config,
            device=self.device,
        )
        if self.rnd_extra_state is not None:
            self.rnd.set_extra_state(self.rnd_extra_state)

        is_image = self.policy.is_image
        obs_shape = (
            self.observation_space.shape
            if is_image
            else (adapter.input_dim,)
        )
        discrete = isinstance(self.action_space, spaces.Discrete)
        action_dim = 1 if discrete else int(np.prod(self.action_space.shape))
        self.storage = InteractionStorage(
            n_envs=self.n_envs,
            n_steps=self.n_steps,
            obs_shape=tuple(obs_shape),
            action_dim=action_dim,
            discrete=discrete,
            device=self.device,
        )

        # One optimizer: policy + predictor (target frozen)
        params = list(self.policy.parameters()) + list(self.rnd.model.predictor.parameters())
        lr = self.learning_rate if isinstance(self.learning_rate, float) else self.learning_rate(1.0)
        self.optimizer = th.optim.Adam(params, lr=lr)

        if self.tensorboard_log and sb3_configure is not None:
            self.logger = sb3_configure(self.tensorboard_log, ["stdout", "tensorboard"])
        elif sb3_configure is not None:
            self.logger = sb3_configure(None, ["stdout"] if self.verbose else [])
        else:
            self.logger = _SimpleLogger(self.verbose)

    def _lr(self) -> float:
        if callable(self.learning_rate):
            return float(self.learning_rate(self._current_progress_remaining))
        return float(self.learning_rate)

    def _cliprange(self) -> float:
        if callable(self.clip_range):
            return float(self.clip_range(self._current_progress_remaining))
        return float(self.clip_range)

    def _maybe_warmup_obs_stats(self) -> None:
        assert self.rnd is not None and self.env is not None
        if not self.rnd_config.update_ob_stats_from_random_agent:
            return
        if not isinstance(self.action_space, spaces.Discrete):
            # OpenAI warmup uses randint; skip for continuous (intentional difference)
            return
        steps = int(self.rnd_config.random_obs_init_steps)
        if steps <= 0:
            return
        obs = self.env.reset()
        buf: list[np.ndarray] = []
        for step in range(steps):
            acs = np.random.randint(0, self.action_space.n, size=(self.n_envs,))
            obs, _, dones, _ = self.env.step(acs)
            if self.rnd.image_shape is not None:
                arr = np.asarray(obs, dtype=np.float32)
                if arr.ndim == 4 and arr.shape[-1] >= 1:
                    buf.append(arr[..., -1:])
            else:
                buf.append(self.rnd.adapter.to_numpy(obs))
            if len(buf) >= 128:
                stacked = np.concatenate([b.reshape(-1, *b.shape[1:]) for b in buf], axis=0)
                if self.rnd.image_shape is not None:
                    self.rnd.stats.update_obs_rms(stacked)
                else:
                    self.rnd.stats.update_obs_rms(stacked)
                buf.clear()
        if buf:
            stacked = np.concatenate([b.reshape(-1, *b.shape[1:]) for b in buf], axis=0)
            self.rnd.stats.update_obs_rms(stacked)

    def collect_rollouts(self) -> dict[str, float]:
        assert self.policy is not None and self.rnd is not None and self.storage is not None
        self.policy.eval()
        self.storage.reset()

        if self._last_obs is None:
            self._last_obs = self.env.reset()
            self._last_episode_starts = np.ones(self.n_envs, dtype=np.float32)
            self._prev_rews = None
            self._maybe_warmup_obs_stats()
            # refresh after warmup
            self._last_obs = self.env.reset()

        adapter = self.policy.adapter
        is_image = self.policy.is_image
        news = self._last_episode_starts if self._last_episode_starts is not None else np.ones(
            self.n_envs, dtype=np.float32
        )

        for t in range(self.n_steps):
            obs = self._last_obs
            obs_arr = _obs_to_storage_array(obs, adapter, is_image)

            with th.no_grad():
                actions_t, v_int, v_ext, nlp, ent = self.policy(obs, deterministic=False)

            actions = actions_t.cpu().numpy()
            if isinstance(self.action_space, spaces.Box):
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

            # OpenAI stores news for *current* obs before step; prev reward into t-1
            rews_prev = self._prev_rews
            self.storage.add_step(
                t=t,
                obs=obs_arr,
                actions=actions if not isinstance(self.action_space, spaces.Discrete) else actions.astype(np.int64),
                neglogp=nlp.cpu().numpy(),
                entropy=ent.cpu().numpy(),
                vpred_int=v_int.cpu().numpy(),
                vpred_ext=v_ext.cpu().numpy(),
                news=np.asarray(news, dtype=np.float32),
                rews_ext_prev=rews_prev,
            )

            new_obs, rewards, dones, infos = self.env.step(actions)
            self.num_timesteps += self.n_envs
            rewards = np.asarray(rewards, dtype=np.float32)
            dones = np.asarray(dones, dtype=np.float32)

            # Episode stats from Monitor infos
            for info in infos:
                if "episode" in info:
                    ep = info["episode"]
                    self._episode_rewards.append(float(ep["r"]))
                    self._episode_lengths.append(float(ep["l"]))
                    self.ep_info_buffer.append(ep)

            self._prev_rews = rewards
            self._last_obs = new_obs
            news = dones
            self._last_episode_starts = dones

            if self.rnd_config.update_ob_stats_every_step and self.rnd.image_shape is not None:
                arr = np.asarray(new_obs, dtype=np.float32)
                if arr.ndim == 4 and arr.shape[-1] >= 1:
                    self.rnd.stats.update_obs_rms(arr[..., -1:])
            elif self.rnd_config.update_ob_stats_every_step:
                self.rnd.stats.update_obs_rms(adapter.to_numpy(new_obs))

        # Bootstrap extra step (OpenAI)
        ob_last = _obs_to_storage_array(self._last_obs, adapter, is_image)
        with th.no_grad():
            v_int_last, v_ext_last = self.policy.predict_values(self._last_obs)
        assert self._prev_rews is not None
        self.storage.set_bootstrap(
            ob_last=ob_last,
            new_last=news,
            vpred_int_last=v_int_last.cpu().numpy(),
            vpred_ext_last=v_ext_last.cpu().numpy(),
            rews_ext_last=self._prev_rews,
        )

        # Obs RMS update after rollout (OpenAI default update_ob_stats_every_step=0)
        if not self.rnd_config.update_ob_stats_every_step:
            if self.rnd.image_shape is not None:
                obs_ = self.storage.buf_obs.astype(np.float32)
                # last channel
                if obs_.shape[-1] >= 1:
                    self.rnd.stats.update_obs_rms(obs_.reshape(-1, *obs_.shape[2:])[..., -1:])
            else:
                self.rnd.stats.update_obs_rms(self.storage.buf_obs.reshape(-1, self.storage.obs_shape[0]))

        # Intrinsic rewards from next observations
        if self.rnd_config.use_rnd:
            next_obs = self.storage.next_obs_for_rnd()
            rews_int_raw = self.rnd.compute_intrinsic_rewards(next_obs, update_rms=False)
            self.storage.set_intrinsic_rewards(rews_int_raw)
            rews_int_norm = self.rnd.normalize_intrinsic_rewards(rews_int_raw)
        else:
            rews_int_raw = np.zeros((self.n_envs, self.n_steps), dtype=np.float32)
            rews_int_norm = rews_int_raw
            self.storage.set_intrinsic_rewards(rews_int_raw)

        self.storage.compute_gae(
            rews_int_norm=rews_int_norm,
            gamma=self.rnd_config.gamma,
            gamma_ext=self.rnd_config.gamma_ext,
            lam=self.rnd_config.lam,
            int_coeff=self.rnd_config.int_coeff if self.rnd_config.use_rnd else 0.0,
            ext_coeff=self.rnd_config.ext_coeff,
            use_news=self.rnd_config.use_news,
        )

        info = self.storage.summary()
        info["rewintmean_norm"] = float(rews_int_norm.mean())
        info["rewintmax_norm"] = float(rews_int_norm.max())
        info["ev_int"] = _explained_variance(
            self.storage.buf_vpreds_int.ravel(),
            self.storage.buf_rets_int.ravel(),
        )
        info["ev_ext"] = _explained_variance(
            self.storage.buf_vpreds_ext.ravel(),
            self.storage.buf_rets_ext.ravel(),
        )
        return info

    def train(self) -> dict[str, float]:
        assert (
            self.policy is not None
            and self.rnd is not None
            and self.storage is not None
            and self.optimizer is not None
        )
        self.policy.train()
        self.rnd.model.predictor.train()
        self.rnd.model.freeze_target()

        lr = self._lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        cliprange = self._cliprange()

        loss_names = [
            "tot",
            "pg",
            "vf",
            "ent",
            "clipfrac",
            "approxkl",
            "auxloss",
            "featvar",
            "maxfeat",
            "gradnorm",
        ]
        accum: dict[str, list[float]] = {n: [] for n in loss_names}

        adapter = self.policy.adapter
        is_image = self.policy.is_image

        for _epoch in range(self.n_epochs):
            for mb in self.storage.env_minibatches(self.nminibatches):
                # Flatten env×time for non-recurrent policy (OpenAI cnn feedforward path)
                mb_envs, mb_steps = mb["obs"].shape[:2]
                obs_flat = mb["obs"].reshape(mb_envs * mb_steps, *mb["obs"].shape[2:])
                if is_image:
                    obs_batch: Any = obs_flat
                else:
                    # rebuild as flat vectors already
                    obs_batch = obs_flat

                acs = mb["acs"].reshape(mb_envs * mb_steps, -1)
                if self.policy.is_discrete:
                    acs_t = th.as_tensor(acs.reshape(-1), device=self.device, dtype=th.long)
                else:
                    acs_t = th.as_tensor(acs, device=self.device, dtype=th.float32)
                    if acs_t.ndim == 1:
                        acs_t = acs_t.unsqueeze(-1)

                oldnlp = th.as_tensor(mb["oldnlp"].reshape(-1), device=self.device, dtype=th.float32)
                adv = th.as_tensor(mb["adv"].reshape(-1), device=self.device, dtype=th.float32)
                ret_int = th.as_tensor(mb["ret_int"].reshape(-1), device=self.device, dtype=th.float32)
                ret_ext = th.as_tensor(mb["ret_ext"].reshape(-1), device=self.device, dtype=th.float32)

                if self.normalize_advantage and adv.numel() > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                neglogp, entropy, v_int, v_ext = self.policy.evaluate_actions(obs_batch, acs_t)

                # OpenAI: ratio = exp(oldnlp - neglogpac)
                ratio = th.exp(oldnlp - neglogp)
                negadv = -adv
                pg1 = negadv * ratio
                pg2 = negadv * th.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
                pg_loss = th.mean(th.maximum(pg1, pg2))

                vf_loss_int = 0.5 * self.vf_coef * th.mean(th.square(v_int - ret_int))
                vf_loss_ext = 0.5 * self.vf_coef * th.mean(th.square(v_ext - ret_ext))
                vf_loss = vf_loss_int + vf_loss_ext

                ent_loss = (-self.ent_coef) * th.mean(entropy)
                approxkl = 0.5 * th.mean(th.square(neglogp - oldnlp))
                clipfrac = th.mean((th.abs(ratio - 1.0) > cliprange).float())

                # RND aux on next-obs slice from obs_seq[:, 1:]
                aux_loss = th.zeros((), device=self.device)
                feat_var = th.zeros((), device=self.device)
                max_feat = th.zeros((), device=self.device)
                if self.rnd_config.use_rnd:
                    obs_seq = mb["obs_seq"]  # (mb_envs, T+1, ...)
                    next_obs = obs_seq[:, 1:]
                    next_flat = next_obs.reshape(mb_envs * mb_steps, *next_obs.shape[2:])
                    aux_loss, feat_var, max_feat = self.rnd.aux_loss(next_flat)

                loss = pg_loss + ent_loss + vf_loss + aux_loss

                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm and self.max_grad_norm > 0:
                    grad_norm = th.nn.utils.clip_grad_norm_(
                        list(self.policy.parameters()) + list(self.rnd.model.predictor.parameters()),
                        self.max_grad_norm,
                    )
                else:
                    grad_norm = th.nn.utils.clip_grad_norm_(
                        list(self.policy.parameters()) + list(self.rnd.model.predictor.parameters()),
                        float("inf"),
                    )
                self.optimizer.step()

                for n, v in zip(
                    loss_names,
                    [
                        loss,
                        pg_loss,
                        vf_loss,
                        th.mean(entropy),
                        clipfrac,
                        approxkl,
                        aux_loss,
                        feat_var,
                        max_feat,
                        grad_norm,
                    ],
                ):
                    accum[n].append(float(v.detach().cpu().item() if th.is_tensor(v) else v))

                if self.target_kl is not None and float(approxkl) > 1.5 * self.target_kl:
                    break

        self._n_updates += 1
        return {f"opt_{k}": float(np.mean(v)) for k, v in accum.items() if v}

    def learn(
        self: SelfPPORND,
        total_timesteps: int,
        callback: Any = None,
        log_interval: int = 1,
        tb_log_name: str = "PPORND",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfPPORND:
        if reset_num_timesteps:
            self.num_timesteps = 0
            self._last_obs = None

        if self.tensorboard_log and sb3_configure is not None:
            run_dir = os.path.join(self.tensorboard_log, f"{tb_log_name}_{int(time.time())}")
            self.logger = sb3_configure(run_dir, ["stdout", "tensorboard"])

        t_start = time.time()
        update = 0
        while self.num_timesteps < total_timesteps:
            self._current_progress_remaining = max(1.0 - self.num_timesteps / max(total_timesteps, 1), 0.0)
            roll_info = self.collect_rollouts()
            train_info = self.train()
            update += 1

            if log_interval and update % log_interval == 0 and self.logger is not None:
                self.logger.record("time/fps", self.num_timesteps / max(time.time() - t_start, 1e-8))
                self.logger.record("time/elapsed", time.time() - t_start)
                self.logger.record("train/n_updates", self._n_updates)
                self.logger.record("train/learning_rate", self._lr())
                self.logger.record("train/clip_range", self._cliprange())
                if self._episode_rewards:
                    self.logger.record("rollout/ep_rew_mean", float(np.mean(self._episode_rewards)))
                    self.logger.record("rollout/ep_len_mean", float(np.mean(self._episode_lengths)))
                for k, v in roll_info.items():
                    self.logger.record(f"rnd/{k}", v)
                for k, v in train_info.items():
                    self.logger.record(f"train/{k}", v)
                self.logger.record("rnd/int_coeff", float(self.rnd_config.int_coeff))
                self.logger.record("rnd/ext_coeff", float(self.rnd_config.ext_coeff))
                self.logger.dump(step=self.num_timesteps)

            if callback is not None:
                # Minimal BaseCallback compatibility
                if hasattr(callback, "on_step"):
                    if callback.on_step() is False:
                        break

        return self

    def predict(
        self,
        observation: Any,
        state: Any = None,
        episode_start: Any = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Any]:
        assert self.policy is not None
        self.policy.eval()
        with th.no_grad():
            actions, _, _, _, _ = self.policy(observation, deterministic=deterministic)
        actions_np = actions.cpu().numpy()
        if isinstance(self.action_space, spaces.Box):
            actions_np = np.clip(actions_np, self.action_space.low, self.action_space.high)
        return actions_np, state

    def _excluded_save_params(self) -> list[str]:
        """Attrs excluded from pickled zip ``data`` (torch modules saved via state_dict)."""
        return [
            "policy",
            "rnd",
            "storage",
            "optimizer",
            "env",
            "device",
            "logger",
            "_last_obs",
            "_last_episode_starts",
            "_prev_rews",
            "_episode_rewards",
            "_episode_lengths",
            "ep_info_buffer",
        ]

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        """Torch state_dict names for zip params (SB3 layout)."""
        return ["policy", "rnd.model", "optimizer"], []

    def get_parameters(self) -> dict[str, dict]:
        assert recursive_getattr is not None
        state_dicts_names, _ = self._get_torch_save_params()
        params: dict[str, dict] = {}
        for name in state_dicts_names:
            attr = recursive_getattr(self, name)
            params[name] = attr.state_dict()
        return params

    def set_parameters(
        self,
        load_path_or_dict: str | dict[str, Any],
        exact_match: bool = True,
        device: th.device | str = "auto",
    ) -> None:
        assert recursive_getattr is not None and load_from_zip_file is not None
        if isinstance(load_path_or_dict, dict):
            params = load_path_or_dict
        else:
            _, params, _ = load_from_zip_file(load_path_or_dict, device=device)
            assert params is not None

        objects_needing_update = set(self._get_torch_save_params()[0])
        updated_objects: set[str] = set()

        for name, param in params.items():
            try:
                attr = recursive_getattr(self, name)
            except Exception as e:
                raise ValueError(f"Key {name} is an invalid object name.") from e
            if isinstance(attr, th.optim.Optimizer):
                attr.load_state_dict(param)
            else:
                attr.load_state_dict(param, strict=exact_match)
            updated_objects.add(name)

        if exact_match and updated_objects != objects_needing_update:
            raise ValueError(
                "Names of parameters do not match.\n"
                f"Missing: {objects_needing_update - updated_objects}\n"
                f"Unexpected: {updated_objects - objects_needing_update}"
            )

    def save(self, path: str | Path, exclude: Any = None, include: Any = None) -> None:
        """Save as SB3 ``.zip`` (same layout as ``PPO.save``)."""
        assert self.policy is not None and self.rnd is not None and self.optimizer is not None
        assert save_to_zip_file is not None, "stable_baselines3 required for SB3 zip save"

        # Persist RND extras into picklable attrs before packing data
        self.rnd_extra_state = self.rnd.get_extra_state()
        self.rnd_config = self.rnd.config

        data = self.__dict__.copy()
        if exclude is None:
            exclude = []
        exclude_set = set(exclude).union(self._excluded_save_params())
        if include is not None:
            exclude_set = exclude_set.difference(include)

        state_dicts_names, torch_variable_names = self._get_torch_save_params()
        for torch_var in state_dicts_names + list(torch_variable_names):
            exclude_set.add(torch_var.split(".")[0])

        for param_name in exclude_set:
            data.pop(param_name, None)

        # Mark format for loaders
        data["format"] = "ppornd_sb3_zip"
        data["policy_class"] = self.policy_name
        if isinstance(self.rnd_config, RNDConfig):
            data["rnd_config"] = self.rnd_config.to_dict()

        params_to_save = self.get_parameters()
        save_to_zip_file(path, data=data, params=params_to_save, pytorch_variables=None)

    @classmethod
    def load(
        cls,
        path: str | Path,
        env: Any = None,
        device: th.device | str = "auto",
        custom_objects: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "PPORND":
        """Load from SB3 ``.zip``. Falls back to legacy ``ppornd_v1`` ``.pt`` if present."""
        path_str = str(path)
        zip_candidate = path_str if path_str.endswith(".zip") else path_str + ".zip"
        # Strip .zip for SB3 open_path which re-adds suffix
        zip_load_path = path_str[:-4] if path_str.endswith(".zip") else path_str

        if os.path.exists(zip_candidate) and load_from_zip_file is not None:
            return cls._load_from_sb3_zip(
                zip_load_path,
                env=env,
                device=device,
                custom_objects=custom_objects,
                **kwargs,
            )

        # Legacy rewrite-session .pt fallback
        return cls._load_from_legacy_pt(path_str, env=env, device=device, **kwargs)

    @classmethod
    def _load_from_sb3_zip(
        cls,
        path: str,
        env: Any = None,
        device: th.device | str = "auto",
        custom_objects: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "PPORND":
        assert load_from_zip_file is not None
        data, params, _pytorch_variables = load_from_zip_file(
            path,
            device=device,
            custom_objects=custom_objects,
        )
        assert data is not None, "No data found in the saved zip"
        assert params is not None, "No params found in the saved zip"
        if env is None:
            raise ValueError("PPORND.load requires env=")

        rnd_cfg_raw = data.get("rnd_config", {})
        if isinstance(rnd_cfg_raw, RNDConfig):
            cfg = rnd_cfg_raw.resolve()
        else:
            cfg = RNDConfig.from_dict(rnd_cfg_raw if isinstance(rnd_cfg_raw, dict) else {})

        policy_name = data.get("policy_name") or data.get("policy_class") or "MlpPolicy"
        if not isinstance(policy_name, str):
            policy_name = "MlpPolicy"

        model = cls(
            policy=policy_name,
            env=env,
            learning_rate=data.get("learning_rate") if isinstance(data.get("learning_rate"), float) else (
                data.get("learning_rate") or 1e-4
            ),
            n_steps=int(data.get("n_steps", 128)),
            batch_size=int(data.get("batch_size", 256)),
            n_epochs=int(data.get("n_epochs", 4)),
            gamma=float(data.get("gamma", 0.99)),
            gae_lambda=float(data.get("gae_lambda", 0.95)),
            clip_range=data.get("clip_range") if isinstance(data.get("clip_range"), (float, int)) else 0.1,
            ent_coef=float(data.get("ent_coef", 0.001)),
            vf_coef=float(data.get("vf_coef", 1.0)),
            max_grad_norm=float(data.get("max_grad_norm", 0.0)),
            device=device,
            rnd_config=cfg,
            policy_kwargs=data.get("policy_kwargs") or {},
            nminibatches=data.get("nminibatches"),
            verbose=int(data.get("verbose", 0)),
            seed=data.get("seed"),
            tensorboard_log=data.get("tensorboard_log"),
            target_kl=data.get("target_kl"),
            **{k: v for k, v in kwargs.items() if k not in ("_init_setup_model",)},
        )
        assert model.policy is not None and model.rnd is not None and model.optimizer is not None
        model.set_parameters(params, exact_match=True, device=device)

        extra = data.get("rnd_extra_state")
        if extra is not None:
            model.rnd.set_extra_state(extra)
            model.rnd_extra_state = extra
        model.num_timesteps = int(data.get("num_timesteps", 0))
        model._n_updates = int(data.get("_n_updates", data.get("n_updates", 0)))
        return model

    @classmethod
    def _load_from_legacy_pt(
        cls,
        path: str,
        env: Any = None,
        device: th.device | str = "auto",
        **kwargs: Any,
    ) -> "PPORND":
        """Load rewrite-session ``ppornd_v1`` ``.pt`` checkpoints."""
        pt_path = path if path.endswith(".pt") else path + ".pt"
        if not os.path.exists(pt_path):
            if os.path.exists(path):
                try:
                    marker = th.load(path, map_location="cpu", weights_only=False)
                    if isinstance(marker, dict) and "pt_path" in marker:
                        pt_path = marker["pt_path"]
                except Exception:
                    pass
            if not os.path.exists(pt_path):
                raise FileNotFoundError(
                    f"No PPORND SB3 zip at {path}.zip and no legacy .pt at {pt_path}"
                )

        payload = th.load(pt_path, map_location="cpu", weights_only=False)
        if payload.get("format") != "ppornd_v1":
            raise ValueError(
                "Checkpoint is not a PPORND SB3 zip or ppornd_v1 .pt. "
                "Pre-rewrite combined-reward RNDPPO zips are not supported."
            )
        if env is None:
            raise ValueError("PPORND.load requires env=")

        hp = payload["hyperparams"]
        cfg = RNDConfig.from_dict(payload.get("rnd_config", {}))
        model = cls(
            policy=hp.get("policy_name", "MlpPolicy"),
            env=env,
            learning_rate=hp.get("learning_rate") or 1e-4,
            n_steps=hp.get("n_steps", 128),
            batch_size=hp.get("batch_size", 256),
            n_epochs=hp.get("n_epochs", 4),
            gamma=hp.get("gamma", 0.99),
            gae_lambda=hp.get("gae_lambda", 0.95),
            clip_range=hp.get("clip_range") or 0.1,
            ent_coef=hp.get("ent_coef", 0.001),
            vf_coef=hp.get("vf_coef", 1.0),
            max_grad_norm=hp.get("max_grad_norm", 0.0),
            device=device,
            rnd_config=cfg,
            policy_kwargs=hp.get("policy_kwargs") or {},
            nminibatches=hp.get("nminibatches"),
            **kwargs,
        )
        assert model.policy is not None and model.rnd is not None and model.optimizer is not None
        model.policy.load_state_dict(payload["policy_state_dict"])
        model.rnd.model.load_state_dict(payload["rnd_model_state_dict"])
        model.optimizer.load_state_dict(payload["optimizer_state_dict"])
        model.rnd.set_extra_state(payload.get("rnd_extra_state", {}))
        model.num_timesteps = int(payload.get("num_timesteps", 0))
        model._n_updates = int(payload.get("n_updates", 0))
        return model

    def get_env(self) -> Any:
        return self.env

    def set_env(self, env: Any) -> None:
        self.env = env
        self.n_envs = int(getattr(env, "num_envs", 1))
        self._last_obs = None


# Backwards-compatible alias
RNDPPO = PPORND


class _SimpleLogger:
    def __init__(self, verbose: int = 0) -> None:
        self.verbose = verbose
        self._vals: dict[str, Any] = {}

    def record(self, key: str, value: Any, exclude: Any = None) -> None:
        self._vals[key] = value

    def dump(self, step: int = 0) -> None:
        if self.verbose:
            print(f"step={step} " + " ".join(f"{k}={v}" for k, v in self._vals.items()))
        self._vals.clear()
