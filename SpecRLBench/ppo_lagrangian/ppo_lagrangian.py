"""PPO-Lagrangian (OpenAI Safety Starter Agents fidelity).

objective_penalized=True, learn_penalty=True, penalty_param_loss=True.
SB3-style minibatch PPO updates + OpenAI objective-penalized Lagrangian.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from torch.optim import Adam

from ppo_lagrangian.buffer import EPS, LagrangianRolloutBuffer

LOG_STD_INIT = -0.5


def _mlp(sizes: list[int], activation: type[nn.Module] = nn.Tanh) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


def _obs_dim(space: spaces.Space) -> int:
    if isinstance(space, spaces.Box):
        return int(np.prod(space.shape))
    if isinstance(space, spaces.Dict):
        return int(sum(np.prod(space.spaces[k].shape) for k in sorted(space.spaces.keys())))
    raise NotImplementedError(type(space))


def _obs_tensor(obs: Any, device: th.device, space: spaces.Space) -> th.Tensor:
    if isinstance(space, spaces.Dict):
        keys = sorted(space.spaces.keys())
        sample = np.asarray(obs[keys[0]])
        if sample.shape == space.spaces[keys[0]].shape:
            flat = np.concatenate([np.asarray(obs[k], dtype=np.float32).reshape(-1) for k in keys])[None, :]
        else:
            b = sample.shape[0]
            flat = np.concatenate(
                [np.asarray(obs[k], dtype=np.float32).reshape(b, -1) for k in keys], axis=-1
            )
        return th.as_tensor(flat, dtype=th.float32, device=device)
    arr = np.asarray(obs, dtype=np.float32)
    if arr.shape == space.shape:  # type: ignore[union-attr]
        arr = arr.reshape(1, -1)
    else:
        arr = arr.reshape(arr.shape[0], -1)
    return th.as_tensor(arr, dtype=th.float32, device=device)


def _gaussian_logp(x: th.Tensor, mu: th.Tensor, log_std: th.Tensor) -> th.Tensor:
    pre = -0.5 * (((x - mu) / (th.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))
    return th.sum(pre, dim=-1)


def _gaussian_kl(mu0: th.Tensor, log_std0: th.Tensor, mu1: th.Tensor, log_std1: th.Tensor) -> th.Tensor:
    var0, var1 = th.exp(2 * log_std0), th.exp(2 * log_std1)
    pre = 0.5 * (((mu1 - mu0) ** 2 + var0) / (var1 + EPS) - 1) + log_std1 - log_std0
    return th.mean(th.sum(pre, dim=-1))


def _gaussian_entropy(log_std: th.Tensor) -> th.Tensor:
    return th.mean(th.sum(log_std + 0.5 * np.log(2 * np.pi * np.e), dim=-1))


def _categorical_kl(logp0: th.Tensor, logp1: th.Tensor) -> th.Tensor:
    return th.mean(th.sum(th.exp(logp1) * (logp1 - logp0), dim=-1))


def _categorical_entropy(logp: th.Tensor) -> th.Tensor:
    return th.mean(-th.sum(logp * th.exp(logp), dim=-1))


class MLPActorCritic(nn.Module):
    """pi / vf / vc (OpenAI mlp_actor_critic)."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        hidden_sizes: tuple[int, ...] = (64, 64),
    ) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_dim = _obs_dim(observation_space)
        hid = list(hidden_sizes)
        self.is_box = isinstance(action_space, spaces.Box)
        self.is_discrete = isinstance(action_space, spaces.Discrete)

        if self.is_box:
            self.act_dim = int(np.prod(action_space.shape))
            self.pi_net = _mlp([self.obs_dim] + hid + [self.act_dim])
            self.log_std = nn.Parameter(LOG_STD_INIT * th.ones(self.act_dim, dtype=th.float32))
            self.pi_info_shapes = {"mu": [self.act_dim], "log_std": [self.act_dim]}
        elif self.is_discrete:
            self.act_dim = int(action_space.n)
            self.pi_net = _mlp([self.obs_dim] + hid + [self.act_dim])
            self.log_std = None  # type: ignore[assignment]
            self.pi_info_shapes = {"logp_all": [self.act_dim]}
        else:
            raise NotImplementedError(type(action_space))

        self.vf = _mlp([self.obs_dim] + hid + [1])
        self.vc = _mlp([self.obs_dim] + hid + [1])

    def value(self, obs: th.Tensor) -> th.Tensor:
        return self.vf(obs).squeeze(-1)

    def cost_value(self, obs: th.Tensor) -> th.Tensor:
        return self.vc(obs).squeeze(-1)

    def step(self, obs: th.Tensor, deterministic: bool = False):
        with th.no_grad():
            v, vc = self.value(obs), self.cost_value(obs)
            if self.is_box:
                mu = self.pi_net(obs)
                log_std = self.log_std.expand_as(mu)
                pi = mu if deterministic else mu + th.randn_like(mu) * th.exp(log_std)
                logp = _gaussian_logp(pi, mu, log_std)
                pi_info = {"mu": mu.detach(), "log_std": log_std.detach()}
            else:
                logits = self.pi_net(obs)
                logp_all = th.log_softmax(logits, dim=-1)
                pi = th.argmax(logits, dim=-1) if deterministic else th.multinomial(
                    th.softmax(logits, dim=-1), 1
                ).squeeze(-1)
                logp = logp_all.gather(-1, pi.unsqueeze(-1)).squeeze(-1)
                pi_info = {"logp_all": logp_all.detach()}
        return pi.detach(), v.detach(), vc.detach(), logp.detach(), pi_info

    def evaluate_actions(self, obs: th.Tensor, act: th.Tensor, old_pi_info: dict[str, th.Tensor]):
        v, vc = self.value(obs), self.cost_value(obs)
        if self.is_box:
            mu = self.pi_net(obs)
            log_std = self.log_std.expand_as(mu)
            logp = _gaussian_logp(act, mu, log_std)
            ent = _gaussian_entropy(log_std)
            d_kl = _gaussian_kl(mu, log_std, old_pi_info["mu"], old_pi_info["log_std"])
        else:
            logp_all = th.log_softmax(self.pi_net(obs), dim=-1)
            if act.dim() > 1:
                act = act.squeeze(-1)
            logp = logp_all.gather(-1, act.long().unsqueeze(-1)).squeeze(-1)
            ent = _categorical_entropy(logp_all)
            d_kl = _categorical_kl(logp_all, old_pi_info["logp_all"])
        return logp, ent, d_kl, v, vc

    def pi_parameters(self) -> list[nn.Parameter]:
        params = list(self.pi_net.parameters())
        if self.log_std is not None:
            params.append(self.log_std)
        return params

    def vf_parameters(self) -> list[nn.Parameter]:
        return list(self.vf.parameters()) + list(self.vc.parameters())


def _cost_from_info(info: dict) -> float:
    """OpenAI: info.get('cost', 0)."""
    return float(info.get("cost", 0))


class PPOLagrangian:
    """Objective-penalized PPO-Lagrangian."""

    def __init__(
        self,
        env: Any,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 256,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.97,
        clip_range: float = 0.2,
        ent_coef: float = 0.0,
        target_kl: float = 0.01,
        max_grad_norm: float = 0.5,
        kl_margin: float = 1.2,  # deprecated; early stop uses 1.5 * target_kl
        cost_lim: float = 25.0,
        penalty_init: float = 1.0,
        penalty_lr: float = 5e-2,
        cost_gamma: float = 0.99,
        cost_gae_lambda: float = 0.97,
        vf_lr: float = 1e-3,
        vf_iters: int | None = None,  # deprecated alias for n_epochs
        pi_iters: int | None = None,  # deprecated alias for n_epochs
        max_ep_len: int = 1000,
        hidden_sizes: tuple[int, ...] = (64, 64),
        seed: int | None = None,
        device: str | th.device = "auto",
        cost_fn: Callable[[dict], float] | None = None,
        policy: str | None = None,
        verbose: int = 0,
        tensorboard_log: str | None = None,
        log_interval: int = 1,
        policy_kwargs: dict | None = None,
        **_kwargs: Any,
    ) -> None:
        _ = policy
        if policy_kwargs and "net_arch" in policy_kwargs:
            arch = policy_kwargs["net_arch"]
            hidden_sizes = tuple(arch) if not isinstance(arch, dict) else tuple(arch.get("pi", [64, 64]))

        # pi_iters / vf_iters deprecated: if either set, use as n_epochs for compat
        if pi_iters is not None or vf_iters is not None:
            n_epochs = pi_iters if pi_iters is not None else vf_iters  # type: ignore[assignment]

        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.n_envs = getattr(env, "num_envs", 1)
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_range
        self.ent_coef = ent_coef
        self.target_kl = target_kl
        self.max_grad_norm = max_grad_norm
        self.kl_margin = kl_margin
        self.cost_lim = cost_lim
        self.penalty_lr = penalty_lr
        self.cost_gamma = cost_gamma
        self.cost_gae_lambda = cost_gae_lambda
        self.max_ep_len = max_ep_len
        self.cost_fn = cost_fn or _cost_from_info
        self.learning_rate = learning_rate
        self.vf_lr = vf_lr
        self.penalty_init = penalty_init
        self.hidden_sizes = hidden_sizes
        self.seed = seed
        self.verbose = verbose
        self.tensorboard_log = tensorboard_log
        self.log_interval = log_interval
        self._tb_writer: Any = None
        self._n_updates = 0

        self.device = th.device("cuda" if th.cuda.is_available() else "cpu") if device == "auto" else th.device(device)
        if seed is not None:
            np.random.seed(seed)
            th.manual_seed(seed)

        self.ac = MLPActorCritic(self.observation_space, self.action_space, hidden_sizes).to(self.device)
        self.pi_optimizer = Adam(self.ac.pi_parameters(), lr=learning_rate)
        self.vf_optimizer = Adam(self.ac.vf_parameters(), lr=vf_lr)
        param_init = float(np.log(max(np.exp(penalty_init) - 1.0, 1e-8)))
        self.penalty_param = nn.Parameter(th.tensor(param_init, dtype=th.float32, device=self.device))
        self.penalty_optimizer = Adam([self.penalty_param], lr=penalty_lr)

        self.buffer_size = int(n_steps * self.n_envs)
        if isinstance(self.observation_space, spaces.Dict):
            obs_shape: Any = {k: self.observation_space.spaces[k].shape for k in self.observation_space.spaces}
        else:
            obs_shape = self.observation_space.shape
        act_shape = self.action_space.shape if isinstance(self.action_space, spaces.Box) else ()
        self.buffer = LagrangianRolloutBuffer(
            self.buffer_size,
            obs_shape,
            act_shape if act_shape is not None else (),
            self.ac.pi_info_shapes,
            gamma,
            gae_lambda,
            cost_gamma,
            cost_gae_lambda,
        )
        self.num_timesteps = 0
        self._last_obs: Any = None
        self._ep_cost = np.zeros(self.n_envs, dtype=np.float64)
        self._ep_rew = np.zeros(self.n_envs, dtype=np.float64)
        self._ep_len = np.zeros(self.n_envs, dtype=np.int64)
        self._ep_info_buffer: deque[dict[str, float]] = deque(maxlen=100)
        self._start_time: float | None = None
        self._last_train_stats: dict[str, float] = {}
        self._tb_writer = None

    @property
    def penalty(self) -> th.Tensor:
        return th.nn.functional.softplus(self.penalty_param)

    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        if isinstance(self.action_space, spaces.Box):
            return np.clip(action, self.action_space.low, self.action_space.high)
        return action

    def _mb_to_tensors(self, data: dict[str, Any]) -> dict[str, Any]:
        device = self.device
        if isinstance(data["obs"], dict):
            obs = _obs_tensor(data["obs"], device, self.observation_space)
        else:
            obs = th.as_tensor(data["obs"].reshape(data["obs"].shape[0], -1), dtype=th.float32, device=device)
        act = th.as_tensor(data["act"], dtype=th.float32, device=device)
        if isinstance(self.action_space, spaces.Discrete):
            act = act.long().view(-1)
        return {
            "obs": obs,
            "act": act,
            "adv": th.as_tensor(data["adv"], dtype=th.float32, device=device),
            "cadv": th.as_tensor(data["cadv"], dtype=th.float32, device=device),
            "ret": th.as_tensor(data["ret"], dtype=th.float32, device=device),
            "cret": th.as_tensor(data["cret"], dtype=th.float32, device=device),
            "logp_old": th.as_tensor(data["logp"], dtype=th.float32, device=device),
            "pi_info": {
                k: th.as_tensor(data["pi_info"][k], dtype=th.float32, device=device) for k in data["pi_info"]
            },
        }

    def collect_rollouts(self) -> float:
        """Collect n_steps * n_envs transitions. Returns mean EpCost for Lagrange."""
        self.ac.eval()
        self.buffer.reset()
        if self._last_obs is None:
            out = self.env.reset()
            self._last_obs = out[0] if isinstance(out, tuple) else out

        n_steps, n_envs = self.n_steps, self.n_envs
        ep_costs: list[float] = []
        stage_boot_v = np.zeros((n_steps, n_envs), dtype=np.float32)
        stage_boot_cv = np.zeros((n_steps, n_envs), dtype=np.float32)
        stage_finish = np.zeros((n_steps, n_envs), dtype=bool)

        for t in range(n_steps):
            obs_t = self._last_obs
            obs_tensor = _obs_tensor(obs_t, self.device, self.observation_space)
            actions, values, cvalues, logps, pi_infos = self.ac.step(obs_tensor)
            actions_np = actions.cpu().numpy()
            values_np = values.cpu().numpy()
            cvalues_np = cvalues.cpu().numpy()
            logps_np = logps.cpu().numpy()
            pi_infos_np = {k: v.detach().cpu().numpy() for k, v in pi_infos.items()}

            step_out = self.env.step(self._clip_action(actions_np))
            if not getattr(self, "_printed_first_step", False):
                self._printed_first_step = True
            if len(step_out) == 5:
                new_obs, rewards, terminated, truncated, infos = step_out
                dones = np.logical_or(terminated, truncated)
                truncated = np.asarray(truncated).reshape(n_envs)
            else:
                new_obs, rewards, dones, infos = step_out
                truncated = np.array([bool(i.get("TimeLimit.truncated", False)) for i in infos])

            rewards = np.asarray(rewards, dtype=np.float64).reshape(n_envs)
            dones = np.asarray(dones).reshape(n_envs)
            costs = np.array([self.cost_fn(infos[i]) for i in range(n_envs)], dtype=np.float32)
            self.num_timesteps += n_envs

            if t == 0:
                act_tail = actions_np.shape[1:] if actions_np.ndim > 1 else ()
                self._s_act = np.zeros((n_steps, n_envs) + act_tail, dtype=np.float32)
                self._s_rew = np.zeros((n_steps, n_envs), dtype=np.float32)
                self._s_cost = np.zeros((n_steps, n_envs), dtype=np.float32)
                self._s_val = np.zeros((n_steps, n_envs), dtype=np.float32)
                self._s_cval = np.zeros((n_steps, n_envs), dtype=np.float32)
                self._s_logp = np.zeros((n_steps, n_envs), dtype=np.float32)
                self._s_done = np.zeros((n_steps, n_envs), dtype=bool)
                self._s_trunc = np.zeros((n_steps, n_envs), dtype=bool)
                self._s_pi = {
                    k: np.zeros((n_steps, n_envs) + v.shape[1:], dtype=np.float32)
                    for k, v in pi_infos_np.items()
                }
                if isinstance(self.observation_space, spaces.Dict):
                    self._s_obs_d = {
                        k: np.zeros(
                            (n_steps, n_envs) + self.observation_space.spaces[k].shape, dtype=np.float32
                        )
                        for k in self.observation_space.spaces
                    }
                else:
                    self._s_obs = np.zeros(
                        (n_steps, n_envs) + self.observation_space.shape, dtype=np.float32  # type: ignore[operator]
                    )

            if isinstance(self.observation_space, spaces.Dict):
                for k in self.observation_space.spaces:
                    self._s_obs_d[k][t] = np.asarray(obs_t[k], dtype=np.float32)
            else:
                self._s_obs[t] = np.asarray(obs_t, dtype=np.float32)

            self._s_act[t] = actions_np.reshape(self._s_act[t].shape)
            self._s_rew[t] = rewards
            self._s_cost[t] = costs
            self._s_val[t] = values_np
            self._s_cval[t] = cvalues_np
            self._s_logp[t] = logps_np
            self._s_done[t] = dones
            self._s_trunc[t] = truncated
            for k, v in pi_infos_np.items():
                self._s_pi[k][t] = v

            self._ep_cost += costs
            self._ep_rew += rewards
            self._ep_len += 1
            for i in range(n_envs):
                if dones[i]:
                    stage_finish[t, i] = True
                    if truncated[i] and infos[i].get("terminal_observation") is not None:
                        with th.no_grad():
                            to = _obs_tensor(infos[i]["terminal_observation"], self.device, self.observation_space)
                            stage_boot_v[t, i] = float(self.ac.value(to)[0].item())
                            stage_boot_cv[t, i] = float(self.ac.cost_value(to)[0].item())
                    else:
                        stage_boot_v[t, i] = 0.0
                        stage_boot_cv[t, i] = 0.0
                if dones[i] or self._ep_len[i] >= self.max_ep_len:
                    ep_costs.append(float(self._ep_cost[i]))
                    self._ep_info_buffer.append(
                        {
                            "r": float(self._ep_rew[i]),
                            "l": float(self._ep_len[i]),
                            "c": float(self._ep_cost[i]),
                        }
                    )
                    self._ep_cost[i] = 0.0
                    self._ep_rew[i] = 0.0
                    self._ep_len[i] = 0

            self._last_obs = new_obs

        with th.no_grad():
            last_t = _obs_tensor(self._last_obs, self.device, self.observation_space)
            last_val = self.ac.value(last_t).cpu().numpy()
            last_cval = self.ac.cost_value(last_t).cpu().numpy()

        for i in range(n_envs):
            for t in range(n_steps):
                obs_i = (
                    {k: self._s_obs_d[k][t, i] for k in self._s_obs_d}
                    if isinstance(self.observation_space, spaces.Dict)
                    else self._s_obs[t, i]
                )
                self.buffer.store(
                    obs_i,
                    self._s_act[t, i],
                    float(self._s_rew[t, i]),
                    float(self._s_val[t, i]),
                    float(self._s_cost[t, i]),
                    float(self._s_cval[t, i]),
                    float(self._s_logp[t, i]),
                    {k: self._s_pi[k][t, i] for k in self._s_pi},
                )
                if stage_finish[t, i]:
                    self.buffer.finish_path(float(stage_boot_v[t, i]), float(stage_boot_cv[t, i]))
            if self.buffer.path_start_idx < self.buffer.ptr:
                if bool(self._s_done[-1, i]) and not bool(self._s_trunc[-1, i]):
                    lv, lcv = 0.0, 0.0
                else:
                    lv, lcv = float(last_val[i]), float(last_cval[i])
                self.buffer.finish_path(lv, lcv)

        assert self.buffer.ptr == self.buffer.max_size
        return float(np.mean(ep_costs)) if ep_costs else 0.0

    def update(self, ep_cost_mean: float) -> dict[str, float]:
        # SB3-style minibatch PPO updates + OpenAI objective-penalized Lagrangian.
        self.ac.train()

        # Penalty update once per rollout (before policy/value minibatch training)
        self.penalty_optimizer.zero_grad()
        (-self.penalty_param * (ep_cost_mean - self.cost_lim)).backward()
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
            for mb in self.buffer.get(self.batch_size):
                t = self._mb_to_tensors(mb)
                obs, act = t["obs"], t["act"]
                adv, cadv = t["adv"], t["cadv"]
                ret, cret = t["ret"], t["cret"]
                logp_old, pi_info = t["logp_old"], t["pi_info"]

                logp, ent, _, _, _ = self.ac.evaluate_actions(obs, act, pi_info)
                ratio = th.exp(logp - logp_old)
                min_adv = th.where(adv > 0, (1 + self.clip_ratio) * adv, (1 - self.clip_ratio) * adv)
                surr_adv = th.mean(th.minimum(ratio * adv, min_adv))
                surr_cost = th.mean(ratio * cadv)
                pen = self.penalty.detach()
                # Entropy only inside Lag objective (not a second SB3 entropy term)
                pi_obj = (surr_adv + self.ent_coef * ent - pen * surr_cost) / (1.0 + pen)
                policy_loss = -pi_obj
                entropy_loss = float((-ent).item())  # ent already mean; SB3: -mean(entropy)

                with th.no_grad():
                    log_ratio = logp - logp_old
                    approx_kl = th.mean((th.exp(log_ratio) - 1) - log_ratio).item()
                    approx_kl_divs.append(approx_kl)
                    clip_fractions.append(th.mean((th.abs(ratio - 1) > self.clip_ratio).float()).item())

                if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at epoch {epoch} due to reaching max kl: {approx_kl:.2f}")
                    break

                self.pi_optimizer.zero_grad()
                policy_loss.backward()
                th.nn.utils.clip_grad_norm_(self.ac.pi_parameters(), self.max_grad_norm)
                self.pi_optimizer.step()

                # Fresh value forward (separate optimizer; avoid shared-graph with pi)
                v = self.ac.value(obs)
                vc = self.ac.cost_value(obs)
                v_loss = th.mean((ret - v) ** 2)
                vc_loss = th.mean((cret - vc) ** 2)
                self.vf_optimizer.zero_grad()
                (v_loss + vc_loss).backward()
                th.nn.utils.clip_grad_norm_(self.ac.vf_parameters(), self.max_grad_norm)
                self.vf_optimizer.step()

                pg_losses.append(float(policy_loss.item()))
                value_losses.append(float(v_loss.item()))
                cost_value_losses.append(float(vc_loss.item()))
                entropy_losses.append(entropy_loss)
                surr_costs.append(float(surr_cost.item()))
                last_loss = float(policy_loss.item() + v_loss.item() + vc_loss.item())

            self._n_updates += 1
            if not continue_training:
                break

        stats = {
            "policy_gradient_loss": float(np.mean(pg_losses)) if pg_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "cost_value_loss": float(np.mean(cost_value_losses)) if cost_value_losses else 0.0,
            "entropy_loss": float(np.mean(entropy_losses)) if entropy_losses else 0.0,
            "approx_kl": float(np.mean(approx_kl_divs)) if approx_kl_divs else 0.0,
            "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
            "surr_cost": float(np.mean(surr_costs)) if surr_costs else 0.0,
            "loss": last_loss,
            "penalty": float(self.penalty.item()),
            "n_updates": float(self._n_updates),
        }
        self._last_train_stats = stats
        return stats

    def _dump_logs(self, iteration: int) -> None:
        """Print + optional TB scalars in SB3-style groups (after each rollout/update)."""
        if self.log_interval > 0 and iteration % self.log_interval != 0:
            return

        ep_rew_mean = float(np.mean([e["r"] for e in self._ep_info_buffer])) if self._ep_info_buffer else 0.0
        ep_len_mean = float(np.mean([e["l"] for e in self._ep_info_buffer])) if self._ep_info_buffer else 0.0
        ep_cost_mean = float(np.mean([e["c"] for e in self._ep_info_buffer])) if self._ep_info_buffer else 0.0
        elapsed = max(time.time() - (self._start_time or time.time()), 1e-8)
        fps = float(self.num_timesteps / elapsed)
        train = self._last_train_stats

        rollout = {
            "ep_rew_mean": ep_rew_mean,
            "ep_len_mean": ep_len_mean,
            "ep_cost_mean": ep_cost_mean,
        }
        time_stats = {
            "fps": fps,
            "total_timesteps": float(self.num_timesteps),
        }

        print("---------------------------------", flush=True)
        print("| rollout/            |          |", flush=True)
        print(f"|    ep_rew_mean      | {ep_rew_mean:8.2f} |", flush=True)
        print(f"|    ep_len_mean      | {ep_len_mean:8.2f} |", flush=True)
        print(f"|    ep_cost_mean     | {ep_cost_mean:8.4g} |", flush=True)
        print("| time/               |          |", flush=True)
        print(f"|    fps              | {fps:8.0f} |", flush=True)
        print(f"|    total_timesteps  | {self.num_timesteps:8d} |", flush=True)
        print("| train/              |          |", flush=True)
        print(f"|    approx_kl        | {train.get('approx_kl', 0):8.4g} |", flush=True)
        print(f"|    clip_fraction    | {train.get('clip_fraction', 0):8.4g} |", flush=True)
        print(f"|    entropy_loss     | {train.get('entropy_loss', 0):8.4g} |", flush=True)
        print(f"|    loss             | {train.get('loss', 0):8.4g} |", flush=True)
        print(f"|    n_updates        | {int(train.get('n_updates', 0)):8d} |", flush=True)
        print(f"|    policy_gradient_loss | {train.get('policy_gradient_loss', 0):8.4g} |", flush=True)
        print(f"|    value_loss       | {train.get('value_loss', 0):8.4g} |", flush=True)
        print(f"|    cost_value_loss  | {train.get('cost_value_loss', 0):8.4g} |", flush=True)
        print(f"|    surr_cost        | {train.get('surr_cost', 0):8.4g} |", flush=True)
        print(f"|    penalty          | {train.get('penalty', 0):8.4g} |", flush=True)
        print("---------------------------------", flush=True)

        if self._tb_writer is not None:
            step = self.num_timesteps
            for k, v in rollout.items():
                self._tb_writer.add_scalar(f"rollout/{k}", v, step)
            for k, v in time_stats.items():
                self._tb_writer.add_scalar(f"time/{k}", v, step)
            for k, v in train.items():
                self._tb_writer.add_scalar(f"train/{k}", v, step)
            self._tb_writer.flush()

    def learn(
      self,
      total_timesteps: int,
      tb_log_name: str = "PPOLagrangian",
      **_kwargs: Any,
  ) -> PPOLagrangian:
        if self.tensorboard_log is not None and self._tb_writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                save_path = Path(self.tensorboard_log) / tb_log_name
                save_path.mkdir(parents=True, exist_ok=True)
                print(f'Training Log Path: {str(save_path)}')
                self._tb_writer = SummaryWriter(str(save_path))
            except ImportError:
                self._tb_writer = None
        out = self.env.reset()
        self._last_obs = out[0] if isinstance(out, tuple) else out
        self._ep_cost[:] = 0.0
        self._ep_rew[:] = 0.0
        self._ep_len[:] = 0
        self.num_timesteps = 0
        self._printed_first_step = False
        self._start_time = time.time()
        n_iters = int(np.ceil(total_timesteps / self.buffer_size))
        for iteration in range(1, n_iters + 1):
            ep_cost = self.collect_rollouts()
            self.update(ep_cost)
            self._dump_logs(iteration)
            if self.num_timesteps >= total_timesteps:
                break
        if self._tb_writer is not None:
            self._tb_writer.close()
        return self

    def predict(self, observation: Any, deterministic: bool = False, **_kwargs: Any):
        self.ac.eval()
        actions, _, _, _, _ = self.ac.step(
            _obs_tensor(observation, self.device, self.observation_space), deterministic=deterministic
        )
        a = self._clip_action(actions.cpu().numpy())
        if a.shape[0] == 1 and not hasattr(self.env, "num_envs"):
            a = a[0]
        return a, None

    def save(self, path: str | Path) -> None:
        path = Path(path)
        if path.suffix not in {".pt", ".pth", ".zip"}:
            path = path.with_suffix(".pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        th.save(
            {
                "cfg": {
                    "learning_rate": self.learning_rate,
                    "n_steps": self.n_steps,
                    "batch_size": self.batch_size,
                    "n_epochs": self.n_epochs,
                    "gamma": self.gamma,
                    "gae_lambda": self.gae_lambda,
                    "clip_range": self.clip_ratio,
                    "ent_coef": self.ent_coef,
                    "target_kl": self.target_kl,
                    "max_grad_norm": self.max_grad_norm,
                    "kl_margin": self.kl_margin,
                    "cost_lim": self.cost_lim,
                    "penalty_init": self.penalty_init,
                    "penalty_lr": self.penalty_lr,
                    "cost_gamma": self.cost_gamma,
                    "cost_gae_lambda": self.cost_gae_lambda,
                    "vf_lr": self.vf_lr,
                    "max_ep_len": self.max_ep_len,
                    "hidden_sizes": self.hidden_sizes,
                    "seed": self.seed,
                    "num_timesteps": self.num_timesteps,
                },
                "ac": self.ac.state_dict(),
                "pi_optimizer": self.pi_optimizer.state_dict(),
                "vf_optimizer": self.vf_optimizer.state_dict(),
                "penalty_param": self.penalty_param.detach().cpu(),
                "penalty_optimizer": self.penalty_optimizer.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, env: Any, device: str | th.device = "auto", **kwargs: Any) -> PPOLagrangian:
        path = Path(path)
        if not path.exists():
            for ext in (".pt", ".pth", ".zip"):
                if path.with_suffix(ext).exists():
                    path = path.with_suffix(ext)
                    break
            else:
                if Path(str(path) + ".pt").exists():
                    path = Path(str(path) + ".pt")
        state = th.load(path, map_location="cpu", weights_only=False)
        cfg = dict(state["cfg"])
        cfg.pop("num_timesteps", None)
        # Drop deprecated keys from old checkpoints
        cfg.pop("pi_iters", None)
        cfg.pop("vf_iters", None)
        model = cls(env=env, device=device, **{**cfg, **kwargs})
        model.ac.load_state_dict(state["ac"])
        model.pi_optimizer.load_state_dict(state["pi_optimizer"])
        model.vf_optimizer.load_state_dict(state["vf_optimizer"])
        model.penalty_param.data.copy_(state["penalty_param"].to(model.device))
        model.penalty_optimizer.load_state_dict(state["penalty_optimizer"])
        model.num_timesteps = state["cfg"].get("num_timesteps", 0)
        return model
