"""PPO-Lagrangian (OpenAI Safety Starter Agents fidelity).

objective_penalized=True, learn_penalty=True, penalty_param_loss=True.
"""

from __future__ import annotations

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
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.97,
        clip_range: float = 0.2,
        ent_coef: float = 0.0,
        target_kl: float = 0.01,
        kl_margin: float = 1.2,
        cost_lim: float = 25.0,
        penalty_init: float = 1.0,
        penalty_lr: float = 5e-2,
        cost_gamma: float = 0.99,
        cost_gae_lambda: float = 0.97,
        vf_lr: float = 1e-3,
        vf_iters: int | None = None,
        pi_iters: int | None = None,
        max_ep_len: int = 1000,
        hidden_sizes: tuple[int, ...] = (64, 64),
        seed: int | None = None,
        device: str | th.device = "auto",
        cost_fn: Callable[[dict], float] | None = None,
        # Accepted but unused (train-script compat)
        policy: str | None = None,
        batch_size: int | None = None,
        verbose: int = 0,
        tensorboard_log: str | None = None,
        policy_kwargs: dict | None = None,
        **_kwargs: Any,
    ) -> None:
        _ = (policy, batch_size, verbose, tensorboard_log)
        if policy_kwargs and "net_arch" in policy_kwargs:
            arch = policy_kwargs["net_arch"]
            hidden_sizes = tuple(arch) if not isinstance(arch, dict) else tuple(arch.get("pi", [64, 64]))

        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.n_envs = getattr(env, "num_envs", 1)
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_range
        self.ent_coef = ent_coef
        self.target_kl = target_kl
        self.kl_margin = kl_margin
        self.cost_lim = cost_lim
        self.penalty_lr = penalty_lr
        self.cost_gamma = cost_gamma
        self.cost_gae_lambda = cost_gae_lambda
        self.pi_iters = pi_iters if pi_iters is not None else n_epochs
        self.vf_iters = vf_iters if vf_iters is not None else n_epochs
        self.max_ep_len = max_ep_len
        self.cost_fn = cost_fn or _cost_from_info
        self.learning_rate = learning_rate
        self.vf_lr = vf_lr
        self.penalty_init = penalty_init
        self.hidden_sizes = hidden_sizes
        self.seed = seed

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
        self._ep_len = np.zeros(self.n_envs, dtype=np.int64)

    @property
    def penalty(self) -> th.Tensor:
        return th.nn.functional.softplus(self.penalty_param)

    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        if isinstance(self.action_space, spaces.Box):
            return np.clip(action, self.action_space.low, self.action_space.high)
        return action

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
                    self._ep_cost[i] = 0.0
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

    def update(self, ep_cost_mean: float) -> None:
        self.ac.train()
        data = self.buffer.get()
        device = self.device
        if isinstance(data["obs"], dict):
            obs = _obs_tensor(data["obs"], device, self.observation_space)
        else:
            obs = th.as_tensor(data["obs"].reshape(data["obs"].shape[0], -1), dtype=th.float32, device=device)
        act = th.as_tensor(data["act"], dtype=th.float32, device=device)
        if isinstance(self.action_space, spaces.Discrete):
            act = act.long().view(-1)
        adv = th.as_tensor(data["adv"], dtype=th.float32, device=device)
        cadv = th.as_tensor(data["cadv"], dtype=th.float32, device=device)
        ret = th.as_tensor(data["ret"], dtype=th.float32, device=device)
        cret = th.as_tensor(data["cret"], dtype=th.float32, device=device)
        logp_old = th.as_tensor(data["logp"], dtype=th.float32, device=device)
        pi_info = {k: th.as_tensor(data["pi_info"][k], dtype=th.float32, device=device) for k in data["pi_info"]}

        # Penalty update (before policy)
        self.penalty_optimizer.zero_grad()
        (-self.penalty_param * (ep_cost_mean - self.cost_lim)).backward()
        self.penalty_optimizer.step()

        # Policy update with KL early stop
        for i in range(self.pi_iters):
            self.pi_optimizer.zero_grad()
            logp, ent, _, _, _ = self.ac.evaluate_actions(obs, act, pi_info)
            ratio = th.exp(logp - logp_old)
            min_adv = th.where(adv > 0, (1 + self.clip_ratio) * adv, (1 - self.clip_ratio) * adv)
            surr_adv = th.mean(th.minimum(ratio * adv, min_adv))
            surr_cost = th.mean(ratio * cadv)
            pen = self.penalty.detach()
            pi_obj = (surr_adv + self.ent_coef * ent - pen * surr_cost) / (1.0 + pen)
            (-pi_obj).backward()
            self.pi_optimizer.step()
            with th.no_grad():
                _, _, d_kl, _, _ = self.ac.evaluate_actions(obs, act, pi_info)
            if float(d_kl.item()) > self.kl_margin * self.target_kl:
                break

        # Value update
        for _ in range(self.vf_iters):
            self.vf_optimizer.zero_grad()
            v, vc = self.ac.value(obs), self.ac.cost_value(obs)
            (th.mean((ret - v) ** 2) + th.mean((cret - vc) ** 2)).backward()
            self.vf_optimizer.step()

    def learn(self, total_timesteps: int, **_kwargs: Any) -> PPOLagrangian:
        out = self.env.reset()
        self._last_obs = out[0] if isinstance(out, tuple) else out
        self._ep_cost[:] = 0.0
        self._ep_len[:] = 0
        self.num_timesteps = 0
        n_epochs = int(np.ceil(total_timesteps / self.buffer_size))
        for _ in range(n_epochs):
            ep_cost = self.collect_rollouts()
            self.update(ep_cost)
            if self.num_timesteps >= total_timesteps:
                break
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
                    "n_epochs": self.pi_iters,
                    "gamma": self.gamma,
                    "gae_lambda": self.gae_lambda,
                    "clip_range": self.clip_ratio,
                    "ent_coef": self.ent_coef,
                    "target_kl": self.target_kl,
                    "kl_margin": self.kl_margin,
                    "cost_lim": self.cost_lim,
                    "penalty_init": self.penalty_init,
                    "penalty_lr": self.penalty_lr,
                    "cost_gamma": self.cost_gamma,
                    "cost_gae_lambda": self.cost_gae_lambda,
                    "vf_lr": self.vf_lr,
                    "vf_iters": self.vf_iters,
                    "pi_iters": self.pi_iters,
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
        model = cls(env=env, device=device, **{**cfg, **kwargs})
        model.ac.load_state_dict(state["ac"])
        model.pi_optimizer.load_state_dict(state["pi_optimizer"])
        model.vf_optimizer.load_state_dict(state["vf_optimizer"])
        model.penalty_param.data.copy_(state["penalty_param"].to(model.device))
        model.penalty_optimizer.load_state_dict(state["penalty_optimizer"])
        model.num_timesteps = state["cfg"].get("num_timesteps", 0)
        return model
