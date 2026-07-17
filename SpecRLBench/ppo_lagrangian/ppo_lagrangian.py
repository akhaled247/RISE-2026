"""PPO-Lagrangian with SB3-like API, OpenAI Safety Starter Agents fidelity.

Canonical reference:
  openai/safety-starter-agents
    safe_rl/pg/algos.py       — ppo_lagrangian flags
    safe_rl/pg/run_agent.py   — training loop / losses / penalty
    safe_rl/pg/agents.py      — PPOAgent.update_pi KL early stop
    safe_rl/pg/buffer.py      — GAE / returns / advantage norm
    safe_rl/pg/network.py     — pi / vf / vc

Default Lagrangian mode (from algos.ppo_lagrangian):
  reward_penalized=False, objective_penalized=True,
  learn_penalty=True, penalty_param_loss=True
"""

from __future__ import annotations

import time
import zipfile
from pathlib import Path
from typing import Any, Callable, TypeVar

import cloudpickle
import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from torch.optim import Adam

from ppo_lagrangian.buffer import LagrangianRolloutBuffer
from ppo_lagrangian.logging import EpochLogger
from ppo_lagrangian.policies import MLPActorCritic, obs_to_tensor

SelfPPOLagrangian = TypeVar("SelfPPOLagrangian", bound="PPOLagrangian")


def default_cost_extractor(infos: list[dict] | dict, n_envs: int) -> np.ndarray:
    """OpenAI default: ``info.get('cost', 0)`` per env."""
    costs = np.zeros(n_envs, dtype=np.float32)
    if isinstance(infos, dict):
        # uncommon single-info path
        costs[0] = float(infos.get("cost", 0))
        return costs
    for i, info in enumerate(infos):
        if "cost" in info:
            costs[i] = float(info["cost"])
        elif "cost_sum" in info:
            costs[i] = float(info["cost_sum"])
        else:
            # Fallback: sum numeric cost_* keys (SpecRLBench / Safety Gymnasium)
            total = 0.0
            for k, v in info.items():
                if isinstance(v, (int, float)) and k.startswith("cost") and k != "cost_sum":
                    total += float(v)
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if isinstance(vv, (int, float)) and "cost" in kk and "cost_sum" not in kk:
                            total += float(vv)
            costs[i] = total
    return costs


class PPOLagrangian:
    """Objective-penalized PPO-Lagrangian (OpenAI Safety Starter Agents)."""

    def __init__(
        self,
        policy: str | type = "MlpPolicy",
        env: Any = None,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.97,
        clip_range: float = 0.2,
        ent_coef: float = 0.0,
        target_kl: float = 0.01,
        kl_margin: float = 1.2,
        # Lagrangian / OpenAI-specific
        cost_lim: float = 25.0,
        penalty_init: float = 1.0,
        penalty_lr: float = 5e-2,
        cost_gamma: float = 0.99,
        cost_gae_lambda: float = 0.97,
        vf_lr: float = 1e-3,
        vf_iters: int | None = None,
        pi_iters: int | None = None,
        max_ep_len: int = 1000,
        # SB3-like misc
        policy_kwargs: dict[str, Any] | None = None,
        tensorboard_log: str | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: str | th.device = "auto",
        cost_extractor: Callable[[list[dict], int], np.ndarray] | None = None,
        # Unused SB3-compat knobs (accepted, ignored or remapped)
        **kwargs: Any,
    ) -> None:
        if env is None:
            raise ValueError("env is required")

        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.n_envs = getattr(env, "num_envs", 1)

        self.learning_rate = learning_rate
        self.pi_lr = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size  # accepted for API compat; OpenAI uses full buffer
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_range
        self.ent_coef = ent_coef
        self.target_kl = target_kl
        self.kl_margin = kl_margin

        self.cost_lim = cost_lim
        self.penalty_init = penalty_init
        self.penalty_lr = penalty_lr
        self.cost_gamma = cost_gamma
        self.cost_gae_lambda = cost_gae_lambda
        self.vf_lr = vf_lr
        # OpenAI: pi_iters=80, vf_iters=80 by default. Map n_epochs → both if unset.
        self.pi_iters = pi_iters if pi_iters is not None else n_epochs
        self.vf_iters = vf_iters if vf_iters is not None else n_epochs
        self.max_ep_len = max_ep_len

        self.policy_str = policy if isinstance(policy, str) else getattr(policy, "__name__", "Custom")
        self.policy_kwargs = policy_kwargs or {}
        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self.seed = seed
        self.cost_extractor = cost_extractor or default_cost_extractor

        # Objective-penalized Lagrangian (algos.ppo_lagrangian)
        self.objective_penalized = True
        self.reward_penalized = False
        self.learn_penalty = True
        self.penalty_param_loss = True

        if device == "auto":
            self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        else:
            self.device = th.device(device)

        if seed is not None:
            self.set_random_seed(seed)

        hidden_sizes = tuple(self.policy_kwargs.get("net_arch", (64, 64)))
        if isinstance(hidden_sizes, dict):
            # SB3-style net_arch dict → use pi sizes
            hidden_sizes = tuple(hidden_sizes.get("pi", [64, 64]))
        activation = self.policy_kwargs.get("activation_fn", nn.Tanh)

        self.ac = MLPActorCritic(
            self.observation_space,
            self.action_space,
            hidden_sizes=hidden_sizes,
            activation=activation,
        ).to(self.device)

        # Optimizers (OpenAI: separate Adam for pi, vf+vc, penalty)
        self.pi_optimizer = Adam(self.ac.pi_parameters(), lr=self.pi_lr)
        self.vf_optimizer = Adam(self.ac.vf_parameters(), lr=self.vf_lr)

        # softplus(penalty_param) = penalty; init so softplus(param) ≈ penalty_init
        # OpenAI: param_init = log(max(exp(penalty_init) - 1, 1e-8))
        param_init = float(np.log(max(np.exp(penalty_init) - 1.0, 1e-8)))
        self.penalty_param = nn.Parameter(th.tensor(param_init, dtype=th.float32, device=self.device))
        self.penalty_optimizer = Adam([self.penalty_param], lr=self.penalty_lr)

        # Buffer size = n_steps * n_envs (local_steps_per_epoch in OpenAI / MPI)
        self.buffer_size = int(n_steps * self.n_envs)
        obs_shape = self._obs_shape()
        act_shape = self.action_space.shape if isinstance(self.action_space, spaces.Box) else ()
        self.buffer = LagrangianRolloutBuffer(
            size=self.buffer_size,
            obs_shape=obs_shape,
            act_shape=act_shape if act_shape is not None else (),
            pi_info_shapes=self.ac.pi_info_shapes,
            gamma=self.gamma,
            lam=self.gae_lambda,
            cost_gamma=self.cost_gamma,
            cost_lam=self.cost_gae_lambda,
        )

        self.num_timesteps = 0
        self._n_updates = 0
        self.logger: EpochLogger | None = None
        self._last_obs: Any = None
        self._ep_ret = np.zeros(self.n_envs, dtype=np.float64)
        self._ep_cost = np.zeros(self.n_envs, dtype=np.float64)
        self._ep_len = np.zeros(self.n_envs, dtype=np.int64)
        self._cum_cost = 0.0
        self._path_start: list[int] = [0] * self.n_envs  # unused with flat single buffer

        # Ignore unused kwargs quietly (SB3 drop-in)
        _ = kwargs

    # ------------------------------------------------------------------
    # Properties / helpers
    # ------------------------------------------------------------------

    @property
    def penalty(self) -> th.Tensor:
        return th.nn.functional.softplus(self.penalty_param)

    def set_random_seed(self, seed: int) -> None:
        np.random.seed(seed)
        th.manual_seed(seed)
        if th.cuda.is_available():
            th.cuda.manual_seed_all(seed)

    def _obs_shape(self) -> tuple[int, ...] | dict[str, tuple[int, ...]]:
        if isinstance(self.observation_space, spaces.Dict):
            return {k: self.observation_space.spaces[k].shape for k in self.observation_space.spaces}  # type: ignore[misc]
        return self.observation_space.shape  # type: ignore[return-value]

    def _obs_tensor(self, obs: Any) -> th.Tensor:
        return obs_to_tensor(obs, self.device, self.observation_space)

    def set_env(self, env: Any) -> None:
        self.env = env
        self.n_envs = getattr(env, "num_envs", 1)
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    # ------------------------------------------------------------------
    # Losses (OpenAI run_agent.py computation graph)
    # ------------------------------------------------------------------

    def _policy_loss(
        self,
        logp: th.Tensor,
        logp_old: th.Tensor,
        adv: th.Tensor,
        cadv: th.Tensor,
        ent: th.Tensor,
        penalty: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        ratio = th.exp(logp - logp_old)
        # Clipped surrogate advantage (OpenAI clipped_adv)
        min_adv = th.where(
            adv > 0,
            (1.0 + self.clip_ratio) * adv,
            (1.0 - self.clip_ratio) * adv,
        )
        surr_adv = th.mean(th.minimum(ratio * adv, min_adv))
        surr_cost = th.mean(ratio * cadv)

        # pi_objective = surr_adv + ent_reg * ent
        pi_objective = surr_adv + self.ent_coef * ent
        # objective_penalized:
        pi_objective = pi_objective - penalty * surr_cost
        pi_objective = pi_objective / (1.0 + penalty)
        pi_loss = -pi_objective
        return pi_loss, surr_adv.detach(), surr_cost.detach()

    def _value_losses(
        self,
        v: th.Tensor,
        vc: th.Tensor,
        ret: th.Tensor,
        cret: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        v_loss = th.mean((ret - v) ** 2)
        vc_loss = th.mean((cret - vc) ** 2)
        return v_loss, vc_loss

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        if isinstance(self.action_space, spaces.Box):
            return np.clip(action, self.action_space.low, self.action_space.high)
        return action

    def _bootstrap_from_info(self, info: dict) -> tuple[float, float]:
        """Bootstrap V/Vc from ``terminal_observation`` when present (timeout)."""
        term_obs = info.get("terminal_observation")
        if term_obs is None:
            return 0.0, 0.0
        with th.no_grad():
            obs_t = self._obs_tensor(term_obs)
            lv = float(self.ac.value(obs_t)[0].item())
            lcv = float(self.ac.cost_value(obs_t)[0].item())
        return lv, lcv

    def collect_rollouts(self) -> dict[str, float]:
        """Collect ``n_steps`` transitions from each env (= buffer_size total).

        Path finishing mirrors OpenAI:
        * true terminal (done and not timeout) → bootstrap 0
        * timeout / epoch cutoff → bootstrap from critics
        """
        assert self.env is not None
        self.ac.eval()
        self.buffer.reset()

        if self._last_obs is None:
            reset_out = self.env.reset()
            self._last_obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

        ep_stats: dict[str, list[float]] = {
            "EpRet": [], "EpCost": [], "EpLen": [], "VVals": [], "CostVVals": []
        }
        n_steps = self.n_steps
        n_envs = self.n_envs

        # Staging: time-major, then flush env-major into CPOBuffer
        stage_boot_v = np.zeros((n_steps, n_envs), dtype=np.float32)
        stage_boot_cv = np.zeros((n_steps, n_envs), dtype=np.float32)
        stage_need_finish = np.zeros((n_steps, n_envs), dtype=bool)

        for t in range(n_steps):
            obs_t = self._last_obs
            obs_tensor = self._obs_tensor(obs_t)
            actions, values, cost_values, logps, pi_infos = self.ac.step(obs_tensor)
            actions_np = actions.cpu().numpy()
            values_np = values.cpu().numpy()
            cvalues_np = cost_values.cpu().numpy()
            logps_np = logps.cpu().numpy()
            pi_infos_np = {k: v.detach().cpu().numpy() for k, v in pi_infos.items()}

            clipped = self._clip_action(actions_np)
            step_out = self.env.step(clipped)
            if len(step_out) == 5:
                new_obs, rewards, terminated, truncated, infos = step_out
                dones = np.logical_or(terminated, truncated)
                truncated = np.asarray(truncated).reshape(n_envs)
            else:
                new_obs, rewards, dones, infos = step_out
                truncated = np.array(
                    [bool(info.get("TimeLimit.truncated", False)) for info in infos],
                    dtype=bool,
                )

            rewards = np.asarray(rewards, dtype=np.float64).reshape(n_envs)
            dones = np.asarray(dones).reshape(n_envs)
            costs = self.cost_extractor(infos, n_envs)

            self.num_timesteps += n_envs
            self._cum_cost += float(np.sum(costs))

            if t == 0:
                act_tail = actions_np.shape[1:] if actions_np.ndim > 1 else ()
                self._stage_act = np.zeros((n_steps, n_envs) + act_tail, dtype=np.float32)
                self._stage_rew = np.zeros((n_steps, n_envs), dtype=np.float32)
                self._stage_cost = np.zeros((n_steps, n_envs), dtype=np.float32)
                self._stage_val = np.zeros((n_steps, n_envs), dtype=np.float32)
                self._stage_cval = np.zeros((n_steps, n_envs), dtype=np.float32)
                self._stage_logp = np.zeros((n_steps, n_envs), dtype=np.float32)
                self._stage_done = np.zeros((n_steps, n_envs), dtype=bool)
                self._stage_trunc = np.zeros((n_steps, n_envs), dtype=bool)
                self._stage_pi_info = {
                    k: np.zeros((n_steps, n_envs) + v.shape[1:], dtype=np.float32)
                    for k, v in pi_infos_np.items()
                }
                if isinstance(self.observation_space, spaces.Dict):
                    self._stage_obs_dict = {
                        k: np.zeros(
                            (n_steps, n_envs) + self.observation_space.spaces[k].shape,
                            dtype=np.float32,
                        )
                        for k in self.observation_space.spaces
                    }
                else:
                    self._stage_obs_arr = np.zeros(
                        (n_steps, n_envs) + self.observation_space.shape,  # type: ignore[operator]
                        dtype=np.float32,
                    )

            if isinstance(self.observation_space, spaces.Dict):
                for k in self.observation_space.spaces:
                    self._stage_obs_dict[k][t] = np.asarray(obs_t[k], dtype=np.float32)
            else:
                self._stage_obs_arr[t] = np.asarray(obs_t, dtype=np.float32)

            self._stage_act[t] = actions_np.reshape(self._stage_act[t].shape)
            self._stage_rew[t] = rewards
            self._stage_cost[t] = costs
            self._stage_val[t] = values_np
            self._stage_cval[t] = cvalues_np
            self._stage_logp[t] = logps_np
            self._stage_done[t] = dones
            self._stage_trunc[t] = truncated
            for k, v in pi_infos_np.items():
                self._stage_pi_info[k][t] = v

            for i in range(n_envs):
                if dones[i]:
                    stage_need_finish[t, i] = True
                    # OpenAI: true terminal → 0; timeout → bootstrap V(o)
                    if truncated[i]:
                        lv, lcv = self._bootstrap_from_info(infos[i])
                    else:
                        lv, lcv = 0.0, 0.0
                    stage_boot_v[t, i] = lv
                    stage_boot_cv[t, i] = lcv

            ep_stats["VVals"].extend(values_np.tolist())
            ep_stats["CostVVals"].extend(cvalues_np.tolist())

            self._ep_ret += rewards
            self._ep_cost += costs
            self._ep_len += 1

            for i in range(n_envs):
                if dones[i] or self._ep_len[i] >= self.max_ep_len:
                    ep_stats["EpRet"].append(float(self._ep_ret[i]))
                    ep_stats["EpCost"].append(float(self._ep_cost[i]))
                    ep_stats["EpLen"].append(float(self._ep_len[i]))
                    self._ep_ret[i] = 0.0
                    self._ep_cost[i] = 0.0
                    self._ep_len[i] = 0

            self._last_obs = new_obs

        # Epoch-end bootstrap from current obs (OpenAI sess.run on o)
        last_obs_tensor = self._obs_tensor(self._last_obs)
        with th.no_grad():
            last_val = self.ac.value(last_obs_tensor).cpu().numpy()
            last_cval = self.ac.cost_value(last_obs_tensor).cpu().numpy()

        # Flush env-major into buffer (contiguous paths per env)
        for i in range(n_envs):
            for t in range(n_steps):
                if isinstance(self.observation_space, spaces.Dict):
                    obs_i = {k: self._stage_obs_dict[k][t, i] for k in self._stage_obs_dict}
                else:
                    obs_i = self._stage_obs_arr[t, i]
                self.buffer.store(
                    obs_i,
                    self._stage_act[t, i],
                    float(self._stage_rew[t, i]),
                    float(self._stage_val[t, i]),
                    float(self._stage_cost[t, i]),
                    float(self._stage_cval[t, i]),
                    float(self._stage_logp[t, i]),
                    {k: self._stage_pi_info[k][t, i] for k in self._stage_pi_info},
                )
                if stage_need_finish[t, i]:
                    self.buffer.finish_path(
                        float(stage_boot_v[t, i]),
                        float(stage_boot_cv[t, i]),
                    )

            # Epoch cutoff if path still open
            if self.buffer.path_start_idx < self.buffer.ptr:
                if bool(self._stage_done[-1, i]) and not bool(self._stage_trunc[-1, i]):
                    # Already finished above; should not happen
                    lv, lcv = 0.0, 0.0
                else:
                    lv, lcv = float(last_val[i]), float(last_cval[i])
                self.buffer.finish_path(lv, lcv)

        assert self.buffer.ptr == self.buffer.max_size, (
            f"Buffer ptr {self.buffer.ptr} != max_size {self.buffer.max_size}"
        )

        summary = {
            "EpRet": float(np.mean(ep_stats["EpRet"])) if ep_stats["EpRet"] else 0.0,
            "EpCost": float(np.mean(ep_stats["EpCost"])) if ep_stats["EpCost"] else 0.0,
            "EpLen": float(np.mean(ep_stats["EpLen"])) if ep_stats["EpLen"] else 0.0,
            "n_episodes": float(len(ep_stats["EpRet"])),
        }
        if self.logger is not None:
            for r in ep_stats["EpRet"]:
                self.logger.store(EpRet=r)
            for c in ep_stats["EpCost"]:
                self.logger.store(EpCost=c)
            for L in ep_stats["EpLen"]:
                self.logger.store(EpLen=L)
            for v in ep_stats["VVals"]:
                self.logger.store(VVals=v)
            for cv in ep_stats["CostVVals"]:
                self.logger.store(CostVVals=cv)
        return summary

    # ------------------------------------------------------------------
    # Update (OpenAI update())
    # ------------------------------------------------------------------

    def _batch_tensors(self, data: dict[str, Any]) -> dict[str, Any]:
        device = self.device
        out: dict[str, Any] = {}
        if isinstance(data["obs"], dict):
            out["obs"] = self._obs_tensor({k: data["obs"][k] for k in data["obs"]})
        else:
            out["obs"] = th.as_tensor(data["obs"].reshape(data["obs"].shape[0], -1), dtype=th.float32, device=device)
        out["act"] = th.as_tensor(data["act"], dtype=th.float32, device=device)
        if isinstance(self.action_space, spaces.Discrete):
            out["act"] = out["act"].long().view(-1)
        out["adv"] = th.as_tensor(data["adv"], dtype=th.float32, device=device)
        out["cadv"] = th.as_tensor(data["cadv"], dtype=th.float32, device=device)
        out["ret"] = th.as_tensor(data["ret"], dtype=th.float32, device=device)
        out["cret"] = th.as_tensor(data["cret"], dtype=th.float32, device=device)
        out["logp"] = th.as_tensor(data["logp"], dtype=th.float32, device=device)
        out["pi_info"] = {
            k: th.as_tensor(data["pi_info"][k], dtype=th.float32, device=device) for k in data["pi_info"]
        }
        return out

    def update(self, ep_cost_mean: float) -> dict[str, float]:
        """One OpenAI-style epoch update: penalty → policy → values."""
        self.ac.train()
        data = self.buffer.get()
        batch = self._batch_tensors(data)
        penalty = self.penalty.detach()

        # Pre-update measures
        with th.no_grad():
            logp, ent, d_kl, v, vc = self.ac.evaluate_actions(batch["obs"], batch["act"], batch["pi_info"])
            pi_loss, _, surr_cost = self._policy_loss(
                logp, batch["logp"], batch["adv"], batch["cadv"], ent, penalty
            )
            v_loss, vc_loss = self._value_losses(v, vc, batch["ret"], batch["cret"])
            pre = {
                "LossPi": float(pi_loss.item()),
                "SurrCost": float(surr_cost.item()),
                "LossV": float(v_loss.item()),
                "LossVC": float(vc_loss.item()),
                "Entropy": float(ent.item()),
                "Penalty": float(self.penalty.item()),
            }

        # ----- Penalty update (OpenAI: before policy) -----
        # penalty_loss = -penalty_param * (cur_cost - cost_lim)  when penalty_param_loss
        cur_cost = ep_cost_mean
        if self.learn_penalty:
            self.penalty_optimizer.zero_grad()
            if self.penalty_param_loss:
                penalty_loss = -self.penalty_param * (cur_cost - self.cost_lim)
            else:
                penalty_loss = -self.penalty * (cur_cost - self.cost_lim)
            penalty_loss.backward()
            self.penalty_optimizer.step()

        # ----- Policy update (PPOAgent.update_pi) -----
        stop_iter = self.pi_iters - 1
        for i in range(self.pi_iters):
            self.pi_optimizer.zero_grad()
            logp, ent, d_kl, _, _ = self.ac.evaluate_actions(batch["obs"], batch["act"], batch["pi_info"])
            pen = self.penalty.detach()
            pi_loss, _, _ = self._policy_loss(logp, batch["logp"], batch["adv"], batch["cadv"], ent, pen)
            pi_loss.backward()
            self.pi_optimizer.step()

            # KL after update (OpenAI sess.run([train_pi, d_kl]) post-step intent)
            with th.no_grad():
                _, _, d_kl_post, _, _ = self.ac.evaluate_actions(
                    batch["obs"], batch["act"], batch["pi_info"]
                )
                kl = float(d_kl_post.item())
            if kl > self.kl_margin * self.target_kl:
                if self.verbose:
                    print(f"Early stopping at step {i} due to reaching max kl.")
                stop_iter = i
                break
            stop_iter = i

        # ----- Value update (vf_iters full-batch) -----
        for _ in range(self.vf_iters):
            self.vf_optimizer.zero_grad()
            v = self.ac.value(batch["obs"])
            vc = self.ac.cost_value(batch["obs"])
            v_loss, vc_loss = self._value_losses(v, vc, batch["ret"], batch["cret"])
            total_v = v_loss + vc_loss
            total_v.backward()
            self.vf_optimizer.step()

        # Post-update measures
        with th.no_grad():
            logp, ent, d_kl, v, vc = self.ac.evaluate_actions(batch["obs"], batch["act"], batch["pi_info"])
            pen = self.penalty.detach()
            pi_loss, _, surr_cost = self._policy_loss(
                logp, batch["logp"], batch["adv"], batch["cadv"], ent, pen
            )
            v_loss, vc_loss = self._value_losses(v, vc, batch["ret"], batch["cret"])
            post = {
                "LossPi": float(pi_loss.item()),
                "SurrCost": float(surr_cost.item()),
                "LossV": float(v_loss.item()),
                "LossVC": float(vc_loss.item()),
                "KL": float(d_kl.item()),
                "Penalty": float(self.penalty.item()),
            }

        self._n_updates += 1
        metrics = {
            **{f"pre/{k}": v for k, v in pre.items()},
            "LossPi": pre["LossPi"],
            "DeltaLossPi": post["LossPi"] - pre["LossPi"],
            "SurrCost": pre["SurrCost"],
            "DeltaSurrCost": post["SurrCost"] - pre["SurrCost"],
            "LossV": pre["LossV"],
            "DeltaLossV": post["LossV"] - pre["LossV"],
            "LossVC": pre["LossVC"],
            "DeltaLossVC": post["LossVC"] - pre["LossVC"],
            "Entropy": pre["Entropy"],
            "KL": post["KL"],
            "Penalty": pre["Penalty"],
            "DeltaPenalty": post["Penalty"] - pre["Penalty"],
            "StopIter": float(stop_iter),
        }
        if self.logger is not None:
            self.logger.store(
                LossPi=metrics["LossPi"],
                DeltaLossPi=metrics["DeltaLossPi"],
                SurrCost=metrics["SurrCost"],
                DeltaSurrCost=metrics["DeltaSurrCost"],
                LossV=metrics["LossV"],
                DeltaLossV=metrics["DeltaLossV"],
                LossVC=metrics["LossVC"],
                DeltaLossVC=metrics["DeltaLossVC"],
                Entropy=metrics["Entropy"],
                KL=metrics["KL"],
                Penalty=metrics["Penalty"],
                DeltaPenalty=metrics["DeltaPenalty"],
                StopIter=metrics["StopIter"],
            )
        return metrics

    # ------------------------------------------------------------------
    # learn / predict / save / load
    # ------------------------------------------------------------------

    def learn(
        self: SelfPPOLagrangian,
        total_timesteps: int,
        callback: Any = None,
        log_interval: int = 1,
        tb_log_name: str = "PPOLagrangian",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfPPOLagrangian:
        if reset_num_timesteps:
            self.num_timesteps = 0
            self._cum_cost = 0.0

        self.logger = EpochLogger(
            tensorboard_log=self.tensorboard_log,
            tb_log_name=tb_log_name,
            verbose=self.verbose,
        )

        # Reset env
        reset_out = self.env.reset()
        self._last_obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        self._ep_ret[:] = 0.0
        self._ep_cost[:] = 0.0
        self._ep_len[:] = 0

        start_time = time.time()
        steps_per_epoch = self.buffer_size
        n_epochs = int(np.ceil(total_timesteps / steps_per_epoch))

        iterator = range(n_epochs)
        if progress_bar:
            try:
                from tqdm import trange

                iterator = trange(n_epochs, desc="PPOLagrangian")
            except ImportError:
                pass

        for epoch in iterator:
            if callback is not None:
                # Minimal callback support
                if hasattr(callback, "on_rollout_start"):
                    callback.on_rollout_start()

            rollout_summary = self.collect_rollouts()
            ep_cost = rollout_summary["EpCost"]
            if self.logger is not None and self.logger.epoch_dict.get("EpCost"):
                ep_cost = self.logger.get_stats("EpCost")[0]

            if ep_cost > self.cost_lim and self.verbose:
                print("Warning! Safety constraint is already violated.")

            self.update(ep_cost_mean=ep_cost)

            if callback is not None and hasattr(callback, "on_rollout_end"):
                callback.on_rollout_end()

            cost_rate = self._cum_cost / max(self.num_timesteps, 1)
            if self.logger is not None:
                self.logger.store(
                    Epoch=epoch,
                    CumulativeCost=self._cum_cost,
                    CostRate=cost_rate,
                    TotalEnvInteracts=self.num_timesteps,
                    Time=time.time() - start_time,
                )
                if (epoch % log_interval) == 0:
                    self.logger.dump_tabular(step=self.num_timesteps)

            if self.num_timesteps >= total_timesteps:
                break

        if self.logger is not None:
            self.logger.close()
        return self

    def predict(
        self,
        observation: Any,
        state: Any = None,
        episode_start: Any = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, None]:
        self.ac.eval()
        obs_tensor = self._obs_tensor(observation)
        actions, _, _, _, _ = self.ac.step(obs_tensor, deterministic=deterministic)
        actions_np = actions.cpu().numpy()
        actions_np = self._clip_action(actions_np)
        # SB3 returns squeezed for single env sometimes
        if actions_np.shape[0] == 1 and not hasattr(self.env, "num_envs"):
            actions_np = actions_np[0]
        return actions_np, None

    def get_env(self) -> Any:
        return self.env

    def save(self, path: str | Path) -> None:
        path = Path(path)
        if path.suffix != ".zip":
            path = path.with_suffix(".zip")
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "policy_str": self.policy_str,
            "policy_kwargs": self.policy_kwargs,
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
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
            "num_timesteps": self.num_timesteps,
            "_n_updates": self._n_updates,
            "_cum_cost": self._cum_cost,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "verbose": self.verbose,
            "seed": self.seed,
            "tensorboard_log": self.tensorboard_log,
            "class_name": "PPOLagrangian",
        }
        state = {
            "ac": self.ac.state_dict(),
            "pi_optimizer": self.pi_optimizer.state_dict(),
            "vf_optimizer": self.vf_optimizer.state_dict(),
            "penalty_param": self.penalty_param.detach().cpu(),
            "penalty_optimizer": self.penalty_optimizer.state_dict(),
        }
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("data.pkl", cloudpickle.dumps(data))
            with zf.open("pytorch_variables.pth", "w") as f:
                th.save(state, f)

    @classmethod
    def load(
        cls,
        path: str | Path,
        env: Any = None,
        device: str | th.device = "auto",
        **kwargs: Any,
    ) -> PPOLagrangian:
        path = Path(path)
        if path.suffix != ".zip":
            path = path.with_suffix(".zip")
        with zipfile.ZipFile(path, "r") as zf:
            data = cloudpickle.loads(zf.read("data.pkl"))
            with zf.open("pytorch_variables.pth") as f:
                state = th.load(f, map_location="cpu", weights_only=False)

        if env is None:
            raise ValueError("env is required to load PPOLagrangian (spaces + rollout)")

        model = cls(
            policy=data.get("policy_str", "MlpPolicy"),
            env=env,
            learning_rate=data["learning_rate"],
            n_steps=data["n_steps"],
            batch_size=data["batch_size"],
            n_epochs=data["n_epochs"],
            gamma=data["gamma"],
            gae_lambda=data["gae_lambda"],
            clip_range=data["clip_range"],
            ent_coef=data["ent_coef"],
            target_kl=data["target_kl"],
            kl_margin=data["kl_margin"],
            cost_lim=data["cost_lim"],
            penalty_init=data["penalty_init"],
            penalty_lr=data["penalty_lr"],
            cost_gamma=data["cost_gamma"],
            cost_gae_lambda=data["cost_gae_lambda"],
            vf_lr=data["vf_lr"],
            vf_iters=data["vf_iters"],
            pi_iters=data["pi_iters"],
            max_ep_len=data["max_ep_len"],
            policy_kwargs=data.get("policy_kwargs"),
            tensorboard_log=data.get("tensorboard_log"),
            verbose=data.get("verbose", 0),
            seed=data.get("seed"),
            device=device,
            **kwargs,
        )
        model.ac.load_state_dict(state["ac"])
        model.pi_optimizer.load_state_dict(state["pi_optimizer"])
        model.vf_optimizer.load_state_dict(state["vf_optimizer"])
        model.penalty_param.data.copy_(state["penalty_param"].to(model.device))
        model.penalty_optimizer.load_state_dict(state["penalty_optimizer"])
        model.num_timesteps = data.get("num_timesteps", 0)
        model._n_updates = data.get("_n_updates", 0)
        model._cum_cost = data.get("_cum_cost", 0.0)
        return model
