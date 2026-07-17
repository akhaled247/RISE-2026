"""Rollout buffer matching OpenAI ``safe_rl/pg/buffer.py`` ``CPOBuffer``.

Fidelity notes
--------------
* Reward GAE uses ``gamma * lam``.
* Cost TD deltas use reward ``gamma`` for the bootstrap term (OpenAI quirk),
  while cost GAE discount uses ``cost_gamma * cost_lam``.
* Reward advantages are mean/std-normalized; cost advantages are mean-centered only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ppo_lagrangian.utils import EPS, combined_shape, discount_cumsum, keys_as_sorted_list, values_as_sorted_list


class LagrangianRolloutBuffer:
    """On-policy buffer for PPO-Lagrangian (OpenAI CPOBuffer semantics)."""

    def __init__(
        self,
        size: int,
        obs_shape: tuple[int, ...] | dict[str, tuple[int, ...]],
        act_shape: tuple[int, ...],
        pi_info_shapes: dict[str, list[int] | tuple[int, ...]],
        gamma: float = 0.99,
        lam: float = 0.97,
        cost_gamma: float = 0.99,
        cost_lam: float = 0.97,
    ) -> None:
        self.obs_is_dict = isinstance(obs_shape, dict)
        if self.obs_is_dict:
            self.obs_buf: dict[str, np.ndarray] | np.ndarray = {
                k: np.zeros(combined_shape(size, v), dtype=np.float32) for k, v in obs_shape.items()
            }
        else:
            self.obs_buf = np.zeros(combined_shape(size, obs_shape), dtype=np.float32)

        self.act_buf = np.zeros(combined_shape(size, act_shape), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.cadv_buf = np.zeros(size, dtype=np.float32)
        self.cost_buf = np.zeros(size, dtype=np.float32)
        self.cret_buf = np.zeros(size, dtype=np.float32)
        self.cval_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.pi_info_bufs = {
            k: np.zeros([size] + list(v), dtype=np.float32) for k, v in pi_info_shapes.items()
        }
        self.sorted_pi_info_keys = keys_as_sorted_list(self.pi_info_bufs)
        self.gamma, self.lam = gamma, lam
        self.cost_gamma, self.cost_lam = cost_gamma, cost_lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(
        self,
        obs: np.ndarray | dict[str, np.ndarray],
        act: np.ndarray,
        rew: float,
        val: float,
        cost: float,
        cval: float,
        logp: float,
        pi_info: dict[str, np.ndarray],
    ) -> None:
        assert self.ptr < self.max_size, "Buffer full; call get() before storing more."
        if self.obs_is_dict:
            assert isinstance(self.obs_buf, dict)
            for k in self.obs_buf:
                self.obs_buf[k][self.ptr] = obs[k]
        else:
            self.obs_buf[self.ptr] = obs  # type: ignore[index]
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.cost_buf[self.ptr] = cost
        self.cval_buf[self.ptr] = cval
        self.logp_buf[self.ptr] = logp
        for k in self.sorted_pi_info_keys:
            self.pi_info_bufs[k][self.ptr] = pi_info[k]
        self.ptr += 1

    def finish_path(self, last_val: float = 0.0, last_cval: float = 0.0) -> None:
        """Compute GAE advantages and returns for the finished path segment.

        Matches OpenAI ``CPOBuffer.finish_path`` exactly, including the cost-delta
        bootstrap term using reward ``gamma`` rather than ``cost_gamma``.
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        costs = np.append(self.cost_buf[path_slice], last_cval)
        cvals = np.append(self.cval_buf[path_slice], last_cval)
        # OpenAI quirk: cost TD uses reward gamma for next-value term.
        cdeltas = costs[:-1] + self.gamma * cvals[1:] - cvals[:-1]
        self.cadv_buf[path_slice] = discount_cumsum(cdeltas, self.cost_gamma * self.cost_lam)
        self.cret_buf[path_slice] = discount_cumsum(costs, self.cost_gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self) -> dict[str, Any]:
        """Normalize advantages and return the full buffer for one update epoch."""
        assert self.ptr == self.max_size, "Buffer must be full before get()."
        self.ptr, self.path_start_idx = 0, 0

        # Advantage normalizing trick for policy gradient (reward).
        adv_mean = float(np.mean(self.adv_buf))
        adv_std = float(np.std(self.adv_buf))
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + EPS)

        # Center, but do NOT rescale advantages for cost gradient.
        cadv_mean = float(np.mean(self.cadv_buf))
        self.cadv_buf -= cadv_mean

        data: dict[str, Any] = {
            "obs": self.obs_buf if not self.obs_is_dict else {k: v.copy() for k, v in self.obs_buf.items()},  # type: ignore[union-attr]
            "act": self.act_buf.copy(),
            "adv": self.adv_buf.copy(),
            "cadv": self.cadv_buf.copy(),
            "ret": self.ret_buf.copy(),
            "cret": self.cret_buf.copy(),
            "logp": self.logp_buf.copy(),
            "pi_info": {k: self.pi_info_bufs[k].copy() for k in self.sorted_pi_info_keys},
        }
        if self.obs_is_dict:
            assert isinstance(self.obs_buf, dict)
            data["obs"] = {k: self.obs_buf[k].copy() for k in self.obs_buf}
        else:
            data["obs"] = self.obs_buf.copy()  # type: ignore[union-attr]
        return data

    def reset(self) -> None:
        self.ptr = 0
        self.path_start_idx = 0
