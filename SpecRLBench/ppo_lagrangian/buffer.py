"""Rollout buffer (OpenAI CPOBuffer): reward/cost GAE + returns."""

from __future__ import annotations

from typing import Any, Generator

import numpy as np
import scipy.signal

EPS = 1e-8


def _shape(length: int, shape: int | tuple[int, ...] | None) -> tuple[int, ...]:
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def discount_cumsum(x: np.ndarray, discount: float) -> np.ndarray:
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class LagrangianRolloutBuffer:
    """On-policy buffer. Cost TD uses reward gamma (OpenAI quirk)."""

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
                k: np.zeros(_shape(size, v), dtype=np.float32) for k, v in obs_shape.items()
            }
        else:
            self.obs_buf = np.zeros(_shape(size, obs_shape), dtype=np.float32)

        self.act_buf = np.zeros(_shape(size, act_shape), dtype=np.float32)
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
        self.pi_info_keys = sorted(self.pi_info_bufs.keys())
        self.gamma, self.lam = gamma, lam
        self.cost_gamma, self.cost_lam = cost_gamma, cost_lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.generator_ready = False
        self._flat: dict[str, Any] | None = None

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
        assert self.ptr < self.max_size
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
        for k in self.pi_info_keys:
            self.pi_info_bufs[k][self.ptr] = pi_info[k]
        self.ptr += 1

    def finish_path(self, last_val: float = 0.0, last_cval: float = 0.0) -> None:
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        costs = np.append(self.cost_buf[path_slice], last_cval)
        cvals = np.append(self.cval_buf[path_slice], last_cval)
        cdeltas = costs[:-1] + self.gamma * cvals[1:] - cvals[:-1]
        self.cadv_buf[path_slice] = discount_cumsum(cdeltas, self.cost_gamma * self.cost_lam)
        self.cret_buf[path_slice] = discount_cumsum(costs, self.cost_gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self, batch_size: int | None = None) -> Generator[dict[str, Any], None, None]:
        """Yield shuffled minibatches. Adv norm / cadv center once per rollout (not per-mb)."""
        if not self.generator_ready:
            assert self.ptr == self.max_size
            # Advantage norm once on full buffer (OpenAI Lag); not SB3 per-minibatch re-norm.
            adv_mean, adv_std = float(np.mean(self.adv_buf)), float(np.std(self.adv_buf))
            self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + EPS)
            self.cadv_buf -= float(np.mean(self.cadv_buf))

            if self.obs_is_dict:
                assert isinstance(self.obs_buf, dict)
                obs: Any = {k: self.obs_buf[k].copy() for k in self.obs_buf}
            else:
                obs = self.obs_buf.copy()  # type: ignore[union-attr]
            self._flat = {
                "obs": obs,
                "act": self.act_buf.copy(),
                "adv": self.adv_buf.copy(),
                "cadv": self.cadv_buf.copy(),
                "ret": self.ret_buf.copy(),
                "cret": self.cret_buf.copy(),
                "logp": self.logp_buf.copy(),
                "pi_info": {k: self.pi_info_bufs[k].copy() for k in self.pi_info_keys},
            }
            self.generator_ready = True

        assert self._flat is not None
        flat = self._flat
        n = self.max_size
        if batch_size is None:
            batch_size = n
        indices = np.random.permutation(n)
        start = 0
        while start < n:
            batch_inds = indices[start : start + batch_size]
            start += batch_size
            obs_f = flat["obs"]
            if self.obs_is_dict:
                obs_mb: Any = {k: obs_f[k][batch_inds] for k in obs_f}
            else:
                obs_mb = obs_f[batch_inds]
            yield {
                "obs": obs_mb,
                "act": flat["act"][batch_inds],
                "adv": flat["adv"][batch_inds],
                "cadv": flat["cadv"][batch_inds],
                "ret": flat["ret"][batch_inds],
                "cret": flat["cret"][batch_inds],
                "logp": flat["logp"][batch_inds],
                "pi_info": {k: flat["pi_info"][k][batch_inds] for k in flat["pi_info"]},
            }

    def reset(self) -> None:
        self.ptr = 0
        self.path_start_idx = 0
        self.generator_ready = False
        self._flat = None
