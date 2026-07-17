"""Env-major rollout storage matching OpenAI ``InteractionState`` buffers."""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np
import torch as th


class InteractionStorage:
    """Buffers shaped ``(n_envs, n_steps, ...)`` plus bootstrap last-step fields."""

    def __init__(
        self,
        n_envs: int,
        n_steps: int,
        obs_shape: tuple[int, ...],
        action_dim: int,
        discrete: bool,
        device: th.device | str = "cpu",
    ) -> None:
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.discrete = discrete
        self.device = th.device(device)
        self.reset()

    def reset(self) -> None:
        self.buf_obs = np.zeros((self.n_envs, self.n_steps, *self.obs_shape), dtype=np.float32)
        self.buf_ob_last = np.zeros((self.n_envs, *self.obs_shape), dtype=np.float32)
        if self.discrete:
            self.buf_acs = np.zeros((self.n_envs, self.n_steps), dtype=np.int64)
        else:
            self.buf_acs = np.zeros((self.n_envs, self.n_steps, self.action_dim), dtype=np.float32)
        self.buf_nlps = np.zeros((self.n_envs, self.n_steps), dtype=np.float32)
        self.buf_ent = np.zeros((self.n_envs, self.n_steps), dtype=np.float32)
        self.buf_vpreds_int = np.zeros((self.n_envs, self.n_steps), dtype=np.float32)
        self.buf_vpreds_ext = np.zeros((self.n_envs, self.n_steps), dtype=np.float32)
        self.buf_vpred_int_last = np.zeros(self.n_envs, dtype=np.float32)
        self.buf_vpred_ext_last = np.zeros(self.n_envs, dtype=np.float32)
        self.buf_rews_int = np.zeros((self.n_envs, self.n_steps), dtype=np.float32)
        self.buf_rews_ext = np.zeros((self.n_envs, self.n_steps), dtype=np.float32)
        self.buf_news = np.zeros((self.n_envs, self.n_steps), dtype=np.float32)
        self.buf_new_last = np.zeros(self.n_envs, dtype=np.float32)
        self.buf_advs = np.zeros((self.n_envs, self.n_steps), dtype=np.float32)
        self.buf_advs_int = np.zeros((self.n_envs, self.n_steps), dtype=np.float32)
        self.buf_advs_ext = np.zeros((self.n_envs, self.n_steps), dtype=np.float32)
        self.buf_rets_int = np.zeros((self.n_envs, self.n_steps), dtype=np.float32)
        self.buf_rets_ext = np.zeros((self.n_envs, self.n_steps), dtype=np.float32)
        self.pos = 0

    def add_step(
        self,
        t: int,
        obs: np.ndarray,
        actions: np.ndarray,
        neglogp: np.ndarray,
        entropy: np.ndarray,
        vpred_int: np.ndarray,
        vpred_ext: np.ndarray,
        news: np.ndarray,
        rews_ext_prev: np.ndarray | None = None,
    ) -> None:
        self.buf_obs[:, t] = obs
        self.buf_acs[:, t] = actions
        self.buf_nlps[:, t] = neglogp
        self.buf_ent[:, t] = entropy
        self.buf_vpreds_int[:, t] = vpred_int
        self.buf_vpreds_ext[:, t] = vpred_ext
        self.buf_news[:, t] = news.astype(np.float32)
        if rews_ext_prev is not None and t > 0:
            self.buf_rews_ext[:, t - 1] = rews_ext_prev

    def set_bootstrap(
        self,
        ob_last: np.ndarray,
        new_last: np.ndarray,
        vpred_int_last: np.ndarray,
        vpred_ext_last: np.ndarray,
        rews_ext_last: np.ndarray,
    ) -> None:
        self.buf_ob_last[:] = ob_last
        self.buf_new_last[:] = new_last.astype(np.float32)
        self.buf_vpred_int_last[:] = vpred_int_last
        self.buf_vpred_ext_last[:] = vpred_ext_last
        self.buf_rews_ext[:, -1] = rews_ext_last

    def set_intrinsic_rewards(self, rews_int: np.ndarray) -> None:
        self.buf_rews_int[:] = rews_int

    def compute_gae(
        self,
        rews_int_norm: np.ndarray,
        gamma: float,
        gamma_ext: float,
        lam: float,
        int_coeff: float,
        ext_coeff: float,
        use_news: bool,
    ) -> None:
        """OpenAI dual GAE; write advs/returns into buffers."""
        rews_int = np.asarray(rews_int_norm, dtype=np.float32)
        rews_ext = self.buf_rews_ext
        lastgaelam = np.zeros(self.n_envs, dtype=np.float32)
        for t in range(self.n_steps - 1, -1, -1):
            if use_news:
                nextnew = self.buf_news[:, t + 1] if t + 1 < self.n_steps else self.buf_new_last
            else:
                nextnew = 0.0
            nextvals = (
                self.buf_vpreds_int[:, t + 1]
                if t + 1 < self.n_steps
                else self.buf_vpred_int_last
            )
            nextnotnew = 1.0 - nextnew
            delta = rews_int[:, t] + gamma * nextvals * nextnotnew - self.buf_vpreds_int[:, t]
            lastgaelam = delta + gamma * lam * nextnotnew * lastgaelam
            self.buf_advs_int[:, t] = lastgaelam
        self.buf_rets_int[:] = self.buf_advs_int + self.buf_vpreds_int

        lastgaelam = np.zeros(self.n_envs, dtype=np.float32)
        for t in range(self.n_steps - 1, -1, -1):
            nextnew = self.buf_news[:, t + 1] if t + 1 < self.n_steps else self.buf_new_last
            nextvals = (
                self.buf_vpreds_ext[:, t + 1]
                if t + 1 < self.n_steps
                else self.buf_vpred_ext_last
            )
            nextnotnew = 1.0 - nextnew
            delta = rews_ext[:, t] + gamma_ext * nextvals * nextnotnew - self.buf_vpreds_ext[:, t]
            lastgaelam = delta + gamma_ext * lam * nextnotnew * lastgaelam
            self.buf_advs_ext[:, t] = lastgaelam
        self.buf_rets_ext[:] = self.buf_advs_ext + self.buf_vpreds_ext
        self.buf_advs[:] = int_coeff * self.buf_advs_int + ext_coeff * self.buf_advs_ext

    def next_obs_for_rnd(self) -> np.ndarray:
        """Concatenate obs[:, 1:] with ob_last → shape ``(n_envs, n_steps, *obs_shape)``.

        OpenAI feeds ``ph_ob[:, 1:]`` (next frames) into RND.
        """
        # For step t, next obs is obs[:, t+1] for t < n_steps-1, else ob_last
        out = np.empty_like(self.buf_obs)
        out[:, :-1] = self.buf_obs[:, 1:]
        out[:, -1] = self.buf_ob_last
        return out

    def env_minibatches(self, nminibatches: int) -> Iterator[dict[str, Any]]:
        """OpenAI env-slice minibatches (no time shuffle)."""
        assert self.n_envs % nminibatches == 0, (
            f"n_envs={self.n_envs} must divide nminibatches={nminibatches}"
        )
        envs_per = self.n_envs // nminibatches
        for start in range(0, self.n_envs, envs_per):
            end = start + envs_per
            sli = slice(start, end)
            # Obs for RND/policy opt: concat current segment + last bootstrap frame
            # OpenAI: concatenate([buf_obs[mbenvinds], buf_ob_last[mbenvinds, None]], 1)
            # → shape (envs_per, n_steps+1, ...)
            obs_seq = np.concatenate(
                [self.buf_obs[sli], self.buf_ob_last[sli, None]],
                axis=1,
            )
            yield {
                "obs": self.buf_obs[sli],  # (mb, T, ...)
                "obs_seq": obs_seq,  # (mb, T+1, ...)
                "acs": self.buf_acs[sli],
                "oldnlp": self.buf_nlps[sli],
                "adv": self.buf_advs[sli],
                "ret_int": self.buf_rets_int[sli],
                "ret_ext": self.buf_rets_ext[sli],
                "news": self.buf_news[sli],
            }

    def summary(self) -> dict[str, float]:
        return {
            "rewint_unnorm_mean": float(self.buf_rews_int.mean()),
            "rewint_unnorm_std": float(self.buf_rews_int.std()),
            "rewint_unnorm_max": float(self.buf_rews_int.max()),
            "rewext_mean": float(self.buf_rews_ext.mean()),
            "adv_mean": float(self.buf_advs.mean()),
            "adv_std": float(self.buf_advs.std()),
            "retint_mean": float(self.buf_rets_int.mean()),
            "retext_mean": float(self.buf_rets_ext.mean()),
            "vpredint_mean": float(self.buf_vpreds_int.mean()),
            "vpredext_mean": float(self.buf_vpreds_ext.mean()),
        }


# Back-compat alias used by older tests
class RNDStorage(InteractionStorage):
    """Deprecated alias; prefer :class:`InteractionStorage`."""

    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        input_dim: int,
        device: th.device | str = "auto",
    ) -> None:
        super().__init__(
            n_envs=n_envs,
            n_steps=n_steps,
            obs_shape=(input_dim,),
            action_dim=1,
            discrete=True,
            device=device if device != "auto" else "cpu",
        )
        # legacy diagnostic buffers
        self.raw_intrinsic = self.buf_rews_int
        self.norm_intrinsic = np.zeros_like(self.buf_rews_int)
        self.extrinsic = self.buf_rews_ext
        self.combined = np.zeros_like(self.buf_rews_ext)
        self.rnd_obs = self.buf_obs
        self.full = False
        self.generator_ready = False

    def add(
        self,
        rnd_obs: np.ndarray,
        raw_intrinsic: np.ndarray,
        norm_intrinsic: np.ndarray,
        extrinsic: np.ndarray,
        combined: np.ndarray,
    ) -> None:
        t = self.pos
        if t >= self.n_steps:
            return
        self.buf_obs[:, t] = rnd_obs
        self.buf_rews_int[:, t] = raw_intrinsic
        self.norm_intrinsic[:, t] = norm_intrinsic
        self.buf_rews_ext[:, t] = extrinsic
        self.combined[:, t] = combined
        self.pos += 1
        if self.pos == self.n_steps:
            self.full = True

    def _prepare(self) -> None:
        if self.generator_ready:
            return
        assert self.full
        self.rnd_obs = self.buf_obs.swapaxes(0, 1).reshape(-1, self.obs_shape[0])
        self.raw_intrinsic = self.buf_rews_int.swapaxes(0, 1).reshape(-1)
        self.norm_intrinsic = self.norm_intrinsic.swapaxes(0, 1).reshape(-1)
        self.extrinsic = self.buf_rews_ext.swapaxes(0, 1).reshape(-1)
        self.combined = self.combined.swapaxes(0, 1).reshape(-1)
        self.generator_ready = True

    def get_rnd_obs_batch(self, batch_inds: np.ndarray) -> th.Tensor:
        self._prepare()
        return th.as_tensor(self.rnd_obs[batch_inds], device=self.device, dtype=th.float32)
