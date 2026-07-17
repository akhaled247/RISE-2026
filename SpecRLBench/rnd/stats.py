"""Running statistics matching OpenAI RND (mpi_util.RunningMeanStd + RewardForwardFilter)."""

from __future__ import annotations

import numpy as np


class RunningMeanStd:
    """Welford / parallel variance algorithm (OpenAI ``mpi_util.RunningMeanStd``).

    Single-process variant (``use_mpi=False`` semantics).
    """

    def __init__(self, epsilon: float = 1e-4, shape: tuple[int, ...] = ()) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)
        self.shape = shape

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = int(x.shape[0])
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: int,
    ) -> None:
        batch_mean = np.asarray(batch_mean, dtype=np.float64)
        batch_var = np.asarray(batch_var, dtype=np.float64)
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.mean = new_mean
        self.var = m2 / tot_count
        self.count = tot_count


class RewardForwardFilter:
    """OpenAI ``RewardForwardFilter``: discounted intrinsic reward stream.

    ``rewems <- gamma * rewems + rews`` with no episode reset (default path).
    """

    def __init__(self, gamma: float) -> None:
        self.rewems: np.ndarray | None = None
        self.gamma = float(gamma)

    def update(self, rews: np.ndarray) -> np.ndarray:
        rews = np.asarray(rews, dtype=np.float64)
        if self.rewems is None:
            self.rewems = rews.copy()
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems


class RNDRunningStats:
    """Observation RMS + intrinsic forward-filter RMS (OpenAI semantics)."""

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        n_envs: int,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
        rms_epsilon: float = 1e-4,
        clip_obs: float = 5.0,
        obs_norm: bool = True,
        return_norm: bool = True,
    ) -> None:
        self.gamma = gamma
        self.epsilon = epsilon
        self.clip_obs = clip_obs
        self.obs_norm = obs_norm
        self.return_norm = return_norm
        self.n_envs = n_envs

        self.ob_rms = RunningMeanStd(epsilon=rms_epsilon, shape=obs_shape)
        self.rff_int = RewardForwardFilter(gamma)
        self.rff_rms_int = RunningMeanStd(epsilon=rms_epsilon, shape=())

    def _batch_for_rms(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if not self.ob_rms.shape:
            return obs.reshape(-1)
        return obs.reshape(-1, *self.ob_rms.shape)

    def normalize_obs(self, obs: np.ndarray, update: bool = True) -> np.ndarray:
        """``clip((obs - mean) / std, -clip_obs, clip_obs)`` like OpenAI RND path."""
        obs = np.asarray(obs, dtype=np.float32)
        if not self.obs_norm:
            return obs
        if update:
            self.ob_rms.update(self._batch_for_rms(obs))
        mean = self.ob_rms.mean
        std = np.sqrt(self.ob_rms.var + self.epsilon)
        normed = (obs - mean) / std
        return np.clip(normed, -self.clip_obs, self.clip_obs).astype(np.float32)

    def update_obs_rms(self, obs: np.ndarray) -> None:
        """Update observation RMS without returning normalized values."""
        self.ob_rms.update(self._batch_for_rms(obs))

    def normalize_intrinsic_rewards(self, rews_int: np.ndarray) -> np.ndarray:
        """OpenAI post-rollout intrinsic reward normalization.

        For each env column ``rew``, update forward filter, then
        ``rews_int / sqrt(rff_rms_int.var)``. No mean subtraction, no clip.
        """
        rews_int = np.asarray(rews_int, dtype=np.float32)
        # OpenAI: rffs_int = [rff.update(rew) for rew in buf_rews_int.T]
        # buf shape (nenvs, nsteps) → iterate over env columns as time series
        if rews_int.ndim == 1:
            rews_int = rews_int[None, :]
        assert rews_int.ndim == 2
        nenvs, nsteps = rews_int.shape
        rffs = np.zeros((nenvs, nsteps), dtype=np.float64)
        # Filter is shared across envs in OpenAI InteractionState (one filter).
        # update() is called once per env's time series (column of .T → rows here).
        # Actually OpenAI does:
        #   rffs_int = np.array([self.I.rff_int.update(rew) for rew in self.I.buf_rews_int.T])
        # buf_rews_int is (nenvs, nsteps); .T is (nsteps, nenvs).
        # So each `rew` is shape (nenvs,) — one timestep across envs!
        # And RewardForwardFilter holds rewems of shape (nenvs,).
        # So we must process timestep-major: for t in range(nsteps): rff.update(rews[:, t])
        # Wait - list comprehension over .T means:
        #   for rew in buf.T:  # each rew has shape (nenvs,) — that's ONE timestep
        #   rff.update(rew) returns rewems of shape (nenvs,)
        #   rffs_int has shape (nsteps, nenvs)
        # Then rff_rms_int.update(rffs_int.ravel())
        # Then rews_int = buf / sqrt(var)
        #
        # Our storage uses (nenvs, nsteps). So:
        rffs_t = []
        for t in range(nsteps):
            rffs_t.append(self.rff_int.update(rews_int[:, t].astype(np.float64)))
        rffs_arr = np.asarray(rffs_t, dtype=np.float64)  # (nsteps, nenvs)
        if self.return_norm:
            self.rff_rms_int.update(rffs_arr.ravel())
            scale = float(np.sqrt(self.rff_rms_int.var + self.epsilon))
            return (rews_int / scale).astype(np.float32)
        return rews_int.astype(np.float32)

    def get_state(self) -> dict:
        return {
            "ob_rms_mean": self.ob_rms.mean.copy(),
            "ob_rms_var": self.ob_rms.var.copy(),
            "ob_rms_count": float(self.ob_rms.count),
            "rff_rms_mean": float(self.rff_rms_int.mean),
            "rff_rms_var": float(self.rff_rms_int.var),
            "rff_rms_count": float(self.rff_rms_int.count),
            "rff_rewems": None if self.rff_int.rewems is None else self.rff_int.rewems.copy(),
            "n_envs": self.n_envs,
            "gamma": self.gamma,
            "clip_obs": self.clip_obs,
        }

    def set_state(self, state: dict) -> None:
        self.ob_rms.mean = np.asarray(state["ob_rms_mean"], dtype=np.float64)
        self.ob_rms.var = np.asarray(state["ob_rms_var"], dtype=np.float64)
        self.ob_rms.count = float(state["ob_rms_count"])
        self.rff_rms_int.mean = np.asarray(state.get("rff_rms_mean", 0.0), dtype=np.float64)
        self.rff_rms_int.var = np.asarray(state.get("rff_rms_var", 1.0), dtype=np.float64)
        self.rff_rms_int.count = float(state.get("rff_rms_count", 1e-4))
        self.n_envs = int(state.get("n_envs", self.n_envs))
        rewems = state.get("rff_rewems")
        self.rff_int.rewems = None if rewems is None else np.asarray(rewems, dtype=np.float64)
        if "gamma" in state:
            self.gamma = float(state["gamma"])
            self.rff_int.gamma = self.gamma
        if "clip_obs" in state:
            self.clip_obs = float(state["clip_obs"])
