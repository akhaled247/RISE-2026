"""CMDP env factory for SafePO (Safety 6-tuple, flat Box obs)."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from envs.cmdp.flatten import AutoResetSafetyWrapper, DictFlattenWrapper
from envs.cmdp.normalize import ObsNormalizeWrapper
from envs.cmdp.safety_step import GymnasiumToSafetyStep


def make_cmdp_env(
    env_name: str,
    *,
    render_mode: str | None = None,
    normalize_obs: bool = True,
    autoreset: bool = True,
    training: bool = True,
    clip_obs: float = 10.0,
):
    """Build single env: WC/SAR Gymnasium wrappers → cost bridge → flatten → norm.

    Uses ``sb3=True`` SpecRLBench wrappers so obs/action are already flat Dict /
    Box for single-agent SAR, then flattens Dict → 1-D Box for SafePO.
    """
    from utils.env_utils import make_env

    env = make_env(env_name, render_mode=render_mode, sb3=True)
    env = GymnasiumToSafetyStep(env)
    env = DictFlattenWrapper(env)
    if autoreset:
        env = AutoResetSafetyWrapper(env)
    if normalize_obs:
        env = ObsNormalizeWrapper(env, clip_obs=clip_obs, training=training)
    return env


class SyncVectorSafetyEnv:
    """Simple sync vector env returning stacked 6-tuples (SafePO-style)."""

    def __init__(self, env_fns: list[Callable[[], Any]]):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space

    @property
    def obs_rms(self):
        """Expose first env RMS if ObsNormalizeWrapper present (SafePO logger)."""
        for e in self.envs:
            cur = e
            while True:
                if isinstance(cur, ObsNormalizeWrapper):
                    return cur.obs_rms
                if not hasattr(cur, "env"):
                    break
                cur = cur.env
        # Dummy RMS-like object so SafePO save_state does not crash
        class _Dummy:
            mean = None
            var = None
            count = 0

        return _Dummy()

    def get_normalize_states(self) -> list[dict] | None:
        states = []
        for e in self.envs:
            cur = e
            found = None
            while True:
                if isinstance(cur, ObsNormalizeWrapper):
                    found = cur.get_rms_state()
                    break
                if not hasattr(cur, "env"):
                    break
                cur = cur.env
            if found is None:
                return None
            states.append(found)
        return states

    def set_normalize_states(self, states: list[dict]) -> None:
        for e, st in zip(self.envs, states):
            cur = e
            while True:
                if isinstance(cur, ObsNormalizeWrapper):
                    cur.set_rms_state(st)
                    break
                if not hasattr(cur, "env"):
                    break
                cur = cur.env

    def set_training(self, training: bool) -> None:
        for e in self.envs:
            cur = e
            while True:
                if isinstance(cur, ObsNormalizeWrapper):
                    cur.training = training
                    break
                if not hasattr(cur, "env"):
                    break
                cur = cur.env

    def reset(self, seed: int | None = None):
        obs_list, info_list = [], []
        for i, env in enumerate(self.envs):
            s = None if seed is None else seed + i
            obs, info = env.reset(seed=s)
            obs_list.append(obs)
            info_list.append(info)
        return np.stack(obs_list), info_list

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions)
        if actions.ndim == 1:
            actions = actions.reshape(self.num_envs, -1)
        obs_l, rew_l, cost_l, term_l, trunc_l, info_l = [], [], [], [], [], []
        for i, env in enumerate(self.envs):
            obs, rew, cost, term, trunc, info = env.step(actions[i])
            obs_l.append(obs)
            rew_l.append(float(rew))
            cost_l.append(float(cost))
            term_l.append(bool(term))
            trunc_l.append(bool(trunc))
            info_l.append(info)
        return (
            np.stack(obs_l).astype(np.float32),
            np.asarray(rew_l, dtype=np.float32),
            np.asarray(cost_l, dtype=np.float32),
            np.asarray(term_l, dtype=np.bool_),
            np.asarray(trunc_l, dtype=np.bool_),
            info_l,
        )

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def seed(self, seed: int = 0) -> None:
        self.reset(seed=seed)


def make_cmdp_vec(
    env_name: str,
    n_envs: int = 1,
    *,
    render_mode: str | None = None,
    normalize_obs: bool = True,
    training: bool = True,
    clip_obs: float = 10.0,
    seed: int | None = 0,
) -> SyncVectorSafetyEnv:
    """Vectorized CMDP envs for SafePO training."""

    def _thunk(rank: int):
        def _fn():
            return make_cmdp_env(
                env_name,
                render_mode=render_mode,
                normalize_obs=normalize_obs,
                autoreset=True,
                training=training,
                clip_obs=clip_obs,
            )

        return _fn

    env = SyncVectorSafetyEnv([_thunk(i) for i in range(n_envs)])
    if seed is not None:
        env.reset(seed=seed)
    return env
