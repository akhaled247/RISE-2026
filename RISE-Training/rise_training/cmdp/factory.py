"""CMDP env factory for SafePO (Safety 6-tuple, flat Box obs)."""

from __future__ import annotations

import sys
from functools import partial
from typing import Any, Callable

import numpy as np

from rise_training.cmdp.flatten import AutoResetSafetyWrapper, DictFlattenWrapper
from rise_training.cmdp.normalize import ObsNormalizeWrapper
from rise_training.cmdp.safety_step import GymnasiumToSafetyStep


def stack_infos_for_safepo(info_l: list[dict]) -> dict[str, Any]:
    """Merge per-env infos into SafePO's dict-of-sequences layout.

    SafePO ``ppo.main`` does ``info["final_observation"][idx]`` and expects
    ``info`` to be a dict (Gymnasium / SafetyAsyncVectorEnv style), not a list
    of dicts. Entries are ``None`` for envs that did not end this step.
    """
    stacked: dict[str, Any] = {}
    fos = [info.get("final_observation") for info in info_l]
    if any(fo is not None for fo in fos):
        stacked["final_observation"] = [
            None if fo is None else np.asarray(fo, dtype=np.float32) for fo in fos
        ]
    return stacked


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

    Uses ``flat=True`` SpecRLBench wrappers so obs/action are already flat Dict /
    Box for single-agent SAR, then flattens Dict → 1-D Box for SafePO.
    """
    from rise_training.env_utils import make_env

    env = make_env(env_name, render_mode=render_mode, flat=True)
    env = GymnasiumToSafetyStep(env)
    env = DictFlattenWrapper(env)
    if autoreset:
        env = AutoResetSafetyWrapper(env)
    if normalize_obs:
        env = ObsNormalizeWrapper(env, clip_obs=clip_obs, training=training)
    return env


def _make_cmdp_worker_env(
    env_name: str,
    normalize_obs: bool,
    training: bool,
    clip_obs: float,
    render_mode: str | None = None,
):
    """Top-level picklable factory for SafetyAsyncVectorEnv workers (spawn-safe)."""
    # Windows spawn starts a fresh interpreter — restore SpecRL + SG paths.
    try:
        from rise_training.paths import ensure_specrlbench_paths

        ensure_specrlbench_paths()
    except Exception:
        pass
    return make_cmdp_env(
        env_name,
        render_mode=render_mode,
        normalize_obs=normalize_obs,
        autoreset=False,  # SafetyAsync worker autoresets + final_observation
        training=training,
        clip_obs=clip_obs,
    )


def _mp_context() -> str:
    """Match SB3: fork on Linux, spawn on Windows."""
    return "fork" if sys.platform != "win32" else "spawn"


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
        # SafePO discards reset info; return empty dict (not list) for consistency.
        return np.stack(obs_list), {}

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
            stack_infos_for_safepo(info_l),
        )

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def seed(self, seed: int = 0) -> None:
        self.reset(seed=seed)


class AsyncVectorSafetyEnv:
    """Thin adapter: SafetyAsyncVectorEnv + SpecRL RMS API for SafePO save/eval."""

    def __init__(self, vec):
        self._vec = vec
        self.num_envs = vec.num_envs
        self.single_observation_space = vec.single_observation_space
        self.single_action_space = vec.single_action_space
        self.observation_space = vec.observation_space
        self.action_space = vec.action_space

    def __getattr__(self, name: str) -> Any:
        return getattr(self._vec, name)

    @property
    def obs_rms(self):
        """First worker RMS (SafePO logger ``env.obs_rms``)."""
        try:
            states = self._vec.call("get_rms_state")
        except Exception:
            states = None
        if not states:
            class _Dummy:
                mean = None
                var = None
                count = 0

            return _Dummy()
        st = states[0]
        from rise_training.cmdp.normalize import RunningMeanStd

        shape = np.asarray(st["mean"]).shape
        out = RunningMeanStd(shape=shape)
        out.mean = np.asarray(st["mean"], dtype=np.float64)
        out.var = np.asarray(st["var"], dtype=np.float64)
        out.count = float(st["count"])
        return out

    def get_normalize_states(self) -> list[dict] | None:
        try:
            states = list(self._vec.call("get_rms_state"))
        except Exception:
            return None
        if not states or any(s is None for s in states):
            return None
        return states

    def set_normalize_states(self, states: list[dict]) -> None:
        if len(states) != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} RMS states, got {len(states)}"
            )
        self._vec.set_attr("_pending_rms", states)
        self._vec.call("flush_pending_rms")

    def set_training(self, training: bool) -> None:
        self._vec.set_attr("training", training)

    def reset(self, seed: int | None = None, **kwargs):
        if seed is not None:
            kwargs = dict(kwargs)
            kwargs["seed"] = seed
        obs, info = self._vec.reset(**kwargs)
        # SafePO discards reset info; keep dict (not list).
        if isinstance(info, list):
            info = {}
        return obs, info if isinstance(info, dict) else {}

    def step(self, actions: np.ndarray):
        return self._vec.step(actions)

    def close(self) -> None:
        self._vec.close()

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
    parallel: bool = True,
) -> SyncVectorSafetyEnv | AsyncVectorSafetyEnv:
    """Vectorized CMDP envs for SafePO training.

    ``parallel=True`` (default) and ``n_envs > 1`` → SafetyAsyncVectorEnv
    (true multi-process, matches stock SafePO / SB3 SubprocVecEnv).
    Otherwise → SyncVectorSafetyEnv (serial; tests / debug).
    """
    use_async = bool(parallel) and n_envs > 1

    if use_async:
        from safety_gymnasium.vector.async_vector_env import SafetyAsyncVectorEnv

        env_fns = [
            partial(
                _make_cmdp_worker_env,
                env_name,
                normalize_obs,
                training,
                clip_obs,
                render_mode,
            )
            for _ in range(n_envs)
        ]
        raw = SafetyAsyncVectorEnv(env_fns, context=_mp_context())
        env: SyncVectorSafetyEnv | AsyncVectorSafetyEnv = AsyncVectorSafetyEnv(raw)
    else:

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
