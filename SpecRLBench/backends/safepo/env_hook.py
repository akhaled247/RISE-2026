"""Patch SafePO ``make_sa_mujoco_env`` to use SpecRLBench CMDP envs.

Does not reimplement PPO/TRPO/CPO/Lag — only swaps the env factory so
``safepo.single_agent.*`` scripts train on PointLTL*SAR*WC tasks.
"""

from __future__ import annotations

import functools
from typing import Any

_PATCHED = False
_ORIGINAL_MAKE = None
_PARALLEL = True  # set by runners / set_parallel before SafePO main()

SPECRL_PREFIXES = ("PointLTL", "CarLTL", "AntLTL")


def is_specrlbench_env(env_id: str) -> bool:
    return any(env_id.startswith(p) for p in SPECRL_PREFIXES)


def set_parallel(parallel: bool) -> None:
    """Control SafetyAsync vs Sync for SpecRL CMDP vec (SafePO factory has no kw)."""
    global _PARALLEL
    _PARALLEL = bool(parallel)


def make_specrlbench_sa_env(
    num_envs: int,
    env_id: str,
    seed: int | None = None,
    *,
    training: bool = True,
    parallel: bool = True,
    render_mode: str | None = None,
):
    """Return ``(env, obs_space, act_space)`` matching SafePO's SA contract.

    ``training=False`` freezes obs RMS updates (eval / post-train load).
    ``parallel=True`` (default) uses SafetyAsyncVectorEnv when ``num_envs > 1``.
    ``render_mode='human'`` requires ``num_envs=1`` (GUI cannot run in vec workers).
    """
    from envs.cmdp.factory import make_cmdp_env, make_cmdp_vec

    if render_mode == "human" and num_envs > 1:
        raise ValueError(
            "render_mode='human' requires num_envs=1; use eval with a single env"
        )

    if num_envs > 1:
        env = make_cmdp_vec(
            env_id,
            n_envs=num_envs,
            normalize_obs=True,
            training=training,
            seed=seed,
            parallel=parallel,
            render_mode=render_mode,
        )
        obs_space = env.single_observation_space
        act_space = env.single_action_space
        return env, obs_space, act_space

    env = make_cmdp_env(
        env_id,
        normalize_obs=True,
        autoreset=True,
        training=training,
        render_mode=render_mode,
    )
    if seed is not None:
        env.reset(seed=seed)
    obs_space = env.observation_space
    act_space = env.action_space
    # SafePO single-env path uses SafeUnsqueeze — batch dim of 1
    env = _UnsqueezeSafetyEnv(env)
    return env, obs_space, act_space


class _UnsqueezeSafetyEnv:
    """Match SafePO SafeUnsqueeze: add leading batch dim on step/reset."""

    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    @property
    def obs_rms(self):
        from envs.cmdp.normalize import find_obs_normalize_wrapper

        wrap = find_obs_normalize_wrapper(self.env)
        if wrap is not None:
            return wrap.obs_rms
        return getattr(self.env, "obs_rms", None)

    @obs_rms.setter
    def obs_rms(self, value) -> None:
        from envs.cmdp.normalize import apply_rms_normalizer

        apply_rms_normalizer(self.env, value, training=False)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        import numpy as np

        return np.expand_dims(np.asarray(obs, dtype=np.float32), 0), info

    def step(self, action):
        import numpy as np

        action = np.asarray(action)
        if action.ndim > 1:
            action = action.squeeze(0)
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        obs = np.expand_dims(np.asarray(obs, dtype=np.float32), 0)
        reward = np.asarray([reward], dtype=np.float32)
        cost = np.asarray([cost], dtype=np.float32)
        terminated = np.asarray([terminated], dtype=np.bool_)
        truncated = np.asarray([truncated], dtype=np.bool_)
        if "final_observation" in info and info["final_observation"] is not None:
            info = dict(info)
            info["final_observation"] = np.array(
                [np.asarray(info["final_observation"], dtype=np.float32)]
            )
        return obs, reward, cost, terminated, truncated, info

    def close(self):
        return self.env.close()


def patch_safepo_env_factory() -> None:
    """Idempotent monkey-patch of ``safepo.common.env.make_sa_mujoco_env``."""
    global _PATCHED, _ORIGINAL_MAKE
    if _PATCHED:
        return

    from backends.safepo.paths import ensure_specrlbench_paths

    ensure_specrlbench_paths()
    import safepo.common.env as safepo_env

    _ORIGINAL_MAKE = safepo_env.make_sa_mujoco_env

    @functools.wraps(_ORIGINAL_MAKE)
    def _patched(num_envs: int, env_id: str, seed: int | None = None):
        if is_specrlbench_env(env_id):
            return make_specrlbench_sa_env(
                num_envs, env_id, seed, parallel=_PARALLEL
            )
        return _ORIGINAL_MAKE(num_envs, env_id, seed)

    safepo_env.make_sa_mujoco_env = _patched
    _PATCHED = True


def unpatch_safepo_env_factory() -> None:
    global _PATCHED, _ORIGINAL_MAKE
    if not _PATCHED or _ORIGINAL_MAKE is None:
        return
    import safepo.common.env as safepo_env

    safepo_env.make_sa_mujoco_env = _ORIGINAL_MAKE
    _PATCHED = False
    _ORIGINAL_MAKE = None
