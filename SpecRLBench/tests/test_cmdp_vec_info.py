"""SafePO vec-info contract: dict with final_observation, not list-of-dicts."""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium
import numpy as np
from gymnasium import spaces

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backends.safepo.paths import ensure_specrlbench_paths

ensure_specrlbench_paths()

from envs.cmdp.factory import (
    AsyncVectorSafetyEnv,
    SyncVectorSafetyEnv,
    make_cmdp_vec,
    stack_infos_for_safepo,
)


def test_stack_infos_empty_when_no_done():
    out = stack_infos_for_safepo([{"cost": 0}, {"cost": 0}])
    assert out == {}
    assert "final_observation" not in out
    print("test_stack_infos_empty_when_no_done OK")


def test_stack_infos_preserves_none_and_arrays():
    fo = np.arange(4, dtype=np.float32)
    out = stack_infos_for_safepo([{"final_observation": fo}, {"cost": 1}])
    assert isinstance(out, dict)
    assert len(out["final_observation"]) == 2
    assert out["final_observation"][0] is not None
    assert np.allclose(out["final_observation"][0], fo)
    assert out["final_observation"][1] is None
    # SafePO indexing path
    idx = 0
    assert out["final_observation"][idx].shape == (4,)
    print("test_stack_infos_preserves_none_and_arrays OK")


class _ToySafetyEnv(gymnasium.Env):
    """Minimal 6-tuple gymnasium env for Sync + SafetyAsync workers."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        rank: int = 0,
        obs_dim: int = 3,
        truncate_every: int = 0,
        *,
        stash_final_obs: bool = True,
    ):
        super().__init__()
        self.rank = rank
        self.obs_dim = obs_dim
        self.truncate_every = truncate_every  # 0 = never; else truncate every N steps
        # Sync path / AutoReset-style: stash final obs + zero. Async workers: False.
        self.stash_final_obs = stash_final_obs
        self.t = 0
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return np.zeros(self.obs_dim, dtype=np.float32), {"cost": 0.0}

    def step(self, action):
        self.t += 1
        obs = np.full(self.obs_dim, float(self.rank), dtype=np.float32)
        truncated = self.truncate_every > 0 and (self.t % self.truncate_every == 0)
        # Legacy sync toy: rank==1 truncates every step when truncate_every unset
        if self.truncate_every == 0 and self.rank == 1:
            truncated = True
        terminated = False
        info: dict = {"cost": 0.0}
        if self.stash_final_obs and (truncated or terminated):
            info["final_observation"] = obs.copy()
            obs = np.zeros(self.obs_dim, dtype=np.float32)
        return obs, 0.0, 0.0, terminated, truncated, info

    def close(self):
        pass


def test_sync_vector_step_info_is_dict():
    env = SyncVectorSafetyEnv([lambda: _ToySafetyEnv(0), lambda: _ToySafetyEnv(1)])
    env.reset(seed=0)
    obs, rew, cost, term, trunc, info = env.step(np.zeros((2, 2), dtype=np.float32))
    assert isinstance(info, dict), f"expected dict info, got {type(info)}"
    assert "final_observation" in info
    assert len(info["final_observation"]) == 2
    assert info["final_observation"][0] is None
    assert info["final_observation"][1] is not None
    # Mimic SafePO timeout bootstrap index
    fo = info["final_observation"][1]
    assert fo.shape == (3,)
    env.close()
    print("test_sync_vector_step_info_is_dict OK")


def test_async_vector_step_info_is_dict():
    """SafetyAsync workers autoreset; info dict must be SafePO-indexable."""
    from safety_gymnasium.vector.async_vector_env import SafetyAsyncVectorEnv

    # truncate_every=1 → every step truncates; stash_final_obs=False (worker owns reset)
    env_fns = [
        lambda: _ToySafetyEnv(0, truncate_every=1, stash_final_obs=False),
        lambda: _ToySafetyEnv(1, truncate_every=1, stash_final_obs=False),
    ]
    raw = SafetyAsyncVectorEnv(env_fns, shared_memory=False, context="spawn")
    env = AsyncVectorSafetyEnv(raw)
    env.reset(seed=0)
    obs, rew, cost, term, trunc, info = env.step(np.zeros((2, 2), dtype=np.float32))
    assert obs.shape[0] == 2
    assert isinstance(info, dict), f"expected dict info, got {type(info)}"
    assert "final_observation" in info
    fo = info["final_observation"]
    # Gymnasium _add_info → list / array length num_envs
    assert len(fo) == 2
    assert fo[0] is not None
    assert fo[1] is not None
    assert np.asarray(fo[0]).shape == (3,)
    env.close()
    print("test_async_vector_step_info_is_dict OK")


def test_make_cmdp_vec_parallel_false_is_sync():
    """parallel=False → SyncVectorSafetyEnv; workers use autoreset=True."""
    import envs.cmdp.factory as factory

    calls = {"autoreset": []}

    def _fake_make_cmdp_env(env_name, **kwargs):
        calls["autoreset"].append(kwargs.get("autoreset"))
        rank = len(calls["autoreset"]) - 1
        return _ToySafetyEnv(rank=rank, truncate_every=0)

    orig = factory.make_cmdp_env
    factory.make_cmdp_env = _fake_make_cmdp_env
    try:
        sync = make_cmdp_vec("Toy", n_envs=2, parallel=False, seed=0, normalize_obs=False)
        assert isinstance(sync, SyncVectorSafetyEnv)
        assert calls["autoreset"] == [True, True]
        sync.close()

        # n_envs==1 forces Sync even if parallel=True
        calls["autoreset"].clear()
        one = make_cmdp_vec("Toy", n_envs=1, parallel=True, seed=0, normalize_obs=False)
        assert isinstance(one, SyncVectorSafetyEnv)
        one.close()
    finally:
        factory.make_cmdp_env = orig
    print("test_make_cmdp_vec_parallel_false_is_sync OK")


def test_make_cmdp_vec_parallel_true_builds_async():
    """parallel=True + n_envs>1 → AsyncVectorSafetyEnv; worker autoreset=False."""
    import envs.cmdp.factory as factory

    calls = {"autoreset": []}

    def _fake_make_cmdp_env(env_name, **kwargs):
        calls["autoreset"].append(kwargs.get("autoreset"))
        rank = len(calls["autoreset"]) - 1
        return _ToySafetyEnv(rank=rank, truncate_every=0)

    class _FakeAsync:
        """In-process stand-in so spawn does not remake real MuJoCo envs."""

        def __init__(self, env_fns, **kwargs):
            self.envs = [fn() for fn in env_fns]
            self.num_envs = len(self.envs)
            self.single_observation_space = self.envs[0].observation_space
            self.single_action_space = self.envs[0].action_space
            self.observation_space = self.single_observation_space
            self.action_space = self.single_action_space

        def reset(self, **kwargs):
            obs = np.stack([e.reset()[0] for e in self.envs])
            return obs, {}

        def step(self, actions):
            outs = [e.step(actions[i]) for i, e in enumerate(self.envs)]
            obs = np.stack([o[0] for o in outs])
            rew = np.asarray([o[1] for o in outs], dtype=np.float32)
            cost = np.asarray([o[2] for o in outs], dtype=np.float32)
            term = np.asarray([o[3] for o in outs], dtype=np.bool_)
            trunc = np.asarray([o[4] for o in outs], dtype=np.bool_)
            return obs, rew, cost, term, trunc, {}

        def close(self):
            for e in self.envs:
                e.close()

        def call(self, name, *args, **kwargs):
            return [getattr(e, name)(*args, **kwargs) if callable(getattr(e, name, None)) else getattr(e, name) for e in self.envs]

        def set_attr(self, name, values):
            if not isinstance(values, (list, tuple)):
                values = [values] * self.num_envs
            for e, v in zip(self.envs, values):
                setattr(e, name, v)

    orig_make = factory.make_cmdp_env
    orig_async = None
    factory.make_cmdp_env = _fake_make_cmdp_env
    try:
        # Patch import site used inside make_cmdp_vec
        import safety_gymnasium.vector.async_vector_env as async_mod

        orig_async = async_mod.SafetyAsyncVectorEnv
        async_mod.SafetyAsyncVectorEnv = _FakeAsync

        env = make_cmdp_vec("Toy", n_envs=2, parallel=True, seed=0, normalize_obs=False)
        assert isinstance(env, AsyncVectorSafetyEnv)
        assert calls["autoreset"] == [False, False]
        obs, _, _, _, _, info = env.step(np.zeros((2, 2), dtype=np.float32))
        assert obs.shape[0] == 2
        assert isinstance(info, dict)
        env.close()
    finally:
        factory.make_cmdp_env = orig_make
        if orig_async is not None:
            import safety_gymnasium.vector.async_vector_env as async_mod

            async_mod.SafetyAsyncVectorEnv = orig_async
    print("test_make_cmdp_vec_parallel_true_builds_async OK")


if __name__ == "__main__":
    test_stack_infos_empty_when_no_done()
    test_stack_infos_preserves_none_and_arrays()
    test_sync_vector_step_info_is_dict()
    test_async_vector_step_info_is_dict()
    test_make_cmdp_vec_parallel_false_is_sync()
    test_make_cmdp_vec_parallel_true_builds_async()
    print("ALL test_cmdp_vec_info PASSED")
