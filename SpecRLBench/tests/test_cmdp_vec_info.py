"""SafePO vec-info contract: dict with final_observation, not list-of-dicts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.cmdp.factory import SyncVectorSafetyEnv, stack_infos_for_safepo


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


class _ToySafetyEnv:
    """Minimal 6-tuple env: truncate every other step for env 1."""

    def __init__(self, rank: int, obs_dim: int = 3):
        self.rank = rank
        self.obs_dim = obs_dim
        self.t = 0
        self.observation_space = type("S", (), {"shape": (obs_dim,)})()
        self.action_space = type("S", (), {"shape": (2,)})()

    def reset(self, seed=None):
        self.t = 0
        return np.zeros(self.obs_dim, dtype=np.float32), {"cost": 0.0}

    def step(self, action):
        self.t += 1
        obs = np.full(self.obs_dim, float(self.rank), dtype=np.float32)
        # Env 1 truncates every step → final_observation present
        truncated = self.rank == 1
        terminated = False
        info: dict = {"cost": 0.0}
        if truncated or terminated:
            info["final_observation"] = obs.copy()
            obs = np.zeros(self.obs_dim, dtype=np.float32)  # autoreset-ish
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


if __name__ == "__main__":
    test_stack_infos_empty_when_no_done()
    test_stack_infos_preserves_none_and_arrays()
    test_sync_vector_step_info_is_dict()
    print("ALL test_cmdp_vec_info PASSED")
