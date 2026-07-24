"""MA adapter + WC/AC make_env contracts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("mujoco")


def test_make_env_wc_and_ac_exclusive_branching():
    from utils.env_utils import make_env
    from specbench.envs.zones.safety_gym_wrapper_ma_sar_wc import SafetyGymWrapperMASARWC
    from specbench.envs.zones.safety_gym_wrapper_ma_sar_ac import SafetyGymWrapperMASARAC

    wc = make_env("PointLTL0MASAR2WC-v0", sb3=False)
    try:
        assert isinstance(wc, SafetyGymWrapperMASARWC)
        assert not isinstance(wc, SafetyGymWrapperMASARAC)
    finally:
        wc.close()

    ac = make_env("PointLTL0MASAR2AC-v0", sb3=False)
    try:
        assert isinstance(ac, SafetyGymWrapperMASARAC)
    finally:
        ac.close()


def test_ma_factory_reset_step_shapes():
    from backends.safepo.ma_factory import SpecRLMultiGoalEnv

    env = SpecRLMultiGoalEnv("PointLTL0MASAR2-v0", seed=0)
    try:
        obs_n, share_n, avail = env.reset(seed=0)
        assert env.num_agents == 2
        assert len(obs_n) == 2
        assert len(share_n) == 2
        assert avail.shape[0] == 2
        actions = [
            np.zeros(env.n_actions, dtype=np.float32),
            np.zeros(env.n_actions, dtype=np.float32),
        ]
        out = env.step(actions)
        assert len(out) == 7
        obs2, share2, rews, costs, dones, infos, avail2 = out
        assert len(obs2) == 2
        assert len(rews) == 2
        assert len(costs) == 2
        assert len(dones) == 2
    finally:
        env.close()
