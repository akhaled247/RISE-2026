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


def test_ma_factory_reset_cycles_layout_seed_like_sa():
    """Autoreset (no seed) must advance SAR ``_layout_seed``; explicit seed pins it."""
    from backends.safepo.ma_factory import SpecRLMultiGoalEnv

    env = SpecRLMultiGoalEnv("PointLTL0MASAR2-v0", seed=7)
    try:
        wrapper = env.env
        assert getattr(wrapper, "_layout_seed", None) == 7
        env.reset()  # vec-env autoreset style
        assert wrapper._layout_seed == 8
        env.reset()
        assert wrapper._layout_seed == 9
        env.reset(seed=42)
        assert wrapper._layout_seed == 42
        env.reset()
        assert wrapper._layout_seed == 43
    finally:
        env.close()


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


def test_ma_factory_cmdp_cost_gated_by_env_suffix():
    from backends.safepo.ma_factory import SpecRLMultiGoalEnv, cmdp_cost_channels

    assert cmdp_cost_channels("PointLTL0MASAR2-v0") == (False, False)
    assert cmdp_cost_channels("PointLTL0MASAR2WC-v0") == (True, False)
    assert cmdp_cost_channels("PointLTL0MASAR2AC-v0") == (False, True)

    env = SpecRLMultiGoalEnv("PointLTL0MASAR2-v0", seed=0)
    try:
        zero_act = np.zeros(env.n_actions, dtype=np.float32)
        actions = [zero_act, zero_act]
        for _ in range(200):
            _, _, _, costs, _, infos, _ = env.step(actions)
            assert all(c[0] == 0.0 for c in costs), (
                "unconstrained SAR must not promote gremlin proximity to CMDP cost"
            )
            if any(
                float(info.get("cost_collision", 0) or 0) > 0
                for info in infos
                if isinstance(info, dict)
            ):
                break
    finally:
        env.close()


def test_ma_factory_cmdp_cost_gated_by_env_suffix():
    from backends.safepo.ma_factory import SpecRLMultiGoalEnv, cmdp_cost_channels

    assert cmdp_cost_channels("PointLTL0MASAR2-v0") == (False, False)
    assert cmdp_cost_channels("PointLTL0MASAR2WC-v0") == (True, False)
    assert cmdp_cost_channels("PointLTL0MASAR2AC-v0") == (False, True)

    env = SpecRLMultiGoalEnv("PointLTL0MASAR2-v0", seed=0)
    try:
        zero_act = np.zeros(env.n_actions, dtype=np.float32)
        actions = [zero_act, zero_act]
        for _ in range(200):
            _, _, _, costs, _, infos, _ = env.step(actions)
            assert all(c[0] == 0.0 for c in costs), (
                "unconstrained SAR must not promote gremlin proximity to CMDP cost"
            )
            if any(
                float(info.get("cost_collision", 0) or 0) > 0
                for info in infos
                if isinstance(info, dict)
            ):
                break
    finally:
        env.close()
