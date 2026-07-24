"""Phase 9: fixed-action env parity — Gymnasium+bridge vs make_cmdp_env.

Compares reward / cost / terminated / truncated (not obs RMS).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import safety_gymnasium  # noqa: F401

from envs.cmdp.factory import make_cmdp_env
from envs.cmdp.flatten import DictFlattenWrapper
from envs.cmdp.safety_step import GymnasiumToSafetyStep
from utils.env_utils import make_env

ENV_ID = "PointLTL1MASAR1WC-v0"
SEED = 0
N_STEPS = 32


def _build_a():
    """Gymnasium path + same adapters as CMDP minus autoreset/norm."""
    env = make_env(ENV_ID, render_mode=None, sb3=True)
    env = GymnasiumToSafetyStep(env)
    env = DictFlattenWrapper(env)
    return env


def _build_b():
    return make_cmdp_env(
        ENV_ID,
        render_mode=None,
        normalize_obs=False,
        autoreset=False,
        training=False,
    )


def _actions(env, n: int) -> list[np.ndarray]:
    space = env.action_space
    low = np.asarray(space.low, dtype=np.float32)
    high = np.asarray(space.high, dtype=np.float32)
    mid = 0.5 * (low + high)
    # Deterministic scripted sequence (no RNG)
    acts = []
    for i in range(n):
        if i % 3 == 0:
            acts.append(np.zeros_like(mid))
        elif i % 3 == 1:
            acts.append(mid.copy())
        else:
            acts.append(np.clip(mid * 0.25, low, high))
    return acts


def test_cmdp_fixed_action_parity():
    env_a = _build_a()
    env_b = _build_b()
    try:
        obs_a, _ = env_a.reset(seed=SEED)
        obs_b, _ = env_b.reset(seed=SEED)
        assert obs_a.shape == obs_b.shape, (obs_a.shape, obs_b.shape)
        np.testing.assert_allclose(obs_a, obs_b, rtol=1e-5, atol=1e-5)

        acts = _actions(env_a, N_STEPS)
        for t, act in enumerate(acts):
            oa, ra, ca, terma, trunca, _ = env_a.step(act)
            ob, rb, cb, termb, truncb, _ = env_b.step(act)
            assert float(ra) == float(rb), f"step {t} reward {ra} vs {rb}"
            assert float(ca) == float(cb), f"step {t} cost {ca} vs {cb}"
            assert bool(terma) == bool(termb), f"step {t} terminated"
            assert bool(trunca) == bool(truncb), f"step {t} truncated"
            if bool(terma) or bool(trunca):
                # Without autoreset, episode ended — stop comparing trajectories
                break
            np.testing.assert_allclose(oa, ob, rtol=1e-5, atol=1e-5)
    finally:
        env_a.close()
        env_b.close()
    print("test_cmdp_fixed_action_parity OK")


if __name__ == "__main__":
    test_cmdp_fixed_action_parity()
