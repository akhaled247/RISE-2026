"""Measure env throughput to explain PPO training ETA."""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "specbench" / "envs" / "zones" / "safety-gymnasium"))

import numpy as np
import safety_gymnasium  # noqa: F401
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from utils.env_utils import make_env

ENV_NAME = "PointLTL0MASAR1-v0"
N_ENVS = 8
WARMUP_STEPS = 50
BENCH_STEPS = 400
TOTAL_TIMESTEPS = 500_000


def _bench(vec_env_cls, label: str) -> float:
    env = make_vec_env(
        lambda: Monitor(make_env(ENV_NAME, render_mode=None, sb3=True)),
        n_envs=N_ENVS,
        vec_env_cls=vec_env_cls,
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    env.reset()
    t0 = time.perf_counter()
    for _ in range(WARMUP_STEPS + BENCH_STEPS):
        action = env.action_space.sample()
        env.step(action)
    elapsed = time.perf_counter() - t0
    steps = BENCH_STEPS * N_ENVS
    sps = steps / elapsed
    env.close()
    eta_hours = (TOTAL_TIMESTEPS / sps) / 3600
    print(f"  eta_500k_hours={eta_hours:.2f}")
    return sps


def _bench_with_resets(vec_env_cls, label: str, reset_every: int) -> float:
    env = make_vec_env(
        lambda: Monitor(make_env(ENV_NAME, render_mode=None, sb3=True)),
        n_envs=N_ENVS,
        vec_env_cls=vec_env_cls,
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    env.reset()
    t0 = time.perf_counter()
    step_in_ep = 0
    reset_count = 0
    for _ in range(WARMUP_STEPS + BENCH_STEPS):
        action = env.action_space.sample()
        env.step(action)
        step_in_ep += 1
        if step_in_ep >= reset_every:
            env.reset()
            step_in_ep = 0
            reset_count += 1
    elapsed = time.perf_counter() - t0
    steps = BENCH_STEPS * N_ENVS
    sps = steps / elapsed
    env.close()
    eta_hours = (TOTAL_TIMESTEPS / sps) / 3600
    print(f"  resets={reset_count} eta_500k_hours={eta_hours:.2f}")
    return sps


if __name__ == "__main__":
    print(f"Benchmarking {ENV_NAME} with n_envs={N_ENVS}")
    dummy_sps = _bench(DummyVecEnv, "DummyVecEnv")
    print(f"DummyVecEnv:  {dummy_sps:.1f} env-steps/s")
    try:
        subproc_sps = _bench(SubprocVecEnv, "SubprocVecEnv")
        print(f"SubprocVecEnv: {subproc_sps:.1f} env-steps/s")
        print(f"Speedup: {subproc_sps / max(dummy_sps, 1e-6):.2f}x")
        reset_sps = _bench_with_resets(DummyVecEnv, "DummyVecEnv", reset_every=176)
        print(f"DummyVecEnv w/ reset@176: {reset_sps:.1f} env-steps/s")
    except Exception as exc:
        print(f"SubprocVecEnv failed: {exc}")
