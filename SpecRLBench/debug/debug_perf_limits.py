"""Profile SAR env episode limits and rollout throughput (debug session 3376cb)."""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "specbench" / "envs" / "zones" / "safety-gymnasium"))
sys.path.insert(0, str(ROOT))

import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from debug.debug_log import agent_log
from utils.env_utils import make_env

ENV_NAME = "PointLTL4MASAR1-v0"
N_ENVS = 8
ROLLOUT_STEPS = 512


def _episode_limits_from_env(env):
    unwrapped = env.unwrapped
    while hasattr(unwrapped, "env"):
        unwrapped = unwrapped.env
    task = getattr(unwrapped, "task", None)
    return {
        "spec_max_episode_steps": getattr(getattr(env, "spec", None), "max_episode_steps", None),
        "task_num_steps": getattr(task, "num_steps", None),
        "task_class": type(task).__name__ if task else None,
    }


def main():
    agent_log(
        "debug_perf_limits.py:main",
        "benchmark_start",
        {"env_name": ENV_NAME, "n_envs": N_ENVS, "rollout_steps": ROLLOUT_STEPS},
        "H5",
        "post-fix",
    )

    vec_cls = DummyVecEnv
    env = make_vec_env(
        lambda: Monitor(make_env(ENV_NAME, render_mode=None, sb3=True)),
        n_envs=N_ENVS,
        vec_env_cls=vec_cls,
    )
    limits = _episode_limits_from_env(env.envs[0])
    agent_log("debug_perf_limits.py:main", "env_limits", limits, "H1", "post-fix")

    env.reset()
    ep_lengths = []
    ep_end_reasons = []
    step_times_ms = []

    t_rollout = time.perf_counter()
    for step_idx in range(ROLLOUT_STEPS):
        t0 = time.perf_counter()
        actions = np.array([env.action_space.sample() for _ in range(N_ENVS)])
        _, _, dones, infos = env.step(actions)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if step_idx % 50 == 0:
            step_times_ms.append(dt_ms)

        for i, done in enumerate(dones):
            if not done:
                continue
            info = infos[i]
            ep_len = info.get("episode", {}).get("l")
            if ep_len is not None:
                ep_lengths.append(int(ep_len))
            reason = "timeout" if info.get("TimeLimit.truncated") else "terminated"
            ep_end_reasons.append(reason)

    rollout_s = time.perf_counter() - t_rollout
    total_env_steps = ROLLOUT_STEPS * N_ENVS
    sps = total_env_steps / rollout_s

    agent_log(
        "debug_perf_limits.py:main",
        "rollout_complete",
        {
            "env_steps": total_env_steps,
            "rollout_seconds": round(rollout_s, 3),
            "env_steps_per_sec": round(sps, 1),
            "mean_step_batch_ms": round(float(np.mean(step_times_ms)), 3) if step_times_ms else None,
            "episodes_finished": len(ep_lengths),
            "mean_ep_len": round(float(np.mean(ep_lengths)), 1) if ep_lengths else None,
            "max_ep_len": max(ep_lengths) if ep_lengths else None,
            "timeout_ends": ep_end_reasons.count("timeout"),
            "terminated_ends": ep_end_reasons.count("terminated"),
            **limits,
        },
        "H2",
        "post-fix",
    )
    print(
        f"limits={limits} sps={sps:.1f} "
        f"episodes={len(ep_lengths)} mean_ep_len={np.mean(ep_lengths) if ep_lengths else 'n/a'}"
    )
    env.close()


if __name__ == "__main__":
    main()
