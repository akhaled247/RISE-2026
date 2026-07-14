"""Runtime perf diagnostic — writes debug-701dbb.log (session 701dbb).

Run from SpecRLBench/ with venv active:
    python debug/debug_perf_701dbb.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "specbench" / "envs" / "zones" / "safety-gymnasium"))
sys.path.insert(0, str(ROOT))

import safety_gymnasium  # noqa: F401
from debug.debug_log import agent_log
from utils.env_utils import make_env, make_vec

N_ENVS = 8
WARMUP = 30
BENCH_STEPS = 150
MINI_STEPS = 16384


def _task_info(env) -> dict:
    u = env.unwrapped
    while hasattr(u, "env"):
        u = u.env
    t = getattr(u, "task", None)
    return {
        "task_class": type(t).__name__ if t else None,
        "num_steps": getattr(t, "num_steps", None),
        "lidar_type": getattr(getattr(t, "lidar_conf", None), "type", None),
        "wall_count": getattr(getattr(t, "walls", None), "num", 0),
    }


def _task_info_for_env_name(env_name: str) -> dict:
    env = make_env(env_name, sb3=True)
    try:
        return _task_info(env)
    finally:
        env.close()


def _vec_env_inner(vec_env):
    v = vec_env
    while hasattr(v, "venv"):
        v = v.venv
    return v


def _log(hid: str, msg: str, data: dict) -> None:
    agent_log(f"debug_perf_701dbb.py:{msg}", msg, data, hid, "diag")


def _task_info(env) -> dict:
    u = env.unwrapped
    while hasattr(u, "env"):
        u = u.env
    t = getattr(u, "task", None)
    return {
        "task_class": type(t).__name__ if t else None,
        "num_steps": getattr(t, "num_steps", None),
        "lidar_type": getattr(getattr(t, "lidar_conf", None), "type", None),
        "wall_count": getattr(getattr(t, "walls", None), "num", 0),
    }


def _bench_vec(env_name: str, cls, kwargs=None) -> float:
    kwargs = kwargs or {}
    env = make_vec_env(
        lambda: Monitor(make_env(env_name, sb3=True)),
        n_envs=N_ENVS,
        vec_env_cls=cls,
        vec_env_kwargs=kwargs or None,
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    env.reset()
    for _ in range(WARMUP):
        env.step(env.action_space.sample())
    t0 = time.perf_counter()
    for _ in range(BENCH_STEPS):
        env.step(env.action_space.sample())
    sps = (BENCH_STEPS * N_ENVS) / (time.perf_counter() - t0)
    env.close()
    return sps


def _bench_obs_paths(env_name: str) -> dict:
    env = make_env(env_name, sb3=True)
    task = env.unwrapped.task
    env.reset(seed=0)
    for _ in range(WARMUP):
        env.step(env.action_space.sample())

    t0 = time.perf_counter()
    task.obs()
    obs_ms = (time.perf_counter() - t0) * 1000

    walls = getattr(task, "walls", None)
    cas = getattr(task, "surface_casualtys", None)
    walls_pseudo_ms = walls_occluded_ms = cas_pseudo_ms = cas_occluded_ms = None
    if walls is not None:
        t0 = time.perf_counter()
        task._obs_lidar_pseudo_new(0, walls.pos)
        walls_pseudo_ms = round((time.perf_counter() - t0) * 1000, 3)
        t0 = time.perf_counter()
        task._obs_lidar_pseudo_occluded_new(0, walls)
        walls_occluded_ms = round((time.perf_counter() - t0) * 1000, 3)
    if cas is not None:
        t0 = time.perf_counter()
        task._obs_lidar_pseudo_new(0, cas.pos)
        cas_pseudo_ms = round((time.perf_counter() - t0) * 1000, 3)
        t0 = time.perf_counter()
        task._obs_lidar_pseudo_occluded_new(0, cas)
        cas_occluded_ms = round((time.perf_counter() - t0) * 1000, 3)

    # exercise try_lidar_ids branches
    obs = {}
    branches = []
    for obstacle in task._obstacles:
        if not obstacle.is_lidar_observed:
            continue
        for i in range(task.agent_num):
            before = set(obs.keys())
            task.try_lidar_ids(obstacle, obs, i)
            branches.append(
                {
                    "obstacle": obstacle.name,
                    "keys_added": sorted(set(obs.keys()) - before),
                }
            )

    t0 = time.perf_counter()
    for _ in range(BENCH_STEPS):
        env.step(env.action_space.sample())
    step_ms = (time.perf_counter() - t0) / BENCH_STEPS * 1000
    env.close()
    return {
        "obs_ms": round(obs_ms, 2),
        "walls_pseudo_ms": walls_pseudo_ms,
        "walls_occluded_ms": walls_occluded_ms,
        "cas_pseudo_ms": cas_pseudo_ms,
        "cas_occluded_ms": cas_occluded_ms,
        "try_lidar_branches": branches,
        "ms_per_step": round(step_ms, 2),
        "fps_1env": round(1000 / step_ms, 1),
    }


def _mini_ppo(env_name: str, n_steps: int) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_vec(env_name, n_envs=N_ENVS, sb3=True)
    env.reset()
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=0,
        n_steps=n_steps,
        batch_size=256,
        n_epochs=10,
        device=device,
        seed=0,
    )
    t0 = time.perf_counter()
    model.learn(total_timesteps=MINI_STEPS, progress_bar=False)
    elapsed = time.perf_counter() - t0
    iters = MINI_STEPS // (n_steps * N_ENVS)
    env.close()
    return {
        "env_name": env_name,
        "n_steps": n_steps,
        "device": device,
        "seconds": round(elapsed, 2),
        "iter_per_sec": round(iters / elapsed, 2),
        "env_steps_per_sec": round(MINI_STEPS / elapsed, 1),
    }


def main() -> None:
    summary = {}
    _log("H0", "diag_start", {"platform": sys.platform, "cuda": torch.cuda.is_available()})

    env = make_vec("PointLTL4MASAR1-v0", n_envs=N_ENVS, sb3=True)
    inner = _vec_env_inner(env)
    summary["make_vec"] = {
        "vec_cls": type(inner).__name__,
        "start_method": getattr(inner, "start_method", None),
        **_task_info_for_env_name("PointLTL4MASAR1-v0"),
    }
    _log("H2", "make_vec_path", summary["make_vec"])
    env.close()

    summary["vec_bench"] = {}
    for label, cls, kw in (
        ("dummy", DummyVecEnv, {}),
        ("subproc_fork", SubprocVecEnv, {"start_method": "fork"}),
    ):
        try:
            sps = _bench_vec("PointLTL4MASAR1-v0", cls, kw)
            summary["vec_bench"][label] = round(sps, 1)
            _log("H2", "vec_bench", {"label": label, "env_steps_per_sec": round(sps, 1)})
            print(label, sps)
        except Exception as exc:
            _log("H2", "vec_bench_fail", {"label": label, "error": str(exc)})
            summary["vec_bench"][label] = f"error: {exc}"

    summary["obs_paths"] = {}
    for name in ("PointLTL0MASAR1-v0", "PointLTL4MASAR1-v0"):
        row = _bench_obs_paths(name)
        row["env_name"] = name
        summary["obs_paths"][name] = row
        _log("H1", "obs_paths", row)
        print(name, row)

    summary["mini_ppo"] = {}
    for name, ns in (("PointLTL0MASAR1-v0", 512), ("PointLTL4MASAR1-v0", 2048)):
        row = _mini_ppo(name, ns)
        summary["mini_ppo"][f"{name}@n{ns}"] = row
        _log("H4", "mini_ppo", row)
        print("ppo", row)

    l0 = summary["mini_ppo"].get("PointLTL0MASAR1-v0@n512", {})
    l4 = summary["mini_ppo"].get("PointLTL4MASAR1-v0@n2048", {})
    if l0.get("iter_per_sec") and l4.get("iter_per_sec"):
        summary["iter_ratio_l4_over_l0"] = round(
            l4["iter_per_sec"] / l0["iter_per_sec"], 2
        )
    _log("H4", "diag_summary", summary, )
    print("SUMMARY", summary)
    print(f"Wrote {ROOT.parent / 'debug-701dbb.log'}")


if __name__ == "__main__":
    main()
