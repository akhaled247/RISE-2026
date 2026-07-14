"""One-shot perf diagnostic for PPO slowdown (debug session 701dbb).

Run from SpecRLBench/ on the Linux training host:
    python debug/debug_perf_701dbb.py
    python debug/debug_perf_701dbb.py --config l0
    python debug/debug_perf_701dbb.py --config l4
    python debug/debug_perf_701dbb.py --config current
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT.parent / "debug-701dbb.log"
sys.path.insert(0, str(ROOT / "specbench" / "envs" / "zones" / "safety-gymnasium"))
sys.path.insert(0, str(ROOT))

import safety_gymnasium  # noqa: F401
from debug.debug_log import agent_log
from utils.env_utils import make_env, make_vec

N_ENVS = 8
WARMUP = 30
BENCH_STEPS = 200
MINI_TRAIN_STEPS = 16_384

# Compare configs that explain ~2k vs ~200 iters/s
PPO_CONFIGS = {
    "l0": {
        "env_name": "PointLTL0MASAR1-v0",
        "n_steps": 512,
        "n_epochs": 10,
        "learning_rate": 3e-4,
        "ent_coef": 0.01,
        "target_kl": 0.02,
    },
    "l4": {
        "env_name": "PointLTL4MASAR1-v0",
        "n_steps": 2048,
        "n_epochs": 10,
        "learning_rate": 5e-5,
        "ent_coef": 0.02,
        "target_kl": 0.03,
    },
    "current": None,  # filled from ppo_train_env at runtime
}


def _load_current_config() -> dict:
    import ppo_train_env as train_mod
    return {
        "env_name": train_mod.env_name,
        "n_steps": train_mod.n_steps,
        "n_epochs": train_mod.n_epochs,
        "learning_rate": train_mod.learning_rate,
        "ent_coef": train_mod.ent_coef,
        "target_kl": 0.03,
    }


def _log(hypothesis_id: str, message: str, data: dict, run_id: str = "diag") -> None:
    agent_log(f"debug_perf_701dbb.py:{message}", message, data, hypothesis_id, run_id)


def _task_limits(env) -> dict:
    unwrapped = env.unwrapped
    while hasattr(unwrapped, "env"):
        unwrapped = unwrapped.env
    task = getattr(unwrapped, "task", None)
    return {
        "spec_max_episode_steps": getattr(getattr(env, "spec", None), "max_episode_steps", None),
        "task_num_steps": getattr(task, "num_steps", None),
        "task_class": type(task).__name__ if task else None,
        "lidar_type": getattr(getattr(task, "lidar_conf", None), "type", None),
        "wall_count": getattr(getattr(task, "walls", None), "num", 0),
    }


def _bench_vec_env(env_name: str, vec_env_cls, vec_env_kwargs: dict | None = None) -> dict:
    vec_env_kwargs = dict(vec_env_kwargs or {})
    env = make_vec_env(
        lambda: Monitor(make_env(env_name, render_mode=None, sb3=True)),
        n_envs=N_ENVS,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs or None,
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    limits = _task_limits(env.envs[0])
    env.reset()
    for _ in range(WARMUP):
        env.step(env.action_space.sample())
    t0 = time.perf_counter()
    for _ in range(BENCH_STEPS):
        env.step(env.action_space.sample())
    elapsed = time.perf_counter() - t0
    env_steps = BENCH_STEPS * N_ENVS
    sps = env_steps / elapsed
    env.close()
    return {
        "vec_env": vec_env_cls.__name__,
        "vec_env_kwargs": vec_env_kwargs,
        "env_steps_per_sec": round(sps, 1),
        "ms_per_vec_step": round(elapsed / BENCH_STEPS * 1000, 2),
        **limits,
    }


def _bench_single_env_obs(env_name: str) -> dict:
    env = make_env(env_name, render_mode=None, sb3=True)
    task = env.unwrapped.task
    limits = _task_limits(env)
    env.reset(seed=0)
    for _ in range(WARMUP):
        env.step(env.action_space.sample())

    t_reward = time.perf_counter()
    task.calculate_reward()
    reward_ms = (time.perf_counter() - t_reward) * 1000

    t_obs = time.perf_counter()
    task.obs()
    obs_ms = (time.perf_counter() - t_obs) * 1000

    walls = getattr(task, "walls", None)
    walls_pseudo_ms = None
    walls_occluded_ms = None
    if walls is not None:
        t0 = time.perf_counter()
        task._obs_lidar_pseudo_new(0, walls.pos)
        walls_pseudo_ms = round((time.perf_counter() - t0) * 1000, 3)
        t0 = time.perf_counter()
        task._obs_lidar_pseudo_occluded_new(0, walls)
        walls_occluded_ms = round((time.perf_counter() - t0) * 1000, 3)

    casualty = getattr(task, "surface_casualtys", None)
    casualty_occluded_ms = None
    if casualty is not None:
        t0 = time.perf_counter()
        task._obs_lidar_pseudo_occluded_new(0, casualty)
        casualty_occluded_ms = round((time.perf_counter() - t0) * 1000, 3)

    t_step = time.perf_counter()
    for _ in range(BENCH_STEPS):
        env.step(env.action_space.sample())
    step_ms = (time.perf_counter() - t_step) / BENCH_STEPS * 1000
    env.close()
    return {
        **limits,
        "calculate_reward_ms": round(reward_ms, 3),
        "obs_ms": round(obs_ms, 3),
        "walls_pseudo_ms": walls_pseudo_ms,
        "walls_occluded_ms": walls_occluded_ms,
        "casualty_occluded_ms": casualty_occluded_ms,
        "ms_per_step_1env": round(step_ms, 3),
        "fps_1env": round(1000 / step_ms, 1),
    }


def _bench_mini_ppo(label: str, cfg: dict) -> dict:
    env_name = cfg["env_name"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_vec(env_name, n_envs=N_ENVS, render_mode=None, sb3=True, normalize=True)
    limits = _task_limits(env.envs[0])
    env.reset()
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=0,
        learning_rate=cfg["learning_rate"],
        n_steps=cfg["n_steps"],
        batch_size=256,
        n_epochs=cfg["n_epochs"],
        ent_coef=cfg["ent_coef"],
        target_kl=cfg["target_kl"],
        device=device,
        seed=0,
    )
    t0 = time.perf_counter()
    model.learn(total_timesteps=MINI_TRAIN_STEPS, progress_bar=False)
    elapsed = time.perf_counter() - t0
    iters = MINI_TRAIN_STEPS // (cfg["n_steps"] * N_ENVS)
    env.close()
    return {
        "label": label,
        "env_name": env_name,
        "device": device,
        "n_steps": cfg["n_steps"],
        "n_epochs": cfg["n_epochs"],
        "ent_coef": cfg["ent_coef"],
        "mini_train_steps": MINI_TRAIN_STEPS,
        "ppo_iterations": iters,
        "seconds": round(elapsed, 2),
        "iter_per_sec": round(iters / elapsed, 2),
        "env_steps_per_sec": round(MINI_TRAIN_STEPS / elapsed, 1),
        **limits,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        choices=["l0", "l4", "current", "all"],
        default="all",
        help="Which PPO config to mini-benchmark (default: all)",
    )
    args = parser.parse_args()

    PPO_CONFIGS["current"] = _load_current_config()

    _log("H0", "diag_start", {"platform": sys.platform, "cuda": torch.cuda.is_available()}, "diag")

    # H3: vec-env parallelism (L0 env, isolate vec-env effect)
    for cls, kwargs, label in (
        (DummyVecEnv, {}, "dummy"),
        (SubprocVecEnv, {}, "subproc_default"),
        (SubprocVecEnv, {"start_method": "fork"}, "subproc_fork"),
    ):
        try:
            row = _bench_vec_env("PointLTL0MASAR1-v0", cls, kwargs)
            row["label"] = label
            _log("H3", "vec_env_bench", row, "diag")
            print("vec_env", row)
        except Exception as exc:
            _log("H3", "vec_env_bench_failed", {"label": label, "error": str(exc)}, "diag")
            print("vec_env FAILED", label, exc)

    # H2: per-step obs / pseudo_occluded cost
    for env_name in ("PointLTL0MASAR1-v0", "PointLTL4MASAR1-v0"):
        try:
            row = _bench_single_env_obs(env_name)
            _log("H2", "single_env_breakdown", {"env_name": env_name, **row}, "diag")
            print("breakdown", env_name, row)
        except Exception as exc:
            _log("H2", "single_env_failed", {"env_name": env_name, "error": str(exc)}, "diag")
            print("breakdown FAILED", env_name, exc)

    # H4: PPO hyperparams dominate iteration rate (not ent_coef)
    configs_to_run = (
        list(PPO_CONFIGS.keys()) if args.config == "all" else [args.config]
    )
    for label in configs_to_run:
        cfg = PPO_CONFIGS[label]
        try:
            row = _bench_mini_ppo(label, cfg)
            _log("H4", "mini_ppo", row, "diag")
            print("mini_ppo", row)
        except Exception as exc:
            _log("H4", "mini_ppo_failed", {"label": label, "error": str(exc)}, "diag")
            print("mini_ppo FAILED", label, exc)

    # H3: actual make_vec path used by ppo_train_env
    try:
        env = make_vec(PPO_CONFIGS["current"]["env_name"], n_envs=N_ENVS, sb3=True)
        vec_cls = type(env.venv).__name__
        start_method = getattr(env.venv, "start_method", None)
        _log(
            "H3",
            "make_vec_training_path",
            {"vec_cls": vec_cls, "start_method": start_method, "env_name": PPO_CONFIGS["current"]["env_name"]},
            "diag",
        )
        print("make_vec", vec_cls, start_method)
        env.close()
    except Exception as exc:
        _log("H3", "make_vec_failed", {"error": str(exc)}, "diag")

    print(f"\nWrote logs to {LOG_PATH}")


if __name__ == "__main__":
    main()
