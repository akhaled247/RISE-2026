"""Compare PPO training throughput across vec-env and hyperparam configs."""

import json
import sys
import time
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT.parent / "debug-3376cb.log"
sys.path.insert(0, str(ROOT / "specbench" / "envs" / "zones" / "safety-gymnasium"))
sys.path.insert(0, str(ROOT))

import safety_gymnasium  # noqa: F401
from utils.env_utils import make_env

ENV_NAME = "PointLTL0MASAR1-v0"
N_ENVS = 8
TRAIN_STEPS = 16_384  # two PPO iterations @ n_steps=1024, or four @ 512


def _log(hypothesis_id: str, message: str, data: dict, run_id: str = "ppo-bench") -> None:
    # #region agent log
    entry = {
        "sessionId": "3376cb",
        "timestamp": int(time.time() * 1000),
        "location": "benchmark_ppo_training.py",
        "message": message,
        "data": data,
        "hypothesisId": hypothesis_id,
        "runId": run_id,
    }
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    # #endregion


def _env_only_sps(vec_env_cls, vec_env_kwargs=None, steps=200) -> float:
    vec_env_kwargs = vec_env_kwargs or {}
    env = make_vec_env(
        lambda: Monitor(make_env(ENV_NAME, render_mode=None, sb3=True)),
        n_envs=N_ENVS,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs,
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    env.reset()
    t0 = time.perf_counter()
    for _ in range(steps):
        env.step(env.action_space.sample())
    elapsed = time.perf_counter() - t0
    env.close()
    return (steps * N_ENVS) / elapsed


def _ppo_fps(
    label: str,
    vec_env_cls,
    *,
    n_steps: int,
    n_epochs: int,
    device: str,
    vec_env_kwargs=None,
) -> dict:
    vec_env_kwargs = vec_env_kwargs or {}
    env = make_vec_env(
        lambda: Monitor(make_env(ENV_NAME, render_mode=None, sb3=True)),
        n_envs=N_ENVS,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs,
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    env.reset()

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=0,
        learning_rate=1e-4,
        n_steps=n_steps,
        batch_size=256,
        n_epochs=n_epochs,
        ent_coef=0.01,
        target_kl=0.02,
        device=device,
        seed=0,
    )

    t0 = time.perf_counter()
    model.learn(total_timesteps=TRAIN_STEPS, progress_bar=False)
    elapsed = time.perf_counter() - t0
    env.close()
    fps = TRAIN_STEPS / elapsed
    result = {
        "label": label,
        "vec_env": vec_env_cls.__name__,
        "n_steps": n_steps,
        "n_epochs": n_epochs,
        "device": device,
        "vec_env_kwargs": vec_env_kwargs,
        "train_steps": TRAIN_STEPS,
        "seconds": round(elapsed, 2),
        "fps": round(fps, 1),
        "eta_500k_min": round((500_000 / fps) / 60, 1),
    }
    _log("T1", "ppo_config", result, "ppo-bench")
    print(result)
    return result


if __name__ == "__main__":
    _log("T0", "benchmark_start", {"env": ENV_NAME, "n_envs": N_ENVS}, "ppo-bench")

    for cls, kwargs, label in (
        (DummyVecEnv, {}, "dummy"),
        (SubprocVecEnv, {}, "subproc_spawn"),
        (SubprocVecEnv, {"start_method": "fork"}, "subproc_fork"),
    ):
        try:
            sps = _env_only_sps(cls, kwargs)
            row = {"label": label, "env_only_sps": round(sps, 1)}
            _log("T2", "env_only", row, "ppo-bench")
            print("env_only", row)
        except Exception as exc:
            _log("T2", "env_only_failed", {"label": label, "error": str(exc)}, "ppo-bench")
            print("env_only FAILED", label, exc)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    configs = [
        ("dummy_n512_e10", DummyVecEnv, 512, 10, {}),
        ("subproc_fork_n512_e10", SubprocVecEnv, 512, 10, {"start_method": "fork"}),
        ("subproc_fork_n1024_e4", SubprocVecEnv, 1024, 4, {"start_method": "fork"}),
        ("dummy_n1024_e4", DummyVecEnv, 1024, 4, {}),
        ("subproc_fork_n1024_e4_cpu", SubprocVecEnv, 1024, 4, {"start_method": "fork"}),
    ]
    for label, cls, n_steps, n_epochs, kwargs in configs:
        dev = "cpu" if label.endswith("_cpu") else device
        try:
            _ppo_fps(label, cls, n_steps=n_steps, n_epochs=n_epochs, device=dev, vec_env_kwargs=kwargs)
        except Exception as exc:
            _log("T1", "ppo_failed", {"label": label, "error": str(exc)}, "ppo-bench")
            print("ppo FAILED", label, exc)
