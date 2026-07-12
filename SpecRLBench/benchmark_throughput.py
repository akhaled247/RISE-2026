"""Measure env throughput to explain PPO training ETA."""

import json
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


# #region agent log
def _dbg(hypothesis_id, message, data):
    try:
        root = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
        payload = {
            "sessionId": "b1323e",
            "hypothesisId": hypothesis_id,
            "location": "benchmark_throughput.py",
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(root / "debug-b1323e.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass
# #endregion


def _bench(vec_env_cls, label: str) -> float:
  env = make_vec_env(
      lambda: Monitor(make_env(ENV_NAME, render_mode=None, sb3=True)),
      n_envs=N_ENVS,
      vec_env_cls=vec_env_cls,
  )
  env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
  obs = env.reset()
  t0 = time.perf_counter()
  for _ in range(WARMUP_STEPS + BENCH_STEPS):
      action = np.array([env.action_space.sample() for _ in range(N_ENVS)])
      obs, rewards, dones, infos = env.step(action)
  elapsed = time.perf_counter() - t0
  steps = BENCH_STEPS * N_ENVS
  sps = steps / elapsed
  env.close()
  _dbg("H1", f"{label} throughput", {
      "vec_env": label,
      "n_envs": N_ENVS,
      "bench_steps_per_env": BENCH_STEPS,
      "total_env_steps": steps,
      "elapsed_sec": round(elapsed, 3),
      "steps_per_sec": round(sps, 2),
      "eta_500k_hours": round((TOTAL_TIMESTEPS / sps) / 3600, 2),
  })
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
  _dbg('H6', f'{label} throughput with periodic reset', {
      'vec_env': label,
      'reset_every': reset_every,
      'reset_count_total': reset_count,
      'elapsed_sec': round(elapsed, 3),
      'steps_per_sec': round(sps, 2),
      'eta_500k_hours': round((TOTAL_TIMESTEPS / sps) / 3600, 2),
  })
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
      _dbg("H1", "SubprocVecEnv failed", {"error": str(exc)})
      print(f"SubprocVecEnv failed: {exc}")
  _dbg("H3", "PPO update cost estimate", {
      "n_steps": 512,
      "n_epochs": 10,
      "batch_size": 256,
      "rollout_size": 512 * N_ENVS,
      "note": "Each PPO iteration = rollout + 10 gradient epochs over 4096 samples",
  })
