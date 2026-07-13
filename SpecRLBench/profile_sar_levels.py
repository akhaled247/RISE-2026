"""Profile per-step cost: level0 vs level4, walls lidar vs rest."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "specbench" / "envs" / "zones" / "safety-gymnasium"))

import safety_gymnasium  # noqa: F401
from utils.env_utils import make_env

LOG_PATH = ROOT.parent / "debug-968999.log"
SESSION_ID = "968999"
WARMUP = 20
STEPS = 100


def _log(hypothesis_id: str, message: str, data: dict) -> None:
    payload = {
        "sessionId": SESSION_ID,
        "hypothesisId": hypothesis_id,
        "location": "debug/profile_sar_levels.py",
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
        "runId": "profile",
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _profile_env(env_name: str) -> dict:
    env = make_env(env_name, render_mode=None, sb3=True)
    task = env.unwrapped.task

    t0 = time.perf_counter()
    env.reset(seed=0)
    first_reset_s = time.perf_counter() - t0

    # #region agent log
    _log("A", "first_reset", {"env": env_name, "seconds": round(first_reset_s, 3)})
    # #endregion

    # obs() breakdown on one post-warmup state
    for _ in range(WARMUP):
        env.step(env.action_space.sample())

    t_obs = time.perf_counter()
    task.obs()
    full_obs_s = time.perf_counter() - t_obs

    walls = getattr(task, "walls", None)
    t_walls = time.perf_counter()
    if walls is not None:
        task._obs_lidar_pseudo_occluded_new(0, walls)
    walls_occluded_s = time.perf_counter() - t_walls

    t_walls_cheap = time.perf_counter()
    if walls is not None:
        task._obs_lidar_pseudo_new(0, walls.pos)
    walls_pseudo_s = time.perf_counter() - t_walls_cheap

    # #region agent log
    _log(
        "B",
        "obs_breakdown",
        {
            "env": env_name,
            "full_obs_ms": round(full_obs_s * 1000, 2),
            "walls_occluded_ms": round(walls_occluded_s * 1000, 2),
            "walls_pseudo_ms": round(walls_pseudo_s * 1000, 2),
            "wall_count": getattr(walls, "num", 0),
            "obs_keys": len(env.observation_space.spaces),
        },
    )
    # #endregion

    t_step = time.perf_counter()
    for _ in range(STEPS):
        env.step(env.action_space.sample())
    step_s = (time.perf_counter() - t_step) / STEPS

    # #region agent log
    _log(
        "C",
        "steady_step",
        {
            "env": env_name,
            "ms_per_step": round(step_s * 1000, 2),
            "fps_1env": round(1.0 / step_s, 1),
            "eta_500k_8env_min": round((500_000 / (1.0 / step_s * 8)) / 60, 1),
        },
    )
    # #endregion

    env.close()
    return {
        "first_reset_s": first_reset_s,
        "ms_per_step": step_s * 1000,
        "walls_occluded_ms": walls_occluded_s * 1000,
        "walls_pseudo_ms": walls_pseudo_s * 1000,
    }


if __name__ == "__main__":
    print(f"Logging to {LOG_PATH}")
    for name in ("PointLTL0MASAR1-v0", "PointLTL4MASAR1-v0"):
        print(f"\n=== {name} ===")
        try:
            stats = _profile_env(name)
            for k, v in stats.items():
                print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
        except Exception as exc:
            print(f"  FAILED: {exc}")
            _log("E", "profile_failed", {"env": name, "error": str(exc)})
