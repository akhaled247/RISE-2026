"""Smoke: ppo_load_env filename routing for Lag algos (Pendulum, no WC)."""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ppo_lagrangian import PPOLag
from sac_lagrangian import SACLag
from trpo_lagrangian import TRPOLag


def _route_load(model_path: str, vec_env, device: str = "cpu"):
    """Mirror ppo_load_env.eval_model routing (without VecNormalize)."""
    name_l = Path(model_path).name.lower()
    if "trpo_lag" in name_l or "trpolagrangian" in name_l:
        return TRPOLag.load(model_path, env=vec_env, device=device)
    if "sac_lag" in name_l or "saclag" in name_l:
        return SACLag.load(model_path, env=vec_env, device=device)
    if "ppo_lag" in name_l or "ppolagrangian" in name_l:
        return PPOLag.load(model_path, env=vec_env, device=device)
    raise AssertionError(f"unexpected path {model_path}")


def test_filename_load_smoke():
    vec = DummyVecEnv([lambda: Monitor(gym.make("Pendulum-v1"))])
    out = Path("_tmp_lag_load_route")
    out.mkdir(parents=True, exist_ok=True)

    configs = [
        (
            "ppo_lag_smoke",
            PPOLag(
                "MlpPolicy",
                vec,
                n_steps=32,
                batch_size=16,
                n_epochs=1,
                device="cpu",
                seed=0,
                policy_kwargs=dict(net_arch=[16, 16]),
                verbose=0,
            ),
            64,
        ),
        (
            "trpo_lag_smoke",
            TRPOLag(
                "MlpPolicy",
                vec,
                n_steps=32,
                batch_size=16,
                n_critic_updates=1,
                device="cpu",
                seed=0,
                policy_kwargs=dict(net_arch=[16, 16]),
                verbose=0,
            ),
            64,
        ),
        (
            "sac_lag_smoke",
            SACLag(
                "MlpPolicy",
                vec,
                learning_starts=32,
                buffer_size=1000,
                batch_size=32,
                train_freq=1,
                gradient_steps=1,
                device="cpu",
                seed=0,
                policy_kwargs=dict(net_arch=[16, 16]),
                verbose=0,
            ),
            96,
        ),
    ]

    for prefix, model, steps in configs:
        model.learn(total_timesteps=steps)
        path = out / prefix
        model.save(path)
        loaded = _route_load(str(path), vec)
        assert type(loaded).__name__ == type(model).__name__
        print(f"load route {prefix} OK")

    vec.close()
    print("ALL LOAD ROUTE SMOKE PASSED")


if __name__ == "__main__":
    test_filename_load_smoke()
