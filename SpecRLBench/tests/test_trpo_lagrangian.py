"""Smoke tests for TRPO-Lagrangian."""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trpo_lagrangian import TRPOLag


def test_learn_predict_save_load_zip():
    def _make():
        return Monitor(gym.make("Pendulum-v1"))

    vec = DummyVecEnv([_make])
    model = TRPOLag(
        "MlpPolicy",
        vec,
        n_steps=64,
        batch_size=32,
        n_critic_updates=2,
        learning_rate=1e-3,
        cost_lim=25.0,
        seed=0,
        device="cpu",
        policy_kwargs=dict(net_arch=dict(pi=[32, 32], vf=[32, 32]), activation_fn=nn.Tanh),
        verbose=0,
    )
    model.learn(total_timesteps=128)
    obs = vec.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape[-1] == 1

    out = Path("_tmp_trpo_lag_test")
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / "model"
    model.save(save_path)

    loaded = TRPOLag.load(save_path, env=vec, device="cpu")
    a2, _ = loaded.predict(obs, deterministic=True)
    assert np.allclose(action, a2, atol=1e-5)
    assert loaded.cost_lim == 25.0
    assert abs(float(loaded.penalty.item()) - float(model.penalty.item())) < 1e-4
    vec.close()
    print("test_learn_predict_save_load_zip OK")


if __name__ == "__main__":
    test_learn_predict_save_load_zip()
    print("ALL TRPO LAG SMOKE TESTS PASSED")
