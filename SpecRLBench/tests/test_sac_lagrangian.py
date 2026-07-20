"""Smoke tests for SAC-Lagrangian."""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sac_lagrangian import SACLag


def test_learn_predict_save_load_zip():
    def _make():
        return Monitor(gym.make("Pendulum-v1"))

    vec = DummyVecEnv([_make])
    model = SACLag(
        "MlpPolicy",
        vec,
        learning_starts=64,
        buffer_size=10_000,
        batch_size=64,
        train_freq=1,
        gradient_steps=1,
        learning_rate=3e-4,
        cost_lim=25.0,
        seed=0,
        device="cpu",
        policy_kwargs=dict(net_arch=[32, 32]),
        verbose=0,
    )
    model.learn(total_timesteps=256)
    obs = vec.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape[-1] == 1

    # Replay stores costs
    assert hasattr(model.replay_buffer, "costs")
    assert model.replay_buffer.costs.shape[0] == model.buffer_size

    out = Path("_tmp_sac_lag_test")
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / "model"
    model.save(save_path)

    loaded = SACLag.load(save_path, env=vec, device="cpu")
    a2, _ = loaded.predict(obs, deterministic=True)
    assert np.allclose(action, a2, atol=1e-5)
    assert loaded.cost_lim == 25.0
    assert abs(float(loaded.penalty.item()) - float(model.penalty.item())) < 1e-4
    vec.close()
    print("test_learn_predict_save_load_zip OK")


def test_actor_uses_qc_term():
    """Smoke: policy has cost_q_mean and train runs without OpenAI Qc-constraint dual."""
    vec = DummyVecEnv([lambda: Monitor(gym.make("Pendulum-v1"))])
    model = SACLag(
        "MlpPolicy",
        vec,
        learning_starts=32,
        buffer_size=1000,
        batch_size=32,
        train_freq=1,
        gradient_steps=2,
        device="cpu",
        seed=1,
        policy_kwargs=dict(net_arch=[32, 32]),
        verbose=0,
    )
    assert hasattr(model.policy, "cost_q_mean")
    model.learn(total_timesteps=96)
    # Dual uses EpCost path (penalty still finite / non-neg)
    assert float(model.penalty.item()) >= 0.0
    vec.close()
    print("test_actor_uses_qc_term OK")


if __name__ == "__main__":
    test_learn_predict_save_load_zip()
    test_actor_uses_qc_term()
    print("ALL SAC LAG SMOKE TESTS PASSED")
