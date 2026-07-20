"""Smoke tests for SB3 Tier-2 PPO-Lagrangian."""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ppo_lagrangian.buffer import LagRolloutBuffer
from ppo_lagrangian.ppo_lagrangian import PPOLag


def test_lag_buffer_gae_and_norm():
    buf = LagRolloutBuffer(
        buffer_size=4,
        observation_space=spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
        action_space=spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
        device="cpu",
        gae_lambda=0.97,
        gamma=0.99,
        n_envs=1,
        cost_gamma=0.99,
        cost_gae_lambda=0.97,
        normalize_advantage_once=True,
    )
    for i in range(4):
        obs = np.zeros((1, 2), dtype=np.float32)
        act = np.zeros((1, 1), dtype=np.float32)
        rew = np.array([float(i + 1)], dtype=np.float32)
        ep_start = np.array([i == 0], dtype=np.float32)
        val = th.tensor([0.5 * i])
        logp = th.tensor([-0.5])
        cost = np.array([0.1 * (i + 1)], dtype=np.float32)
        cval = th.tensor([0.2])
        buf.add(obs, act, rew, ep_start, val, logp, cost=cost, cost_value=cval)
    last_v = th.tensor([0.0])
    last_cv = th.tensor([0.0])
    dones = np.array([True])
    buf.compute_returns_and_advantage(last_v, dones)
    buf.compute_cost_returns_and_advantage(last_cv, dones)
    data = next(buf.get(4))
    assert abs(float(data.advantages.mean())) < 1e-5
    assert abs(float(data.cost_advantages.mean())) < 1e-5
    # numpy population-normalized; torch.std defaults to unbiased (N-1)
    assert abs(float(data.advantages.std(unbiased=False)) - 1.0) < 1e-4
    batches = list(buf.get(2))
    assert len(batches) == 2


def test_learn_predict_save_load_zip():
    def _make():
        return Monitor(gym.make("Pendulum-v1"))

    vec = DummyVecEnv([_make])
    model = PPOLag(
        "MlpPolicy",
        vec,
        n_steps=64,
        batch_size=32,
        n_epochs=2,
        learning_rate=3e-4,
        cost_lim=25.0,
        seed=0,
        device="cpu",
        policy_kwargs=dict(net_arch=dict(pi=[32, 32], vf=[32, 32]), activation_fn=nn.Tanh),
        lag_mode="openai",
        verbose=0,
    )
    # Inject cost into infos via wrapper-less Monitor — cost_fn default reads info['cost']
    # Pendulum infos lack cost → always 0; still smoke-tests Lag loop
    model.learn(total_timesteps=128)
    obs = vec.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape[-1] == 1

    out = Path("_tmp_ppo_lag_test")
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / "model"
    model.save(save_path)
    assert save_path.with_suffix(".zip").exists() or Path(str(save_path) + ".zip").exists() or save_path.exists()

    loaded = PPOLag.load(save_path, env=vec, device="cpu")
    a2, _ = loaded.predict(obs, deterministic=True)
    assert np.allclose(action, a2, atol=1e-5)
    assert loaded.batch_size == 32
    assert loaded.n_epochs == 2
    assert loaded.lag_mode == "openai"
    vec.close()
    print("test_learn_predict_save_load_zip OK")


def test_legacy_pt_rejected():
    out = Path("_tmp_ppo_lag_test")
    out.mkdir(parents=True, exist_ok=True)
    pt = out / "legacy.pt"
    th.save({"cfg": {}}, pt)
    try:
        PPOLag.load(pt, env=None, device="cpu")
        raise AssertionError("expected ValueError for .pt")
    except ValueError as e:
        assert "Legacy" in str(e) or ".pt" in str(e)
    print("test_legacy_pt_rejected OK")


def test_lag_mode_sb3_smoke():
    def _make():
        return Monitor(gym.make("Pendulum-v1"))

    vec = DummyVecEnv([_make])
    model = PPOLag(
        "MlpPolicy",
        vec,
        n_steps=64,
        batch_size=32,
        n_epochs=1,
        learning_rate=3e-4,
        device="cpu",
        seed=1,
        policy_kwargs=dict(net_arch=[32, 32]),
        lag_mode="sb3",
        verbose=0,
    )
    model.learn(total_timesteps=64)
    vec.close()
    print("test_lag_mode_sb3_smoke OK")


if __name__ == "__main__":
    test_lag_buffer_gae_and_norm()
    print("test_lag_buffer_gae_and_norm OK")
    test_learn_predict_save_load_zip()
    test_legacy_pt_rejected()
    test_lag_mode_sb3_smoke()
    print("ALL SMOKE TESTS PASSED")
