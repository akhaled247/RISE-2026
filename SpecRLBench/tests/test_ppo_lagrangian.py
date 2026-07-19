"""Smoke tests for slim PPO-Lagrangian."""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch as th

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ppo_lagrangian.buffer import LagrangianRolloutBuffer, discount_cumsum
from ppo_lagrangian.ppo_lagrangian import PPOLagrangian, _gaussian_entropy, _gaussian_kl, _gaussian_logp


def test_discount_cumsum():
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    out = discount_cumsum(x, 0.9)
    expected = np.array([1 + 0.9 * 2 + 0.81 * 3, 2 + 0.9 * 3, 3.0])
    assert np.allclose(out, expected), (out, expected)


def test_buffer_gae_and_norm():
    buf = LagrangianRolloutBuffer(
        size=4,
        obs_shape=(2,),
        act_shape=(1,),
        pi_info_shapes={"mu": [1], "log_std": [1]},
        gamma=0.99,
        lam=0.97,
        cost_gamma=0.99,
        cost_lam=0.97,
    )
    for _ in range(4):
        buf.store(
            obs=np.zeros(2, dtype=np.float32),
            act=np.zeros(1, dtype=np.float32),
            rew=1.0,
            val=0.5,
            cost=0.1,
            cval=0.2,
            logp=-0.5,
            pi_info={"mu": np.zeros(1, dtype=np.float32), "log_std": np.zeros(1, dtype=np.float32)},
        )
    buf.finish_path(last_val=0.0, last_cval=0.0)
    data = next(buf.get(4))
    assert abs(float(np.mean(data["adv"]))) < 1e-5
    assert abs(float(np.mean(data["cadv"]))) < 1e-5
    assert abs(float(np.std(data["adv"])) - 1.0) < 1e-4
    # Second epoch call reuses once-normalized flat data
    batches = list(buf.get(2))
    assert len(batches) == 2
    assert batches[0]["adv"].shape[0] == 2


def test_gaussian_formulas():
    mu = th.zeros(4, 2)
    log_std = th.zeros(2).expand_as(mu)
    logp = _gaussian_logp(th.ones(4, 2) * 0.5, mu, log_std)
    assert logp.shape == (4,)
    kl = _gaussian_kl(mu, log_std, mu + 0.1, log_std)
    assert float(kl.item()) >= -1e-6
    assert _gaussian_entropy(log_std).numel() == 1


def test_learn_predict_save_load():
    class _Single:
        def __init__(self, e):
            self.env = e
            self.observation_space = e.observation_space
            self.action_space = e.action_space
            self.num_envs = 1

        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            return np.expand_dims(obs, 0), [info]

        def step(self, actions):
            action = actions[0] if actions.ndim > 1 else actions
            obs, reward, terminated, truncated, info = self.env.step(action)
            info["cost"] = float(abs(action).sum()) * 0.01
            done = terminated or truncated
            if done:
                obs, _ = self.env.reset()
                if truncated:
                    info["TimeLimit.truncated"] = True
            return (
                np.expand_dims(obs, 0),
                np.array([reward], dtype=np.float64),
                np.array([done]),
                [info],
            )

        def close(self):
            self.env.close()

    vec = _Single(gym.make("Pendulum-v1"))
    model = PPOLagrangian(
        vec,
        n_steps=64,
        batch_size=32,
        n_epochs=2,
        learning_rate=3e-4,
        cost_lim=25.0,
        seed=0,
        device="cpu",
        hidden_sizes=(32, 32),
        log_interval=1,
    )
    model.learn(total_timesteps=128)
    obs = vec.reset()[0]
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape[-1] == 1

    out = Path("_tmp_ppo_lag_test")
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / "model"
    model.save(save_path)
    loaded = PPOLagrangian.load(save_path, env=vec, device="cpu")
    a2, _ = loaded.predict(obs, deterministic=True)
    assert np.allclose(action, a2, atol=1e-5)
    assert loaded.batch_size == 32
    assert loaded.n_epochs == 2
    vec.close()
    print("test_learn_predict_save_load OK")


if __name__ == "__main__":
    test_discount_cumsum()
    print("test_discount_cumsum OK")
    test_buffer_gae_and_norm()
    print("test_buffer_gae_and_norm OK")
    test_gaussian_formulas()
    print("test_gaussian_formulas OK")
    test_learn_predict_save_load()
    print("ALL SMOKE TESTS PASSED")
