"""Smoke tests for PPO-Lagrangian OpenAI-fidelity components."""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch as th

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ppo_lagrangian.buffer import LagrangianRolloutBuffer
from ppo_lagrangian.policies import MLPActorCritic, gaussian_entropy, gaussian_kl, gaussian_likelihood
from ppo_lagrangian.ppo_lagrangian import PPOLagrangian
from ppo_lagrangian.utils import EPS, discount_cumsum


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
    for t in range(4):
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
    data = buf.get()
    assert data["adv"].shape == (4,)
    assert data["cadv"].shape == (4,)
    assert abs(float(np.mean(data["adv"]))) < 1e-5
    assert abs(float(np.mean(data["cadv"]))) < 1e-5
    # reward adv normalized to unit std
    assert abs(float(np.std(data["adv"])) - 1.0) < 1e-4
    # cost adv centered only (std not forced to 1)
    assert data["ret"].shape == (4,)
    assert data["cret"].shape == (4,)


def test_gaussian_formulas_match_openai():
    mu = th.zeros(4, 2)
    log_std = th.zeros(2)
    x = th.ones(4, 2) * 0.5
    logp = gaussian_likelihood(x, mu, log_std.expand_as(mu))
    assert logp.shape == (4,)
    ent = gaussian_entropy(log_std.expand_as(mu))
    assert ent.ndim == 0 or ent.numel() == 1
    old_mu = mu + 0.1
    old_log_std = log_std.expand_as(mu)
    kl = gaussian_kl(mu, log_std.expand_as(mu), old_mu, old_log_std)
    assert kl.ndim == 0 or kl.numel() == 1
    assert float(kl.item()) >= 0.0 - 1e-6


def test_learn_predict_save_load(tmp_path=None):
    env = gym.make("Pendulum-v1")
    # Wrap as single-env vec-like via Dummy interface: just use gym and n_envs=1
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
            # inject cost for Lagrangian
            info["cost"] = float(abs(action).sum()) * 0.01
            done = terminated or truncated
            if done:
                obs, _ = self.env.reset()
                info["terminal_observation"] = info.get("terminal_observation", obs)
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

    vec = _Single(env)
    model = PPOLagrangian(
        "MlpPolicy",
        vec,
        n_steps=64,
        n_epochs=2,
        pi_iters=2,
        vf_iters=2,
        learning_rate=3e-4,
        cost_lim=25.0,
        verbose=0,
        seed=0,
        device="cpu",
        policy_kwargs={"net_arch": (32, 32)},
    )
    model.learn(total_timesteps=128, log_interval=1, progress_bar=False)
    obs = vec.reset()[0]
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape[-1] == 1

    out = Path(tmp_path) if tmp_path else Path("_tmp_ppo_lag_test")
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / "model"
    model.save(save_path)
    loaded = PPOLagrangian.load(save_path, env=vec, device="cpu")
    a2, _ = loaded.predict(obs, deterministic=True)
    assert np.allclose(action, a2, atol=1e-5)
    vec.close()
    print("test_learn_predict_save_load OK")


if __name__ == "__main__":
    test_discount_cumsum()
    print("test_discount_cumsum OK")
    test_buffer_gae_and_norm()
    print("test_buffer_gae_and_norm OK")
    test_gaussian_formulas_match_openai()
    print("test_gaussian_formulas_match_openai OK")
    test_learn_predict_save_load()
    print("ALL SMOKE TESTS PASSED")
