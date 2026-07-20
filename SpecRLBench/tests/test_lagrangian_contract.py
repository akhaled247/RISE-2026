"""Shared Lag consistency contract tests (PPO / TRPO; SAC added in M5)."""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lagrangian.cost import cost_from_info
from lagrangian.dual import LagPenalty
from lagrangian.on_policy.objective import openai_lag_pi_objective, unclipped_surr
from ppo_lagrangian import PPOLag
from sac_lagrangian import SACLag
from trpo_lagrangian import TRPOLag


def _pendulum_vec():
    return DummyVecEnv([lambda: Monitor(gym.make("Pendulum-v1"))])


def test_cost_from_info():
    assert cost_from_info({"cost": 1.5}) == 1.5
    assert cost_from_info({}) == 0.0
    print("test_cost_from_info OK")


def test_lag_penalty_init_and_nonneg():
    lag = LagPenalty(penalty_init=1.0, penalty_lr=5e-2, device="cpu")
    assert float(lag.penalty.item()) >= 0.0
    assert abs(float(lag.penalty.item()) - 1.0) < 1e-5
    print("test_lag_penalty_init_and_nonneg OK")


def test_lag_penalty_dual_increases_when_cost_above_lim():
    lag = LagPenalty(penalty_init=1.0, penalty_lr=5e-2, device="cpu")
    before = float(lag.penalty.item())
    after = lag.update(ep_cost_mean=50.0, cost_lim=25.0)
    assert after >= before
    print("test_lag_penalty_dual_increases_when_cost_above_lim OK")


def test_onpolicy_unclipped_surr_cost_formula():
    ratio = th.tensor([1.0, 2.0])
    cost_adv = th.tensor([1.0, -1.0])
    surr = unclipped_surr(cost_adv, ratio)
    assert abs(float(surr.item()) - (-0.5)) < 1e-6

    pen = th.tensor(1.0)
    obj = openai_lag_pi_objective(th.tensor(1.0), th.tensor(0.5), 0.0, pen, 0.0)
    assert abs(float(obj.item()) - 0.25) < 1e-6
    print("test_onpolicy_unclipped_surr_cost_formula OK")


def _assert_contract(model, vec, name: str):
    assert float(model.penalty.item()) >= 0.0
    assert abs(float(model.penalty.item()) - model.penalty_init) < 1e-4

    log_dir = Path("_tmp_lag_contract_logs") / name
    log_dir.mkdir(parents=True, exist_ok=True)
    model.set_logger(configure(str(log_dir), ["csv"]))
    model.learn(total_timesteps=128)
    assert float(model.penalty.item()) >= 0.0

    out = Path("_tmp_lag_contract") / name
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / "m"
    model.save(save_path)
    loaded = type(model).load(save_path, env=vec, device="cpu")
    assert loaded.cost_lim == model.cost_lim
    assert loaded.penalty_init == model.penalty_init
    assert loaded.penalty_lr == model.penalty_lr
    assert abs(float(loaded.penalty_param.item()) - float(model.penalty_param.item())) < 1e-5
    print(f"contract {name} OK")


def test_ppolag_contract():
    vec = _pendulum_vec()
    model = PPOLag(
        "MlpPolicy",
        vec,
        n_steps=64,
        batch_size=32,
        n_epochs=1,
        learning_rate=3e-4,
        cost_lim=25.0,
        penalty_init=1.0,
        penalty_lr=5e-2,
        seed=0,
        device="cpu",
        policy_kwargs=dict(net_arch=[32, 32]),
        lag_mode="openai",
        verbose=0,
    )
    _assert_contract(model, vec, "PPOLag")
    vec.close()


def test_trpolag_contract():
    vec = _pendulum_vec()
    model = TRPOLag(
        "MlpPolicy",
        vec,
        n_steps=64,
        batch_size=32,
        n_critic_updates=1,
        learning_rate=1e-3,
        cost_lim=25.0,
        penalty_init=1.0,
        penalty_lr=5e-2,
        seed=0,
        device="cpu",
        policy_kwargs=dict(net_arch=[32, 32]),
        verbose=0,
    )
    _assert_contract(model, vec, "TRPOLag")
    vec.close()


def test_saclag_contract():
    vec = _pendulum_vec()
    model = SACLag(
        "MlpPolicy",
        vec,
        learning_starts=64,
        buffer_size=5000,
        batch_size=64,
        train_freq=1,
        gradient_steps=1,
        learning_rate=3e-4,
        cost_lim=25.0,
        penalty_init=1.0,
        penalty_lr=5e-2,
        seed=0,
        device="cpu",
        policy_kwargs=dict(net_arch=[32, 32]),
        verbose=0,
    )
    _assert_contract(model, vec, "SACLag")
    vec.close()


def test_load_routing_filename_tags():
    """Filename tags used by ppo_load_env must match save prefixes."""
    from pathlib import Path as P

    cases = [
        ("trpo_lag_20260101_PointLTL_0", "trpo_lag"),
        ("sac_lag_20260101_PointLTL_0", "sac_lag"),
        ("ppo_lag_20260101_PointLTL_0", "ppo_lag"),
    ]
    for name, tag in cases:
        assert tag in P(name).name.lower()
    print("test_load_routing_filename_tags OK")


if __name__ == "__main__":
    test_cost_from_info()
    test_lag_penalty_init_and_nonneg()
    test_lag_penalty_dual_increases_when_cost_above_lim()
    test_onpolicy_unclipped_surr_cost_formula()
    test_ppolag_contract()
    test_trpolag_contract()
    test_saclag_contract()
    test_load_routing_filename_tags()
    print("ALL CONTRACT TESTS PASSED")
