"""Tests for SpecRL MA recipe-B overrides."""

from __future__ import annotations

import os

import torch

from backends.safepo.config import MA_SPECRL_RECIPE_B
from backends.safepo.ma_runners import _apply_specrl_ma_recipe_b


def test_apply_specrl_ma_recipe_b_masar2():
    cfg = {"episode_length": 1000, "actor_lr": 9e-5, "target_kl": 0.016}
    _apply_specrl_ma_recipe_b(cfg, "PointLTL0MASAR2-v0")
    assert cfg["episode_length"] == MA_SPECRL_RECIPE_B["episode_length"]
    assert cfg["actor_lr"] == MA_SPECRL_RECIPE_B["actor_lr"]
    assert cfg["target_kl"] == MA_SPECRL_RECIPE_B["target_kl"]
    assert cfg["entropy_coef"] == MA_SPECRL_RECIPE_B["entropy_coef"]


def test_apply_specrl_ippo_recipe_ec0():
    from backends.safepo.config import MA_IPPO_RECIPE_B

    cfg = {"entropy_coef": 0.02, "ent_coef": 0.02}
    _apply_specrl_ma_recipe_b(cfg, "PointLTL0MASAR2-v0", algo="ippo")
    assert cfg["entropy_coef"] == 0.0
    assert cfg["ent_coef"] == 0.0
    assert cfg["eval_interval"] == MA_IPPO_RECIPE_B["eval_interval"]


def test_apply_specrl_ma_recipe_b_skips_non_specrl():
    cfg = {"episode_length": 1000}
    _apply_specrl_ma_recipe_b(cfg, "Safety2x4AntVelocity-v0")
    assert cfg["episode_length"] == 1000


def test_apply_specrl_ma_recipe_b_cli_override():
    cfg = {}
    _apply_specrl_ma_recipe_b(
        cfg,
        "PointLTL1MASAR2WC-v0",
        overrides={"episode_length": 2000, "learning_iters": 8},
    )
    assert cfg["episode_length"] == 2000
    assert cfg["learning_iters"] == 8
    assert cfg["gamma"] == MA_SPECRL_RECIPE_B["gamma"]


def test_ippo_cfg_train_to_ppo_config():
    from safepo.multi_agent.ippo import _cfg_train_to_ppo_config

    cfg = {
        "n_rollout_threads": 8,
        "episode_length": 4096,
        "num_env_steps": 4_000_000,
        "entropy_coef": 0.02,
        "clip_param": 0.2,
        "hidden_size": 64,
        "num_mini_batch": 1,
        "batch_size": 256,  # ignored (MAPPO-style num_mini_batch wins)
    }
    ppo = _cfg_train_to_ppo_config(cfg)
    assert ppo["steps_per_epoch"] == 32768
    assert ppo["local_steps_per_epoch"] == 4096
    assert ppo["ent_coef"] == 0.02
    assert ppo["clip_ratio"] == 0.2
    assert ppo["num_mini_batch"] == 1
    assert "batch_size" not in ppo
    assert ppo["hidden_sizes"] == [64, 64]


def test_ippo_num_mini_batch_default():
    from safepo.multi_agent.ippo import _cfg_train_to_ppo_config

    ppo = _cfg_train_to_ppo_config(
        {"n_rollout_threads": 8, "episode_length": 4096}
    )
    assert ppo["num_mini_batch"] == 1
    assert MA_SPECRL_RECIPE_B["num_mini_batch"] == 1


def test_ippo_ent_coef_alias():
    from safepo.multi_agent.ippo import _cfg_train_to_ppo_config

    ppo = _cfg_train_to_ppo_config({"ent_coef": 0.05, "n_rollout_threads": 1, "episode_length": 100})
    assert ppo["ent_coef"] == 0.05


def test_ippo_should_record_episode():
    from safepo.multi_agent.ippo import _should_record_episode

    horizon = 4096
    assert _should_record_episode(
        done=True, epoch_end=False, episode_steps=600, rollout_horizon=horizon
    )
    assert _should_record_episode(
        done=True, epoch_end=True, episode_steps=4096, rollout_horizon=horizon
    )
    assert _should_record_episode(
        done=False, epoch_end=True, episode_steps=4096, rollout_horizon=horizon
    )
    assert not _should_record_episode(
        done=False, epoch_end=True, episode_steps=1900, rollout_horizon=horizon
    )
    assert not _should_record_episode(
        done=False, epoch_end=False, episode_steps=4096, rollout_horizon=horizon
    )


def test_sa_style_eplen_deque_includes_timeout():
    """Rolling EpLen mean includes terminal/timeout completions (SA parity)."""
    import os

    from safepo.common.logger import EpochLogger
    from safepo.multi_agent.ma_episode_metrics import (
        make_metric_deques,
        record_sa_style_episode_metrics,
    )

    logger = EpochLogger(log_dir=os.devnull, seed="0")
    rew_deque, cost_deque, len_deque = make_metric_deques()
    # Short success then env-max timeout (2500)
    record_sa_style_episode_metrics(logger, rew_deque, cost_deque, len_deque, 1.0, 0.0, 800.0)
    record_sa_style_episode_metrics(logger, rew_deque, cost_deque, len_deque, 0.5, 0.0, 2500.0)
    assert list(len_deque) == [800.0, 2500.0]
    assert logger.epoch_dict["Metrics/EpLen"][-1] == (800.0 + 2500.0) / 2.0
    assert MA_SPECRL_RECIPE_B["episode_length"] == 4096


def test_ippo_ppo_update_changes_actor_weights():
    from gymnasium.spaces import Box

    from safepo.common.buffer import VectorizedOnPolicyBuffer
    from safepo.common.logger import EpochLogger
    from safepo.common.model import ActorVCritic
    from safepo.multi_agent.ippo import AgentPPOBundle, _ppo_update_agent

    device = torch.device("cpu")
    obs_dim, act_dim = 8, 2
    obs_space = Box(low=-1, high=1, shape=(obs_dim,))
    act_space = Box(low=-1, high=1, shape=(act_dim,))
    policy = ActorVCritic(obs_dim, act_dim, hidden_sizes=[16, 16]).to(device)
    buffer = VectorizedOnPolicyBuffer(
        obs_space=obs_space,
        act_space=act_space,
        size=4,
        device=device,
        num_envs=1,
    )
    bundle = AgentPPOBundle(
        policy=policy,
        buffer=buffer,
        actor_optimizer=torch.optim.Adam(policy.actor.parameters(), lr=1e-3),
        reward_critic_optimizer=torch.optim.Adam(
            policy.reward_critic.parameters(), lr=1e-3
        ),
        cost_critic_optimizer=torch.optim.Adam(
            policy.cost_critic.parameters(), lr=1e-3
        ),
    )
    for step in range(4):
        obs = torch.randn(1, obs_dim, device=device)
        act = torch.randn(1, act_dim, device=device)
        with torch.no_grad():
            _, log_prob, value_r, value_c = policy.step(obs)
        buffer.store(
            obs=obs,
            act=act,
            reward=torch.tensor([1.0], device=device),
            cost=torch.zeros(1, device=device),
            value_r=value_r,
            value_c=value_c,
            log_prob=log_prob,
        )
    buffer.finish_path(
        last_value_r=torch.zeros(1, device=device),
        last_value_c=torch.zeros(1, device=device),
        idx=0,
    )

    before = [p.detach().clone() for p in policy.actor.parameters()]
    logger = EpochLogger(log_dir=os.devnull, seed="0")
    ppo_cfg = {
        "num_mini_batch": 1,
        "learning_iters": 1,
        "clip_ratio": 0.2,
        "ent_coef": 0.0,
        "target_kl": 1.0,
        "max_grad_norm": 40.0,
        "use_critic_norm": False,
        "use_value_coefficient": False,
    }
    iters, kl = _ppo_update_agent(
        bundle, ppo_cfg, logger, use_lagrange=False
    )
    after = [p.detach() for p in policy.actor.parameters()]
    delta = float(torch.sqrt(sum((a - b).pow(2).sum() for a, b in zip(after, before))).item())

    assert iters >= 1
    assert delta > 0.0
    assert logger.epoch_dict.get("Train/ActorParamDelta", [0.0])[-1] > 0.0
