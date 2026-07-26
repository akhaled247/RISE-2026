"""Tests for SpecRL MA recipe-B overrides."""

from __future__ import annotations

from backends.safepo.config import MA_SPECRL_RECIPE_B
from backends.safepo.ma_runners import _apply_specrl_ma_recipe_b


def test_apply_specrl_ma_recipe_b_masar2():
    cfg = {"episode_length": 1000, "actor_lr": 9e-5, "target_kl": 0.016}
    _apply_specrl_ma_recipe_b(cfg, "PointLTL0MASAR2-v0")
    assert cfg["episode_length"] == MA_SPECRL_RECIPE_B["episode_length"]
    assert cfg["actor_lr"] == MA_SPECRL_RECIPE_B["actor_lr"]
    assert cfg["target_kl"] == MA_SPECRL_RECIPE_B["target_kl"]


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
        "episode_length": 2500,
        "num_env_steps": 4_000_000,
        "entropy_coef": 0.02,
        "clip_param": 0.2,
        "hidden_size": 64,
        "batch_size": 256,
    }
    ppo = _cfg_train_to_ppo_config(cfg)
    assert ppo["steps_per_epoch"] == 20000
    assert ppo["local_steps_per_epoch"] == 2500
    assert ppo["ent_coef"] == 0.02
    assert ppo["clip_ratio"] == 0.2
    assert ppo["batch_size"] == 256
    assert ppo["hidden_sizes"] == [64, 64]


def test_ippo_ent_coef_alias():
    from safepo.multi_agent.ippo import _cfg_train_to_ppo_config

    ppo = _cfg_train_to_ppo_config({"ent_coef": 0.05, "n_rollout_threads": 1, "episode_length": 100})
    assert ppo["ent_coef"] == 0.05
