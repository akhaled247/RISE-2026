"""Unit tests for SA→MA deploy obs helpers (no MuJoCo)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from rise_training.cmdp.obs_spec import (
    assert_deploy_obs_compatible,
    flatten_agent_obs,
    flatten_ma_agent_for_sa_deploy,
    infer_actor_obs_dim,
    remap_ma_agent_obs_to_sa_train,
)


def test_infer_actor_obs_dim_from_checkpoint():
    state_dict = {"mean.0.weight": torch.zeros(64, 81)}
    assert infer_actor_obs_dim(state_dict) == 81


def test_assert_deploy_obs_compatible_passes_on_match():
    keys = ["a", "b"]
    obs = {"a": np.zeros(3), "b": np.ones(2)}
    flat = flatten_agent_obs(obs, keys)
    assert_deploy_obs_compatible(
        flat,
        keys,
        5,
        train_env="PointLTL0MASAR1WC-v0",
        eval_env="PointLTL0MASAR2WC-v0",
    )


def test_assert_deploy_obs_compatible_raises_on_mismatch():
    keys = ["a"]
    flat = np.zeros(3, dtype=np.float32)
    with pytest.raises(ValueError, match="Deploy obs dim"):
        assert_deploy_obs_compatible(
            flat,
            keys,
            5,
            train_env="PointLTL0MASAR1WC-v0",
            eval_env="PointLTL0MASAR2WC-v0",
        )


def test_remap_ma_agent_obs_to_sa_train_agent1():
    train_keys = ["accelerometer_0", "velocimeter_0", "terracotta_buildings_visited"]
    agent1_obs = {
        "accelerometer_1": np.array([1.0, 0.0]),
        "velocimeter_1": np.array([0.5, 0.5]),
        "terracotta_buildings_visited": np.array([1.0]),
    }
    mapped = remap_ma_agent_obs_to_sa_train(agent1_obs, 1, train_keys)
    assert set(mapped.keys()) == set(train_keys)
    flat = flatten_ma_agent_for_sa_deploy(agent1_obs, 1, train_keys)
    assert flat.shape == (5,)
    np.testing.assert_allclose(flat[:2], [1.0, 0.0])
    np.testing.assert_allclose(flat[2:4], [0.5, 0.5])
    assert flat[4] == 1.0


def test_remap_ma_agent_obs_raises_when_key_missing():
    with pytest.raises(KeyError, match="accelerometer_0"):
        remap_ma_agent_obs_to_sa_train({}, 1, ["accelerometer_0"])
