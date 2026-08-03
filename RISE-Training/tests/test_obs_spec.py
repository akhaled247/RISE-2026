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
    is_buildings_visited_key,
    is_gremlins_lidar_key,
    remap_ma_agent_obs_to_sa_train,
    zero_buildings_visited_channels,
    zero_gremlins_lidar_channels,
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


def test_remap_ma_agent_obs_to_sa_train_agent0():
    train_keys = ["accelerometer_0", "velocimeter_0", "terracotta_buildings_visited"]
    agent0_obs = {
        "accelerometer_0": np.array([1.0, 0.0]),
        "velocimeter_0": np.array([0.5, 0.5]),
        "terracotta_buildings_visited": np.array([1.0]),
    }
    flat0 = flatten_ma_agent_for_sa_deploy(agent0_obs, 0, train_keys)
    assert flat0.shape == (5,)


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


def test_is_gremlins_lidar_key():
    assert is_gremlins_lidar_key("gremlins_lidar_0")
    assert is_gremlins_lidar_key("gremlins_lidar_1")
    assert not is_gremlins_lidar_key("walls_lidar_0")
    assert not is_gremlins_lidar_key("terracotta_buildings_lidar_0")


def test_zero_gremlins_lidar_channels():
    mapped = {
        "gremlins_lidar_0": np.array([0.5, 0.25, 0.1], dtype=np.float32),
        "accelerometer_0": np.array([1.0, 0.0], dtype=np.float32),
    }
    out = zero_gremlins_lidar_channels(mapped)
    np.testing.assert_array_equal(out["gremlins_lidar_0"], np.zeros(3, dtype=np.float32))
    np.testing.assert_allclose(out["accelerometer_0"], [1.0, 0.0])


def test_flatten_ma_agent_zeros_gremlins_by_default():
    train_keys = ["accelerometer_0", "gremlins_lidar_0"]
    agent1_obs = {
        "accelerometer_1": np.array([1.0, 0.0], dtype=np.float32),
        "gremlins_lidar_1": np.array([0.9, 0.8, 0.7], dtype=np.float32),
    }
    flat = flatten_ma_agent_for_sa_deploy(agent1_obs, 1, train_keys)
    np.testing.assert_allclose(flat[:2], [1.0, 0.0])
    np.testing.assert_array_equal(flat[2:], np.zeros(3, dtype=np.float32))


def test_flatten_ma_agent_can_keep_gremlins():
    train_keys = ["gremlins_lidar_0"]
    agent0_obs = {"gremlins_lidar_0": np.array([0.5, 0.25], dtype=np.float32)}
    flat = flatten_ma_agent_for_sa_deploy(
        agent0_obs, 0, train_keys, zero_gremlins=False,
    )
    np.testing.assert_allclose(flat, [0.5, 0.25])


def test_is_buildings_visited_key():
    assert is_buildings_visited_key("terracotta_buildings_visited")
    assert not is_buildings_visited_key("terracotta_buildings_lidar_0")


def test_zero_buildings_visited_keeps_dim():
    train_keys = ["accelerometer_0", "terracotta_buildings_visited"]
    agent0_obs = {
        "accelerometer_0": np.array([1.0, 0.0], dtype=np.float32),
        "terracotta_buildings_visited": np.array([1.0], dtype=np.float32),
    }
    flat = flatten_ma_agent_for_sa_deploy(
        agent0_obs, 0, train_keys, zero_buildings_visited=True,
    )
    np.testing.assert_allclose(flat[:2], [1.0, 0.0])
    np.testing.assert_array_equal(flat[2:], np.zeros(1, dtype=np.float32))
    kept = zero_buildings_visited_channels(
        {"terracotta_buildings_visited": np.array([1.0], dtype=np.float32)},
    )
    np.testing.assert_array_equal(
        kept["terracotta_buildings_visited"], np.zeros(1, dtype=np.float32),
    )
