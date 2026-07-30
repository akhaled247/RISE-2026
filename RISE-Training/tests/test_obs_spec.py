"""Unit tests for SA→MA deploy obs helpers (no MuJoCo)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from rise_training.cmdp.obs_spec import (
    assert_deploy_obs_compatible,
    flatten_agent_obs,
    infer_actor_obs_dim,
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
