"""Unit tests for RND PPO components (CartPole / synthetic)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from rnd.config import RNDConfig
from rnd.module import RNDModule
from rnd.networks import RNDModel
from rnd.obs_adapter import RNDObsAdapter, resolve_rnd_obs_keys
from rnd.rnd_ppo import RND
from rnd.stats import RNDRunningStats
from rnd.storage import RNDStorage


def test_obs_adapter_box():
    space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
    adapter = RNDObsAdapter(space)
    obs = np.zeros((8, 4), dtype=np.float32)
    out = adapter.to_numpy(obs)
    assert out.shape == (8, 4)


def test_obs_adapter_dict():
    space = spaces.Dict(
        {
            "a": spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
            "b": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
        }
    )
    adapter = RNDObsAdapter(space)
    obs = {"a": np.ones((4, 3), dtype=np.float32), "b": np.zeros((4, 2), dtype=np.float32)}
    out = adapter.to_numpy(obs)
    assert out.shape == (4, 5)


def test_obs_adapter_multi_keys():
    space = spaces.Dict(
        {
            "walls_lidar_0": spaces.Box(low=0, high=1, shape=(16,), dtype=np.float32),
            "buildings_lidar_0": spaces.Box(low=0, high=1, shape=(16,), dtype=np.float32),
            "surface_casualtys_lidar_0": spaces.Box(low=0, high=1, shape=(16,), dtype=np.float32),
            "wall_sensor_0": spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32),
        }
    )
    adapter = RNDObsAdapter(
        space,
        obs_keys=["buildings_lidar_0", "walls_lidar_0", "wall_sensor_0"],
    )
    assert adapter.input_dim == 16 + 16 + 4
    obs = {
        "walls_lidar_0": np.ones((2, 16), dtype=np.float32),
        "buildings_lidar_0": np.zeros((2, 16), dtype=np.float32),
        "surface_casualtys_lidar_0": np.ones((2, 16), dtype=np.float32),
        "wall_sensor_0": np.zeros((2, 4), dtype=np.float32),
    }
    out = adapter.to_numpy(obs)
    assert out.shape == (2, 36)


def test_resolve_rnd_obs_keys_l4_l5_profiles():
    space = spaces.Dict(
        {
            "walls_lidar_0": spaces.Box(low=0, high=1, shape=(16,), dtype=np.float32),
            "buildings_lidar_0": spaces.Box(low=0, high=1, shape=(16,), dtype=np.float32),
            "building0_ltl_walls_lidar_0": spaces.Box(low=0, high=1, shape=(16,), dtype=np.float32),
            "entrapped_casualtys_lidar_0": spaces.Box(low=0, high=1, shape=(16,), dtype=np.float32),
            "wall_sensor_0": spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32),
            "accelerometer": spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
        }
    )
    l4 = resolve_rnd_obs_keys(space, include_substrings=["walls", "wall_sensor"])
    assert "walls_lidar_0" in l4
    assert "wall_sensor_0" in l4
    assert "buildings_lidar_0" not in l4
    assert "entrapped_casualtys_lidar_0" not in l4

    l5 = resolve_rnd_obs_keys(
        space,
        include_substrings=["buildings", "walls", "ltl_walls", "wall_sensor"],
    )
    assert "buildings_lidar_0" in l5
    assert "building0_ltl_walls_lidar_0" in l5
    assert "walls_lidar_0" in l5
    assert "wall_sensor_0" in l5
    assert "entrapped_casualtys_lidar_0" not in l5
    assert "accelerometer" not in l5


def test_sparse_config_defaults():
    cfg = RNDConfig()
    assert cfg.intrinsic_reward_coef == 0.5
    assert cfg.feature_dim == 256
    assert cfg.obs_keys is None


def test_target_frozen_after_train():
    model = RNDModel(input_dim=8, feature_dim=16)
    target_before = {k: v.clone() for k, v in model.target.state_dict().items()}
    x = th.randn(32, 8)
    opt = th.optim.Adam(model.predictor.parameters(), lr=1e-3)
    for _ in range(5):
        loss = model.predictor_loss(x)
        opt.zero_grad()
        loss.backward()
        opt.step()
    for k, v in model.target.state_dict().items():
        assert th.allclose(v, target_before[k]), f"target param {k} changed"


def test_predictor_loss_decreases_on_fixed_batch():
    model = RNDModel(input_dim=8, feature_dim=16)
    x = th.randn(64, 8)
    opt = th.optim.Adam(model.predictor.parameters(), lr=1e-3)
    losses = []
    for _ in range(20):
        loss = model.predictor_loss(x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_intrinsic_return_norm_done_reset():
    stats = RNDRunningStats(obs_shape=(4,), n_envs=2, return_norm=True)
    r = np.array([1.0, 1.0], dtype=np.float64)
    dones = np.array([0.0, 1.0], dtype=np.float64)
    stats.normalize_intrinsic_reward(r, dones, update=True)
    assert stats.int_returns[1] == 1.0  # reset then add: gamma*0*(1-1)+1
    # next step: env1 continues, env2 restarted previous done already applied
    r2 = np.array([1.0, 0.5], dtype=np.float64)
    dones2 = np.array([0.0, 0.0], dtype=np.float64)
    stats.normalize_intrinsic_reward(r2, dones2, update=True)
    assert stats.int_returns[0] > 1.0


def test_reward_norm_scale():
    stats = RNDRunningStats(obs_shape=(2,), n_envs=4, return_norm=True, reward_clip=100.0)
    # Warm up variance
    for _ in range(50):
        r = np.ones(4, dtype=np.float64) * 10.0
        dones = np.zeros(4, dtype=np.float64)
        stats.normalize_intrinsic_reward(r, dones, update=True)
    r = np.ones(4, dtype=np.float64) * 10.0
    normed = stats.normalize_intrinsic_reward(r, np.zeros(4), update=False)
    # Should be order-1 after return-std normalization
    assert np.mean(np.abs(normed)) < 5.0


def test_rnd_storage_shapes():
    store = RNDStorage(n_steps=5, n_envs=2, input_dim=3)
    for i in range(5):
        store.add(
            rnd_obs=np.ones((2, 3), dtype=np.float32) * i,
            raw_intrinsic=np.ones(2, dtype=np.float32),
            norm_intrinsic=np.ones(2, dtype=np.float32) * 0.5,
            extrinsic=np.zeros(2, dtype=np.float32),
            combined=np.ones(2, dtype=np.float32) * 0.5,
        )
    assert store.full
    batch = store.get_rnd_obs_batch(np.array([0, 1, 2]))
    assert batch.shape == (3, 3)


def test_rndppo_beta_zero_trains():
    env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
    model = RND(
        "MlpPolicy",
        env,
        n_steps=64,
        batch_size=32,
        n_epochs=2,
        use_rnd=True,
        intrinsic_reward_coef=0.0,
        verbose=0,
    )
    model.learn(total_timesteps=128)
    env.close()


def test_rndppo_with_rnd_trains_and_logs():
    env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
    cfg = RNDConfig(intrinsic_reward_coef=0.1, feature_dim=32, target_net_arch=[64], predictor_net_arch=[64, 64])
    model = RND(
        "MlpPolicy",
        env,
        n_steps=64,
        batch_size=32,
        n_epochs=2,
        rnd_config=cfg,
        verbose=0,
    )
    model.learn(total_timesteps=128)
    assert model.rnd is not None
    assert model.rnd._n_updates > 0
    env.close()


def test_rndppo_save_load_roundtrip():
    env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
    cfg = RNDConfig(intrinsic_reward_coef=0.05, feature_dim=16)
    model = RND("MlpPolicy", env, n_steps=32, batch_size=16, n_epochs=1, rnd_config=cfg)
    model.learn(total_timesteps=64)

    # Snapshot target + predictor + obs rms
    assert model.rnd is not None
    target_sd = {k: v.clone() for k, v in model.rnd.model.target.state_dict().items()}
    pred_sd = {k: v.clone() for k, v in model.rnd.model.predictor.state_dict().items()}
    obs_mean = model.rnd.stats.obs_rms.mean.copy()
    n_upd = model.rnd._n_updates

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "rnd_model")
        model.save(path)
        loaded = RND.load(path, env=env)

    assert loaded.rnd is not None
    for k, v in loaded.rnd.model.target.state_dict().items():
        assert th.allclose(v, target_sd[k])
    for k, v in loaded.rnd.model.predictor.state_dict().items():
        assert th.allclose(v, pred_sd[k])
    assert np.allclose(loaded.rnd.stats.obs_rms.mean, obs_mean)
    assert loaded.rnd._n_updates == n_upd
    env.close()


def test_eval_does_not_update_rnd_stats():
    """predict() path must not touch RND stats (no collect_rollouts)."""
    env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
    model = RND(
        "MlpPolicy",
        env,
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        rnd_config=RNDConfig(intrinsic_reward_coef=0.1),
    )
    model.learn(total_timesteps=64)
    assert model.rnd is not None
    count_before = model.rnd.stats.obs_rms.count
    mean_before = model.rnd.stats.obs_rms.mean.copy()
    obs = env.reset()
    for _ in range(10):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = env.step(action)
    assert model.rnd.stats.obs_rms.count == count_before
    assert np.allclose(model.rnd.stats.obs_rms.mean, mean_before)
    env.close()


def test_module_compute_shapes():
    space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
    mod = RNDModule(space, n_envs=3, config=RNDConfig(feature_dim=8))
    obs = np.random.randn(3, 5).astype(np.float32)
    dones = np.zeros(3, dtype=np.float32)
    rnd_obs, raw, norm = mod.compute_intrinsic_reward(obs, dones, training=True)
    assert rnd_obs.shape == (3, 5)
    assert raw.shape == (3,)
    assert norm.shape == (3,)
    combined = mod.combine_rewards(np.ones(3), norm)
    assert combined.shape == (3,)
