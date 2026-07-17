"""Unit tests for OpenAI-faithful PPORND / RND components."""

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
from rnd.rnd_ppo import PPORND, RNDPPO
from rnd.stats import RewardForwardFilter, RNDRunningStats
from rnd.storage import InteractionStorage


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


def test_config_openai_aliases():
    cfg = RNDConfig(intrinsic_reward_coef=0.5).resolve()
    assert cfg.int_coeff == 0.5
    cfg2 = RNDConfig(feature_dim=256).resolve()
    assert cfg2.rnd_rep_size == 256


def test_target_frozen_after_aux():
    model = RNDModel(input_dim=8, rep_size=16, enlargement=1)
    target_before = {k: v.clone() for k, v in model.target.state_dict().items()}
    x = th.randn(32, 8)
    opt = th.optim.Adam(model.predictor.parameters(), lr=1e-3)
    for _ in range(5):
        loss, _, _ = model.aux_loss(x)
        opt.zero_grad()
        loss.backward()
        opt.step()
    for k, v in model.target.state_dict().items():
        assert th.allclose(v, target_before[k]), f"target param {k} changed"


def test_openai_intrinsic_reward_is_mean_mse():
    model = RNDModel(input_dim=8, rep_size=16, enlargement=1)
    x = th.randn(4, 8)
    with th.no_grad():
        pred, tgt = model.forward(x)
        expected = th.mean(th.square(tgt - pred), dim=-1)
        got = model.prediction_error(x)
    assert th.allclose(expected, got)


def test_reward_forward_filter():
    rff = RewardForwardFilter(0.99)
    r0 = np.ones(2, dtype=np.float64)
    out0 = rff.update(r0)
    assert np.allclose(out0, r0)
    r1 = np.ones(2, dtype=np.float64) * 2.0
    out1 = rff.update(r1)
    assert np.allclose(out1, 0.99 * r0 + r1)


def test_intrinsic_norm_openai_no_done_reset():
    stats = RNDRunningStats(obs_shape=(4,), n_envs=2, return_norm=True, gamma=0.99)
    rews = np.ones((2, 5), dtype=np.float32)
    normed = stats.normalize_intrinsic_rewards(rews)
    assert normed.shape == (2, 5)
    assert stats.rff_int.rewems is not None
    assert float(stats.rff_int.rewems.mean()) > 1.0


def test_interaction_storage_gae_shapes():
    store = InteractionStorage(
        n_envs=4, n_steps=8, obs_shape=(3,), action_dim=1, discrete=True
    )
    for t in range(8):
        store.add_step(
            t=t,
            obs=np.zeros((4, 3), dtype=np.float32),
            actions=np.zeros(4, dtype=np.int64),
            neglogp=np.zeros(4, dtype=np.float32),
            entropy=np.zeros(4, dtype=np.float32),
            vpred_int=np.zeros(4, dtype=np.float32),
            vpred_ext=np.zeros(4, dtype=np.float32),
            news=np.zeros(4, dtype=np.float32),
            rews_ext_prev=np.ones(4, dtype=np.float32) if t > 0 else None,
        )
    store.set_bootstrap(
        ob_last=np.zeros((4, 3), dtype=np.float32),
        new_last=np.zeros(4, dtype=np.float32),
        vpred_int_last=np.zeros(4, dtype=np.float32),
        vpred_ext_last=np.zeros(4, dtype=np.float32),
        rews_ext_last=np.ones(4, dtype=np.float32),
    )
    store.set_intrinsic_rewards(np.ones((4, 8), dtype=np.float32) * 0.1)
    store.compute_gae(
        rews_int_norm=np.ones((4, 8), dtype=np.float32) * 0.1,
        gamma=0.99,
        gamma_ext=0.99,
        lam=0.95,
        int_coeff=1.0,
        ext_coeff=2.0,
        use_news=False,
    )
    assert store.buf_advs.shape == (4, 8)
    batches = list(store.env_minibatches(nminibatches=2))
    assert len(batches) == 2
    assert batches[0]["obs"].shape[0] == 2


def test_aux_loss_mask_proportion():
    model = RNDModel(input_dim=8, rep_size=16, enlargement=1)
    x = th.randn(64, 8)
    loss_full, _, _ = model.aux_loss(x, proportion=1.0)
    loss_half, _, _ = model.aux_loss(x, proportion=0.5)
    assert loss_full.ndim == 0
    assert loss_half.ndim == 0


def test_ppornd_trains_cartpole():
    env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
    cfg = RNDConfig(
        int_coeff=1.0,
        ext_coeff=2.0,
        rnd_rep_size=32,
        update_ob_stats_from_random_agent=False,
        random_obs_init_steps=0,
        policy_size="small",
    )
    model = PPORND(
        "MlpPolicy",
        env,
        n_steps=64,
        batch_size=128,
        n_epochs=2,
        nminibatches=1,
        rnd_config=cfg,
        verbose=0,
        learning_rate=1e-3,
        ent_coef=0.001,
        clip_range=0.1,
    )
    model.learn(total_timesteps=128)
    assert model._n_updates > 0
    env.close()


def test_rndppo_alias_trains():
    env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
    model = RNDPPO(
        "MlpPolicy",
        env,
        n_steps=32,
        n_epochs=1,
        nminibatches=1,
        rnd_config=RNDConfig(
            update_ob_stats_from_random_agent=False,
            random_obs_init_steps=0,
            policy_size="small",
        ),
        verbose=0,
    )
    model.learn(total_timesteps=64)
    env.close()


def test_ppornd_save_load_roundtrip():
    env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
    cfg = RNDConfig(
        int_coeff=0.5,
        rnd_rep_size=16,
        update_ob_stats_from_random_agent=False,
        random_obs_init_steps=0,
        policy_size="small",
    )
    model = PPORND(
        "MlpPolicy",
        env,
        n_steps=32,
        n_epochs=1,
        nminibatches=1,
        rnd_config=cfg,
    )
    model.learn(total_timesteps=64)
    assert model.rnd is not None and model.policy is not None
    target_sd = {k: v.clone() for k, v in model.rnd.model.target.state_dict().items()}
    pred_sd = {k: v.clone() for k, v in model.rnd.model.predictor.state_dict().items()}
    pol_sd = {k: v.clone() for k, v in model.policy.state_dict().items()}
    obs_mean = model.rnd.stats.ob_rms.mean.copy()

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "rnd_model")
        model.save(path)
        assert Path(path + ".zip").is_file(), "SB3 zip must exist after save"
        loaded = PPORND.load(path, env=env)
        loaded_zip = PPORND.load(path + ".zip", env=env)

    assert loaded.rnd is not None and loaded.policy is not None
    for k, v in loaded.rnd.model.target.state_dict().items():
        assert th.allclose(v, target_sd[k])
    for k, v in loaded.rnd.model.predictor.state_dict().items():
        assert th.allclose(v, pred_sd[k])
    for k, v in loaded.policy.state_dict().items():
        assert th.allclose(v, pol_sd[k])
    assert np.allclose(loaded.rnd.stats.ob_rms.mean, obs_mean)
    # path.zip suffix also works
    assert loaded_zip.rnd is not None
    for k, v in loaded_zip.policy.state_dict().items():
        assert th.allclose(v, pol_sd[k])
    env.close()


def test_eval_does_not_update_rnd_stats():
    env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
    model = PPORND(
        "MlpPolicy",
        env,
        n_steps=32,
        n_epochs=1,
        nminibatches=1,
        rnd_config=RNDConfig(
            update_ob_stats_from_random_agent=False,
            random_obs_init_steps=0,
            policy_size="small",
        ),
    )
    model.learn(total_timesteps=64)
    assert model.rnd is not None
    count_before = model.rnd.stats.ob_rms.count
    mean_before = model.rnd.stats.ob_rms.mean.copy()
    obs = env.reset()
    for _ in range(10):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = env.step(action)
    assert model.rnd.stats.ob_rms.count == count_before
    assert np.allclose(model.rnd.stats.ob_rms.mean, mean_before)
    env.close()


def test_dual_value_heads_exist():
    env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
    model = PPORND(
        "MlpPolicy",
        env,
        n_steps=16,
        nminibatches=1,
        rnd_config=RNDConfig(
            update_ob_stats_from_random_agent=False,
            random_obs_init_steps=0,
            policy_size="small",
        ),
    )
    assert model.policy is not None
    assert hasattr(model.policy, "vf_int") and hasattr(model.policy, "vf_ext")
    env.close()


def test_module_compute_shapes():
    space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
    mod = RNDModule(space, n_envs=3, config=RNDConfig(rnd_rep_size=8, policy_size="small"))
    obs = np.random.randn(3, 5).astype(np.float32)
    dones = np.zeros(3, dtype=np.float32)
    rnd_obs, raw, norm = mod.compute_intrinsic_reward(obs, dones, training=True)
    assert raw.shape == (3,)
    assert norm.shape == (3,)
