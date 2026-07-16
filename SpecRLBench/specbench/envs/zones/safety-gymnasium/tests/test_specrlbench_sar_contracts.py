"""SpecRLBench SAR architecture contracts.

These tests pin public behavior before conservative refactors. They are not
intended to prove policy quality; they guard registration, wrapper, reset, and
SB3 integration surfaces that later cleanup must preserve.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from gymnasium import spaces


ROOT = Path(__file__).resolve().parents[5]
SAFETY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SAFETY_ROOT))

pytest.importorskip('mujoco')

import safety_gymnasium  # noqa: E402
from safety_gymnasium.utils.registration import safe_registry  # noqa: E402
from safety_gymnasium.utils.task_utils import get_task_class_name  # noqa: E402
from safety_gymnasium.tasks.safe_multi_agent.utils.sar_utils import (  # noqa: E402
    agent_inside_building_idx,
    building_geom,
)

from utils.env_utils import make_env, make_vec  # noqa: E402


SAR_ENV_IDS = {
    'PointLTL0MASAR1-v0': 'MultiGoalSARLevel0',
    'PointLTL4MASAR1-v0': 'MultiGoalSARLevel4',
    'PointLTL5MASAR1-v0': 'MultiGoalSARLevel5',
    'PointLTL0MASAR2-v0': 'MultiGoalSARLevel0',
    'PointLTL1MASAR2-v0': 'MultiGoalSARLevel1',
    'PointLTL2MASAR2-v0': 'MultiGoalSARLevel2',
    'PointLTL3MASAR2-v0': 'MultiGoalSARLevel3',
    'PointLTL3MASAR5-v0': 'MultiGoalSARLevel3',
}


def _layout_snapshot(task) -> tuple[tuple[str, tuple[float, ...]], ...]:
    layout = task.world_info.layout
    return tuple(
        (key, tuple(np.asarray(value, dtype=float).round(8).reshape(-1)))
        for key, value in sorted(layout.items())
    )


def test_sar_env_ids_are_registered_and_resolve_to_expected_tasks():
    """Current public SAR env IDs and class-name mapping are compatibility API."""
    for env_id, expected_class in SAR_ENV_IDS.items():
        assert env_id in safe_registry
        assert get_task_class_name(env_id) == expected_class

        debug_id = env_id.replace('-v0', 'Debug-v0')
        vision_id = env_id.replace('-v0', 'Vision-v0')
        assert debug_id in safe_registry
        assert vision_id in safe_registry
        assert get_task_class_name(debug_id) == expected_class
        assert get_task_class_name(vision_id) == expected_class


def test_sar_non_sb3_wrapper_preserves_multi_agent_api():
    """Non-SB3 SAR path returns per-agent dict observations/actions/rewards."""
    env = make_env('PointLTL0MASAR2-v0', sb3=False)
    try:
        obs, info = env.reset(seed=0)

        assert set(obs) == {'agent_0', 'agent_1'}
        assert info['propositions'] == []
        assert isinstance(env.observation_space, spaces.Dict)

        action = {
            agent: env.action_space(agent).sample()
            for agent in env.unwrapped.possible_agents
        }
        next_obs, reward, terminated, truncated, step_info = env.step(action)

        assert set(next_obs) == {'agent_0', 'agent_1'}
        assert set(reward) == {'agent_0', 'agent_1'}
        assert set(terminated) == {'agent_0', 'agent_1'}
        assert set(truncated) == {'agent_0', 'agent_1'}
        assert isinstance(step_info['propositions'], list)
        for i, agent in enumerate(env.unwrapped.possible_agents):
            assert f'wall_sensor_{i}' in next_obs[agent]
    finally:
        env.close()


def test_sar_sb3_wrapper_flattens_obs_and_actions_for_multiinput_policy():
    """SB3 SAR path exposes one flat Dict obs and one Box action space."""
    env = make_env('PointLTL0MASAR2-v0', sb3=True)
    try:
        obs, info = env.reset(seed=0)

        assert isinstance(env.action_space, spaces.Box)
        assert env.action_space.shape == (4,)
        assert isinstance(env.observation_space, spaces.Dict)
        assert set(obs) == set(env.observation_space.spaces)
        assert all(not isinstance(value, dict) for value in obs.values())
        assert info['propositions'] == []

        next_obs, reward, terminated, truncated, step_info = env.step(env.action_space.sample())

        assert isinstance(next_obs, dict)
        assert set(next_obs) == set(env.observation_space.spaces)
        assert isinstance(reward, float)
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert isinstance(step_info['propositions'], list)
        assert isinstance(step_info['casualty_visible'], bool)
    finally:
        env.close()


@pytest.mark.parametrize('env_id', SAR_ENV_IDS)
def test_all_sar_levels_keep_sb3_reset_and_step_contract(env_id):
    """Every public SAR level must support SB3 reset and one sampled step."""
    env = make_env(env_id, sb3=True)
    try:
        obs, info = env.reset(seed=0)

        assert isinstance(env.action_space, spaces.Box)
        assert env.action_space.shape == (env.unwrapped.num_agents * env.action_dim,)
        assert isinstance(env.observation_space, spaces.Dict)
        assert set(obs) == set(env.observation_space.spaces)
        assert info['propositions'] == []

        next_obs, reward, terminated, truncated, step_info = env.step(env.action_space.sample())

        assert set(next_obs) == set(env.observation_space.spaces)
        assert isinstance(reward, float)
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert isinstance(step_info['propositions'], list)
    finally:
        env.close()


def test_sar_reset_seed_reproduces_layout_on_same_env():
    """Same explicit reset seed must reproduce authoritative task layout."""
    env = make_env('PointLTL0MASAR1-v0', sb3=True)
    try:
        env.reset(seed=7)
        first = _layout_snapshot(env.unwrapped.task)
        env.reset(seed=7)
        second = _layout_snapshot(env.unwrapped.task)
        env.reset(seed=8)
        third = _layout_snapshot(env.unwrapped.task)

        assert first == second
        assert first != third
    finally:
        env.close()


def _building_layout_snapshot(task) -> tuple[tuple[str, tuple[float, ...]], ...]:
    """Layout keys produced by building sync (buildings, entrapped, perimeter walls)."""
    layout = task.world_info.layout
    prefixes = ('terracotta_building', 'entrapped_casualty', 'building')
    return tuple(
        (key, tuple(np.asarray(value, dtype=float).round(8).reshape(-1)))
        for key, value in sorted(layout.items())
        if key.startswith(prefixes)
    )


def test_building_entrapped_layout_pinned():
    """Entrapped casualties must spawn at building centers (runtime positions)."""
    env = make_env('PointLTL5MASAR1-v0', sb3=True)
    try:
        env.reset(seed=11)
        task = env.unwrapped.task
        assert hasattr(task, 'entrapped_casualtys')
        assert hasattr(task, 'terracotta_buildings')
        entrapped_num = task.entrapped_casualtys.num
        assert entrapped_num > 0

        for i in range(entrapped_num):
            building_xy = np.asarray(task.terracotta_buildings.pos[i][:2], dtype=float)
            entrapped_xy = np.asarray(task.entrapped_casualtys.pos[i][:2], dtype=float)
            np.testing.assert_allclose(building_xy, entrapped_xy, rtol=0, atol=1e-5)
    finally:
        env.close()


def test_building_layout_seed_reproducible():
    """Building layout sync must reproduce on repeated fast-path resets."""
    env = make_env('PointLTL5MASAR1-v0', sb3=True)
    try:
        # First reset builds MuJoCo and draws wall sizes; later resets use fast layout resample.
        env.reset(seed=0)
        env.reset(seed=7)
        first = _building_layout_snapshot(env.unwrapped.task)
        env.reset(seed=7)
        second = _building_layout_snapshot(env.unwrapped.task)
        env.reset(seed=8)
        third = _building_layout_snapshot(env.unwrapped.task)

        assert first == second
        assert first != third
    finally:
        env.close()


def test_building_perimeter_wall_keys_exist():
    """Each building must expose four perimeter wall segment layout keys."""
    env = make_env('PointLTL1MASAR2-v0', sb3=True)
    try:
        env.reset(seed=3)
        layout = env.unwrapped.task.world_info.layout
        agent_num = env.unwrapped.task.agent_num
        for i in range(agent_num):
            for seg_idx in range(4):
                assert f'building{i}_ltl_wall{seg_idx}' in layout
    finally:
        env.close()


def test_entered_building_suppresses_shell_lidar_and_render():
    """Entered building shell is hidden and omitted from building lidar + LoS rays."""
    import mujoco

    env = make_env('PointLTL5MASAR1-v0', sb3=True)
    try:
        env.reset(seed=11)
        task = env.unwrapped.task
        buildings = building_geom(task)
        assert buildings is not None

        center = buildings.pos[0]
        suffix = '_0'
        adr_x = task.model.jnt_qposadr[
            mujoco.mj_name2id(task.model, mujoco.mjtObj.mjOBJ_JOINT, f'x{suffix}')
        ]
        adr_y = task.model.jnt_qposadr[
            mujoco.mj_name2id(task.model, mujoco.mjtObj.mjOBJ_JOINT, f'y{suffix}')
        ]
        agent_xy = task.agent.get_agent_pos(0)[:2]
        task.data.qpos[adr_x] += float(center[0] - agent_xy[0])
        task.data.qpos[adr_y] += float(center[1] - agent_xy[1])
        mujoco.mj_forward(task.model, task.data)

        assert agent_inside_building_idx(task, 0) == 0

        task._sync_entered_building_state()
        shell_geom_id = task._obstacle_geom_id_for_instance(buildings, 0)
        assert shell_geom_id in task._lidar_suppressed_geom_ids
        assert task.model.geom_rgba[shell_geom_id][-1] == 0.0

        obs = task.obs()
        building_lidar = obs['terracotta_buildings_lidar_0']
        assert float(np.max(building_lidar)) == 0.0
    finally:
        env.close()


def test_sar_vecnormalize_stack_preserves_sb3_contract():
    """Training helper must keep SAR compatible with SB3 VecNormalize."""
    pytest.importorskip('stable_baselines3')

    vec_env = make_vec(
        'PointLTL0MASAR1-v0',
        n_envs=1,
        sb3=True,
        normalize=True,
        parallel=False,
    )
    try:
        assert vec_env.norm_reward is False
        obs = vec_env.reset()

        assert isinstance(obs, dict)
        assert all(value.shape[0] == 1 for value in obs.values())

        obs, reward, done, info = vec_env.step([vec_env.action_space.sample()])

        assert isinstance(obs, dict)
        assert reward.shape == (1,)
        assert done.shape == (1,)
        assert isinstance(info[0]['propositions'], list)
    finally:
        vec_env.close()
