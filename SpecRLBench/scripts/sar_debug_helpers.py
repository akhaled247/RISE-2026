"""Helpers for SAR reset / reward / spawn debugging."""
from __future__ import annotations

import numpy as np


def get_task_from_vec(vec_env):
    base = vec_env.venv.envs[0] if hasattr(vec_env, 'venv') else vec_env.envs[0]
    while hasattr(base, 'env'):
        base = base.env
    return base.unwrapped.task


def get_wrapped_env_from_vec(vec_env):
    base = vec_env.venv.envs[0] if hasattr(vec_env, 'venv') else vec_env.envs[0]
    while hasattr(base, 'env'):
        base = base.env
    return base


def rng_state_digest(task) -> int:
    rg = task.random_generator.random_generator
    if rg is None:
        return -1
    return int(rg.get_state()[1][0])


def inside_building_cost(task, agent_idx: int = 0) -> float:
    if not hasattr(task, 'terracotta_buildings'):
        return 0.0
    cost = task.terracotta_buildings.cal_cost()
    return float(cost[f'agent_{agent_idx}'].get('cost_buildings_terracotta', 0))


def touch_threshold(task) -> float:
    if hasattr(task, 'entrapped_casualtys'):
        return float(task.entrapped_casualtys.size + 0.15)
    if hasattr(task, 'surface_casualtys'):
        return float(task.surface_casualtys.size + 0.15)
    return 0.0


def snapshot_positions(task) -> dict:
    building_xy = entrapped_xy = agent_xy = None
    if hasattr(task, 'terracotta_buildings'):
        building_xy = task.terracotta_buildings.pos[0][:2].copy()
    if hasattr(task, 'entrapped_casualtys'):
        entrapped_xy = task.entrapped_casualtys.pos[0][:2].copy()
    agent_xy = np.asarray(task.agent.pos[0][:2], dtype=float)

    dist_casualty = task._dist_to_casualty(0) if hasattr(task, '_dist_to_casualty') else None
    dist_building = (
        float(np.linalg.norm(agent_xy - building_xy))
        if building_xy is not None else None
    )

    cached = getattr(task, '_cached_building_locations', None)
    cached0 = tuple(np.round(cached[0], 3)) if cached else None

    entrapped_loc = None
    if hasattr(task, 'entrapped_casualtys'):
        locs = getattr(task.entrapped_casualtys, 'locations', None)
        entrapped_loc = tuple(np.round(locs[0], 3)) if locs else None

    building_loc = None
    if hasattr(task, 'terracotta_buildings'):
        locs = getattr(task.terracotta_buildings, 'locations', None)
        building_loc = tuple(np.round(locs[0], 3)) if locs else None

    return {
        'building_xy': None if building_xy is None else tuple(np.round(building_xy, 3)),
        'entrapped_xy': None if entrapped_xy is None else tuple(np.round(entrapped_xy, 3)),
        'agent_xy': tuple(np.round(agent_xy, 3)),
        'dist_agent_casualty': dist_casualty,
        'dist_agent_building': dist_building,
        'inside_building_cost': inside_building_cost(task),
        'touch_threshold': touch_threshold(task),
        'building_inside_radius': (
            float(task.terracotta_buildings.size)
            if hasattr(task, 'terracotta_buildings') else None
        ),
        'cached_building_0': cached0,
        'entrapped_locations_0': entrapped_loc,
        'building_locations_0': building_loc,
        'rng_digest': rng_state_digest(task),
    }


def format_snapshot(ep: int, snap: dict, prefix: str = '') -> str:
    return (
        f"{prefix}ep={ep} building={snap['building_xy']} entrapped={snap['entrapped_xy']} "
        f"cached={snap['cached_building_0']} ent_loc={snap['entrapped_locations_0']} "
        f"bld_loc={snap['building_locations_0']} rng={snap['rng_digest']}"
    )


def reward_attribution(task, vec_reward: float, info: dict) -> dict:
    task_rewards = task.calculate_reward()
    task_total = float(task_rewards.get('agent_0', 0.0))
    wrapper_bonus = float(vec_reward) - task_total

    entrapped_lidar_max = None
    if hasattr(task, 'original_obs') and task.original_obs:
        obs0 = task.original_obs.get('agent_0', {})
        key = 'entrapped_casualtys_lidar_0'
        if key in obs0:
            entrapped_lidar_max = float(np.max(obs0[key]))

    props = info.get('propositions', []) if isinstance(info, dict) else []
    return {
        'vec_reward': float(vec_reward),
        'task_reward': task_total,
        'wrapper_bonus': wrapper_bonus,
        'entrapped_lidar_max': entrapped_lidar_max,
        'propositions': list(props),
        'inside_building_cost': inside_building_cost(task),
        'dist_agent_casualty': task._dist_to_casualty(0),
        'touch_threshold': touch_threshold(task),
    }


def reset_vec_with_layout_seed(vec_env, seed: int):
    """Reset SB3 vec stack so Builder.set_seed(seed) runs via wrapped reset(seed=)."""
    env = get_wrapped_env_from_vec(vec_env)
    obs, info = env.reset(seed=seed)
    if hasattr(vec_env, 'normalize_obs'):
        obs = vec_env.normalize_obs(obs)
    return np.array([obs]), info


def diagnose_verdict(
    path_a_buildings: list,
    path_b_buildings: list,
    path_a_entrapped: list,
    path_b_entrapped: list,
    sync_mismatches: int,
) -> str:
    verdicts = []
    a_unique = len(set(path_a_buildings))
    b_unique = len(set(path_b_buildings))
    if a_unique > 1 and b_unique <= 1:
        verdicts.append('SEED_PLUMBING')
    if sync_mismatches > 0:
        verdicts.append('GEOM_SYNC')
    if not verdicts:
        if a_unique <= 1 and b_unique <= 1:
            verdicts.append('CHECK_RNG_OR_ENV')
        else:
            verdicts.append('OK_VARIATION')
    return ' | '.join(verdicts)
