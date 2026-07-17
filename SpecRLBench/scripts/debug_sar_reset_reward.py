#!/usr/bin/env python3
"""Structured debug: building spawn + reward attribution for SAR / ppo_load path."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT / 'specbench' / 'envs' / 'zones' / 'safety-gymnasium'))

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from sar_debug_helpers import (
    diagnose_verdict,
    format_snapshot,
    get_task_from_vec,
    reset_vec_with_layout_seed,
    reward_attribution,
    snapshot_positions,
)
from utils.env_utils import make_env


def run_phase0(env_name: str, episodes: int = 8) -> dict:
    import safety_gymnasium.tasks.safe_multi_agent.builder as builder_mod
    seeds_seen = []
    _orig_set_seed = builder_mod.Builder.set_seed

    def _trace_set_seed(self, seed=None):
        seeds_seen.append(seed)
        return _orig_set_seed(self, seed)

    builder_mod.Builder.set_seed = _trace_set_seed

    print('=' * 60)
    print('Phase 0A: direct env.reset(seed=s)')
    env = make_env(env_name, sb3=True)
    path_a_b, path_a_e = [], []
    for s in range(episodes):
        env.reset(seed=s)
        snap = snapshot_positions(env.unwrapped.task)
        path_a_b.append(snap['building_xy'])
        path_a_e.append(snap['entrapped_xy'])
        print(format_snapshot(s, snap, 'A '))
    env.close()

    print('=' * 60)
    print('Phase 0B: DummyVecEnv seed(ep) + reset() [ppo_load style]')
    base = make_env(env_name, sb3=True)
    vec = DummyVecEnv([lambda: Monitor(base)])
    path_b_b, path_b_e = [], []
    for ep in range(episodes):
        vec.seed(ep)
        vec.reset()
        snap = snapshot_positions(get_task_from_vec(vec))
        path_b_b.append(snap['building_xy'])
        path_b_e.append(snap['entrapped_xy'])
        print(format_snapshot(ep, snap, 'B '))
    vec.close()

    print('=' * 60)
    print('Phase 0B2: reset_vec_with_layout_seed(vec, ep) [fixed seed path]')
    base2 = make_env(env_name, sb3=True)
    vec2 = DummyVecEnv([lambda: Monitor(base2)])
    path_b2_b, path_b2_e = [], []
    for ep in range(episodes):
        reset_vec_with_layout_seed(vec2, ep)
        snap = snapshot_positions(get_task_from_vec(vec2))
        path_b2_b.append(snap['building_xy'])
        path_b2_e.append(snap['entrapped_xy'])
        print(format_snapshot(ep, snap, 'B2'))
    vec2.close()

    sync_mismatches = sum(
        1 for b, e in zip(path_a_b, path_a_e) if b != e
    )
    verdict = diagnose_verdict(path_a_b, path_b_b, path_a_e, path_b_e, sync_mismatches)
    print('=' * 60)
    print(f'unique A building={len(set(path_a_b))} entrapped={len(set(path_a_e))}')
    print(f'unique B building={len(set(path_b_b))} entrapped={len(set(path_b_e))}')
    print(f'unique B2 building={len(set(path_b2_b))} entrapped={len(set(path_b2_e))}')
    print(f'A entrapped vs building mismatches: {sync_mismatches}/{episodes}')
    print(f'Builder.set_seed calls (phase 0): {seeds_seen}')
    print(f'VERDICT: {verdict}')
    builder_mod.Builder.set_seed = _orig_set_seed
    return {
        'path_a_buildings': path_a_b,
        'path_b_buildings': path_b_b,
        'verdict': verdict,
    }


def run_phase1_4(env_name: str, episodes: int = 8, steps_per_ep: int = 200):
    print('=' * 60)
    print('Phase 1-4: ppo_load-style loop + reward attribution on reward>0')
    base = make_env(env_name, sb3=True)
    vec = DummyVecEnv([lambda: Monitor(base)])
    task = get_task_from_vec(vec)
    sync_after_ep2 = 0

    for ep in range(episodes):
        obs, _ = reset_vec_with_layout_seed(vec, ep)
        snap = snapshot_positions(task)
        print(format_snapshot(ep, snap, 'RESET '))

        if ep >= 1:
            if snap['cached_building_0'] != snap['entrapped_locations_0']:
                sync_after_ep2 += 1
            if snap['building_xy'] != snap['entrapped_xy']:
                sync_after_ep2 += 1

        done = False
        step = 0
        while not done and step < steps_per_ep:
            action = vec.action_space.sample()
            obs, reward, done, info = vec.step(action)
            r0 = float(reward[0])
            if r0 > 0:
                attr = reward_attribution(task, r0, info[0])
                print(
                    f'  REWARD ep={ep} step={step} vec={attr["vec_reward"]:.3f} '
                    f'task={attr["task_reward"]:.3f} wrapper={attr["wrapper_bonus"]:.3f} '
                    f'dist_cas={attr["dist_agent_casualty"]:.3f} '
                    f'inside={attr["inside_building_cost"]} '
                    f'lidar_max={attr["entrapped_lidar_max"]} props={attr["propositions"]}'
                )
            step += 1
            done = bool(done[0])

    vec.close()
    print(f'sync mismatches after ep1 (cached/entrapped/body): {sync_after_ep2}')


def main():
    parser = argparse.ArgumentParser(description='SAR reset/reward debug harness')
    parser.add_argument('--env', default='PointLTL5MASAR1-v0')
    parser.add_argument('--episodes', type=int, default=8)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--phase0-only', action='store_true')
    args = parser.parse_args()

    run_phase0(args.env, args.episodes)
    if not args.phase0_only:
        run_phase1_4(args.env, args.episodes, args.steps)


if __name__ == '__main__':
    main()
