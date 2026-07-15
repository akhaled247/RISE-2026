#!/usr/bin/env python3
"""Diagnose SAR env layout: perimeter, building XY, ring-wall clearance, reset timing."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "specbench" / "envs" / "zones" / "safety-gymnasium"))

import safety_gymnasium  # noqa: F401
from utils.env_utils import make_env, make_vec


def parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    return seeds


def unwrap_task(env):
    """Unwrap single SafetyGym env to task."""
    base = env
    while hasattr(base, "env"):
        base = base.env
    return base.unwrapped.task


def unwrap_vec_task(vec_env, worker: int = 0):
    """Unwrap first (or given) SubprocVecEnv worker to task."""
    base = vec_env.venv.envs[worker]
    while hasattr(base, "env"):
        base = base.env
    return base.unwrapped.task


def perimeter_body_xy(task) -> list[np.ndarray]:
    if not hasattr(task, "ltl_walls"):
        return []
    wall = task.ltl_walls
    return [task.model.body(f"{wall.name[:-1]}{i}").xpos.copy() for i in range(wall.num)]


def expected_perimeter_corners(task) -> list[tuple[float, float]]:
    if not hasattr(task, "ltl_walls"):
        return []
    locs = task.ltl_walls.locations
    if not locs:
        lf = task.ltl_walls.locate_factor
        return [(lf, 0.0), (-lf, 0.0), (0.0, lf), (0.0, -lf)]
    return [tuple(map(float, p)) for p in locs]


def diagnose_task(task, seed: int, args, building_xys: list) -> list[str]:
    failures: list[str] = []
    mode = os.environ.get("SAR_LAYOUT_MODE", "current")

    buildings = task._building_geom()
    if buildings is None:
        failures.append(f"seed {seed}: no building geom")
        return failures

    bpos = buildings.pos[0][:2]
    building_xys.append(tuple(np.round(bpos, 3)))

    perim_xy = perimeter_body_xy(task)
    corners = expected_perimeter_corners(task)

    if hasattr(task, "ltl_walls"):
        locs = task.ltl_walls.locations
        if locs is None:
            failures.append(f"seed {seed}: ltl_walls.locations is None")
        elif len(locs) != task.ltl_walls.num:
            failures.append(
                f"seed {seed}: ltl_walls.locations len {len(locs)} != {task.ltl_walls.num}"
            )

    print(f"\n--- seed {seed} | SAR_LAYOUT_MODE={mode} ---")
    print(f"  building_xy: {tuple(np.round(bpos, 3))}")
    print(f"  perimeter bodies: {[tuple(np.round(p[:2], 3)) for p in perim_xy]}")
    print(f"  perimeter corners (geom): {corners}")

    if hasattr(task, "walls"):
        wall_positions = task.walls.pos
        print(f"  ring walls: {len(wall_positions)} segments")
        b_keepout = float(buildings.size) + getattr(task, "building_wall_clearance", 0.1)
        wall_keepout = float(getattr(task.walls, "keepout", 0.4))
        margin = task.placements_conf.margin
        min_allowed = b_keepout + wall_keepout + margin - 0.05
        for i, wpos in enumerate(wall_positions):
            d = float(np.linalg.norm(wpos[:2] - bpos))
            print(f"    wall{i} xy={tuple(np.round(wpos[:2], 3))} dist_to_building={d:.3f}")
            if args.check_wall_clearance and d < min_allowed:
                failures.append(
                    f"seed {seed} wall{i}: dist {d:.3f} < {min_allowed:.3f}"
                )

    if args.check_perimeter and perim_xy and corners:
        for j, pos in enumerate(perim_xy):
            d = min(np.linalg.norm(pos[:2] - np.array(c)) for c in corners)
            if d > 0.15:
                failures.append(
                    f"seed {seed} ltl_wall{j}: body far from corner (d={d:.3f})"
                )

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose SAR layout resets")
    parser.add_argument("--env", default="PointLTL5MASAR1Debug-v0")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument(
        "--mode",
        default=None,
        help="Set SAR_LAYOUT_MODE (current|A|B|C); default: env var or current",
    )
    parser.add_argument("--vec", action="store_true", help="Use SubprocVecEnv smoke test")
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--max-reset-s", type=float, default=5.0)
    parser.add_argument("--check-perimeter", action="store_true")
    parser.add_argument("--check-building-variance", action="store_true")
    parser.add_argument("--check-wall-clearance", action="store_true")
    parser.add_argument("--out", default=None, help="Write stdout copy to file")
    args = parser.parse_args()

    if args.mode is not None:
        os.environ["SAR_LAYOUT_MODE"] = args.mode.upper()

    mode = os.environ.get("SAR_LAYOUT_MODE", "current")
    seeds = parse_seeds(args.seeds)
    building_xys: list[tuple] = []
    all_failures: list[str] = []

    print(f"env={args.env} mode={mode} seeds={seeds} vec={args.vec}")

    if args.vec:
        vec_env = make_vec(args.env, n_envs=args.n_envs, render_mode=None, sb3=True, normalize=False)
        t0 = time.perf_counter()
        vec_env.reset()
        elapsed = time.perf_counter() - t0
        print(f"vec reset ({args.n_envs} envs): {elapsed:.2f}s")
        if elapsed > args.max_reset_s:
            all_failures.append(f"vec reset slow: {elapsed:.2f}s > {args.max_reset_s}s")
        task = unwrap_vec_task(vec_env, 0)
        all_failures.extend(diagnose_task(task, seeds[0] if seeds else 0, args, building_xys))
        vec_env.close()
    else:
        env = make_env(args.env, render_mode=None)
        for seed in seeds:
            t0 = time.perf_counter()
            env.reset(seed=seed)
            elapsed = time.perf_counter() - t0
            print(f"reset time: {elapsed:.2f}s")
            if elapsed > args.max_reset_s:
                all_failures.append(f"seed {seed}: reset slow {elapsed:.2f}s")
            task = unwrap_task(env)
            all_failures.extend(diagnose_task(task, seed, args, building_xys))
        env.close()

    if args.check_building_variance:
        unique = len(set(building_xys))
        print(f"\nunique building XY across seeds: {unique} -> {building_xys}")
        if unique < 3:
            all_failures.append(f"building variance: only {unique} unique XY (need >= 3)")

    if all_failures:
        print("\nFAIL:")
        for f in all_failures:
            print(f"  - {f}")
        return 1

    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
