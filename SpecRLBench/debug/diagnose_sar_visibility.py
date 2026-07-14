"""Diagnostic script for SAR visibility vs reward leakage."""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "specbench" / "envs" / "zones" / "safety-gymnasium"))
sys.path.insert(0, str(ROOT))

import safety_gymnasium  # noqa: F401
from utils.env_utils import make_env


def _get_task(env):
    task = env.unwrapped
    while hasattr(task, "env"):
        task = task.env
    return task.task


def _casualty_visible_raw(task, agent_idx: int = 0) -> bool:
    if not hasattr(task, "surface_casualtys"):
        return False
    row = task._nearest_casualty_row(agent_idx)
    return task._casualty_los_visible(agent_idx, row)


def run_diagnostics(
    env_name: str,
    episodes: int = 20,
    seed: int = 0,
    sb3: bool = True,
):
    env = make_env(env_name, render_mode=None, sb3=sb3)
    task = _get_task(env)
    rng = np.random.default_rng(seed)

    episode_returns = []
    rescue_count = 0
    visible_steps = 0
    occluded_steps = 0
    reward_when_visible = []
    reward_when_occluded = []
    abs_dist_delta_occluded = []

    print(f"Environment: {env_name}")
    print(f"Agents: {env.unwrapped.num_agents}")
    print("-" * 60)

    for ep in range(episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
        ep_return = 0.0
        rescued = False
        prev_dist = None
        if hasattr(task, "last_dist_casualty") and task.last_dist_casualty:
            prev_dist = task.last_dist_casualty[0]

        done = False
        while not done:
            visible_raw = _casualty_visible_raw(task, 0)
            visible_sticky = (
                task._casualty_visible_sticky[0]
                if getattr(task, "_casualty_visible_sticky", None)
                else visible_raw
            )

            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            step_reward = float(reward)
            ep_return += step_reward

            if visible_sticky:
                visible_steps += 1
                reward_when_visible.append(step_reward)
            else:
                occluded_steps += 1
                reward_when_occluded.append(step_reward)

            if prev_dist is not None and hasattr(task, "last_dist_casualty"):
                cur_dist = task.last_dist_casualty[0]
                if not visible_sticky:
                    abs_dist_delta_occluded.append(abs(prev_dist - cur_dist))
                prev_dist = cur_dist

            prop_keys = info.get("propositions", [])
            if any("cost_casualtys_surface" in k for k in prop_keys):
                rescued = True

            done = terminated or truncated

        if rescued:
            rescue_count += 1
        episode_returns.append(ep_return)
        print(f"Ep {ep + 1:02d}: return={ep_return:7.3f} rescued={rescued}")

    total_steps = visible_steps + occluded_steps
    env.close()

    print("-" * 60)
    print(f"Mean return:                 {np.mean(episode_returns):.4f}")
    print(f"Rescue rate:                 {rescue_count}/{episodes} "
          f"({100 * rescue_count / episodes:.1f}%)")
    print(f"Visible steps:               {visible_steps}/{total_steps} "
          f"({100 * visible_steps / max(total_steps, 1):.1f}%)")
    if reward_when_visible:
        print(f"Mean reward when visible:    {np.mean(reward_when_visible):.5f}")
    if reward_when_occluded:
        print(f"Mean reward when occluded:   {np.mean(reward_when_occluded):.5f}")
    if abs_dist_delta_occluded:
        print(f"Mean |dist delta| occluded:  {np.mean(abs_dist_delta_occluded):.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAR visibility/reward diagnostics")
    parser.add_argument("--env", default="PointLTL4MASAR1-v0")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-sb3", action="store_true")
    args = parser.parse_args()
    run_diagnostics(
        args.env,
        episodes=args.episodes,
        seed=args.seed,
        sb3=not args.no_sb3,
    )
