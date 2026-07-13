"""Diagnostic script for SAR0 reward and episode statistics."""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "specbench" / "envs" / "zones" / "safety-gymnasium"))
sys.path.insert(0, str(ROOT))

import safety_gymnasium  # noqa: F401
from utils.env_utils import make_env


def _dist_to_casualty(env) -> float | None:
    task = env.unwrapped.task
    if not hasattr(task, "surface_casualtys"):
        return None
    pos = task.surface_casualtys.pos[0]
    return task.agent.dist_xy(0, pos)


def run_diagnostics(env_name: str, episodes: int = 20, seed: int = 0, sb3: bool = True):
    env = make_env(env_name, render_mode=None, sb3=sb3)
    rng = np.random.default_rng(seed)

    episode_returns = []
    episode_lengths = []
    rescue_count = 0
    collision_steps = 0
    total_steps = 0

    print(f"Environment: {env_name}")
    print(f"Agents: {env.unwrapped.num_agents}")
    print(f"SB3 mode: {sb3}")
    print("-" * 50)

    for ep in range(episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
        ep_return = 0.0
        ep_len = 0
        rescued = False

        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            ep_len += 1
            total_steps += 1

            if sb3:
                agent_info = info.get("agent_0", {})
            else:
                agent_info = info.get("agent_0", {})

            if agent_info.get("cost_collision", 0) > 0 or agent_info.get("cost_ltl_walls", 0) > 0:
                collision_steps += 1

            prop_keys = info.get("propositions", [])
            if any("cost_casualtys_surface" in k for k in prop_keys):
                rescued = True

            done = terminated or truncated

        if rescued:
            rescue_count += 1
        episode_returns.append(ep_return)
        episode_lengths.append(ep_len)

        dist = _dist_to_casualty(env)
        dist_str = f"{dist:.3f}" if dist is not None else "n/a"
        print(
            f"Ep {ep + 1:02d}: return={ep_return:7.3f} len={ep_len:4d} "
            f"rescued={rescued} final_dist={dist_str}"
        )

    env.close()

    print("-" * 50)
    print(f"Mean return:      {np.mean(episode_returns):.4f} +/- {np.std(episode_returns):.4f}")
    print(f"Mean ep length:   {np.mean(episode_lengths):.1f}")
    print(f"Rescue rate:      {rescue_count}/{episodes} ({100 * rescue_count / episodes:.1f}%)")
    print(f"Collision steps:  {collision_steps}/{total_steps} ({100 * collision_steps / max(total_steps, 1):.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAR0 reward diagnostics")
    parser.add_argument("--env", default="PointLTL0MASAR1-v0")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-sb3", action="store_true", help="Use dict (non-SB3) env API")
    args = parser.parse_args()
    run_diagnostics(args.env, episodes=args.episodes, seed=args.seed, sb3=not args.no_sb3)
