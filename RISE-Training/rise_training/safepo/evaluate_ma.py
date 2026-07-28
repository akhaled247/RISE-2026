"""Multi-agent SafePO evaluation for SpecRLBench MASAR runs."""

from __future__ import annotations

import argparse
import json
import os
from distutils.util import strtobool
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import trange

from rise_training.safepo.evaluate import (
    _classify_fail,
    _format_line,
    _format_rescue_rates,
    _print_eval_path,
    _seed_everything,
    _write_eval_summary,
)
from rise_training.safepo.ma_factory import cmdp_cost_channels


MA_ALGOS = {"mappo", "happo", "mappolag", "mappo_lag", "macpo", "ippo", "ippo_lag"}


def _load_ma_config(run_dir: str) -> dict[str, Any]:
    path = os.path.join(os.path.abspath(run_dir), "config.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _find_models_dir(run_dir: str, seed: int) -> str:
    run_dir = os.path.abspath(run_dir)
    candidates = [
        os.path.join(run_dir, f"models_seed{seed}"),
        os.path.join(run_dir, f"models_seed{seed}.0"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    # Fallback: any models_seed*
    for name in os.listdir(run_dir):
        if name.startswith("models_seed") and os.path.isdir(os.path.join(run_dir, name)):
            return os.path.join(run_dir, name)
    raise FileNotFoundError(f"No models_seed* under {run_dir}")


def eval_ma_run(
    run_dir: str,
    eval_episodes: int = 50,
    *,
    device: str = "cuda",
    seed: int | None = 0,
    render_mode: str = None,
    deterministic: bool = True,
) -> dict[str, float]:
    from rise_training.safepo.env_hook import is_specrlbench_env, patch_safepo_ma_env_factory
    from rise_training.safepo.ma_factory import SpecRLMultiGoalEnv
    from rise_training.paths import ensure_specrlbench_paths
    from safepo.common.model import ActorVCritic, MultiAgentActor

    ensure_specrlbench_paths()
    patch_safepo_ma_env_factory()

    if seed is not None:
        _seed_everything(int(seed))

    config = _load_ma_config(run_dir)
    env_id = config.get("env_name") or config.get("task")
    if env_id is None:
        raise KeyError("config.json missing env_name/task")
    if not is_specrlbench_env(str(env_id)):
        raise ValueError(f"Not a SpecRL env: {env_id}")

    algo = str(config.get("algorithm_name", "mappo")).lower().replace("-", "_")
    device_t = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    use_ippo_spine = algo in ("ippo", "ippo_lag")

    env = SpecRLMultiGoalEnv(task=str(env_id), seed=int(seed or 0), render_mode=render_mode)
    use_walls, use_collision = cmdp_cost_channels(str(env_id))
    num_agents = env.num_agents
    models_dir = _find_models_dir(run_dir, int(config.get("seed", seed or 0)))

    actors = []
    share_policy = bool(config.get("share_policy", algo.startswith("ippo")))
    hidden = int(config.get("hidden_size", 64))
    hidden_sizes = config.get("hidden_sizes", [hidden, hidden])
    for agent_id in range(num_agents):
        aid = 0 if share_policy else agent_id
        actor_path = os.path.join(models_dir, f"actor_agent{aid}.pt")
        if not os.path.isfile(actor_path):
            actor_path = os.path.join(models_dir, f"actor_agent{agent_id}.pt")
        if use_ippo_spine:
            obs_dim = int(env.observation_spaces[f"agent_{agent_id}"].shape[0])
            act_dim = int(env.action_spaces[env.possible_agents[agent_id]].shape[0])
            policy = ActorVCritic(
                obs_dim=obs_dim,
                act_dim=act_dim,
                hidden_sizes=list(hidden_sizes),
            ).to(device_t)
            state = torch.load(actor_path, map_location=device_t, weights_only=False)
            policy.actor.load_state_dict(state)
            policy.eval()
            actors.append(policy)
        else:
            actor = MultiAgentActor(
                config,
                env.observation_spaces[f"agent_{agent_id}"],
                env.action_spaces[env.possible_agents[agent_id]],
                device_t,
            )
            state = torch.load(actor_path, map_location=device_t, weights_only=False)
            actor.load_state_dict(state)
            actor.eval()
            actors.append(actor)

    rews, costs, lens = [], [], []
    full_count = partial_count = none_count = 0
    total_casualty_rescues = 0
    fail_walls = fail_collision = fail_timeout = fail_other = 0
    casualty_num = 1
    ep_seed = int(seed or 0)

    for _ in trange(eval_episodes):
        obs_n, _, _ = env.reset(seed=ep_seed)
        ep_rew = ep_cost = ep_len = 0.0
        saw_walls = saw_collision = False
        last_trunc = False
        done = False
        rnn = [
            torch.zeros(1, int(config.get("recurrent_N", 1)), int(config.get("hidden_size", 128)), device=device_t)
            for _ in range(num_agents)
        ]
        masks = [torch.ones(1, 1, device=device_t) for _ in range(num_agents)]

        while not done:
            actions = []
            for agent_id in range(num_agents):
                obs_t = torch.as_tensor(obs_n[agent_id], dtype=torch.float32, device=device_t).unsqueeze(0)
                with torch.no_grad():
                    if use_ippo_spine:
                        act, _, _, _ = actors[agent_id].step(
                            obs_t, deterministic=deterministic
                        )
                    else:
                        act, _, rnn[agent_id] = actors[agent_id](
                            obs_t, rnn[agent_id], masks[agent_id], deterministic=True
                        )
                actions.append(act.squeeze(0))
            obs_n, share_n, rewards, costs_l, dones, infos, _ = env.step(actions)
            del share_n
            ep_rew += float(np.mean([r[0] for r in rewards]))
            ep_cost += float(np.mean([c[0] for c in costs_l]))
            ep_len += 1
            for info in infos:
                if use_walls and float(info.get("cost_walls", 0) or 0) > 0:
                    saw_walls = True
                if use_collision and float(info.get("cost_collision", 0) or 0) > 0:
                    saw_collision = True
            done = all(bool(d) for d in dones)
            # Approximate trunc: no wall/collision and hit length budget
            last_trunc = done and not (saw_walls or saw_collision)

        flags = []
        try:
            task = env.env.unwrapped.task
            if hasattr(task, "_casualtys_rescued"):
                flags = [bool(x) for x in task._casualtys_rescued()]
        except Exception:
            flags = []
        if flags:
            casualty_num = len(flags)
            n_rescued = sum(flags)
        else:
            n_rescued = 0
        total_casualty_rescues += n_rescued
        if n_rescued >= casualty_num and casualty_num > 0:
            full_count += 1
        elif n_rescued > 0:
            partial_count += 1
        else:
            none_count += 1
        fail = _classify_fail(
            success=n_rescued >= casualty_num and casualty_num > 0,
            saw_walls=saw_walls,
            saw_collision=saw_collision,
            truncated=last_trunc,
        )
        if fail == "cost_walls":
            fail_walls += 1
        elif fail == "cost_collision":
            fail_collision += 1
        elif fail == "timeout":
            fail_timeout += 1
        elif fail == "other":
            fail_other += 1
        rews.append(ep_rew)
        costs.append(ep_cost)
        lens.append(ep_len)
        ep_seed += 1

    env.close()
    return {
        "mean_reward": float(np.mean(rews)),
        "std_reward": float(np.std(rews)),
        "mean_cost": float(np.mean(costs)),
        "std_cost": float(np.std(costs)),
        "mean_ep_len": float(np.mean(lens)),
        "std_ep_len": float(np.std(lens)),
        "rescue_rate": float(full_count) / float(eval_episodes),
        "eval_episodes": float(eval_episodes),
        "casualty_num": float(casualty_num),
        "rescue_full": float(full_count),
        "rescue_partial": float(partial_count),
        "rescue_none": float(none_count),
        "average_rescue_rate": float(total_casualty_rescues)
        / float(max(1, eval_episodes * casualty_num)),
        "total_casualty_rescues": float(total_casualty_rescues),
        "fail_cost_walls": float(fail_walls),
        "fail_cost_collision": float(fail_collision),
        "fail_timeout": float(fail_timeout),
        "fail_other": float(fail_other),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="SpecRLBench SafePO MA evaluation")
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--deterministic",
        type=lambda v: bool(strtobool(str(v))),
        default=True,
        help="Use mean actions (default True). Pass False for stochastic eval.",
    )
    p.add_argument(
            "--render-mode",
            type=str,
            default=None,
            help="Gymnasium render mode, e.g. human (live window) or rgb_array",
        )
    args = p.parse_args(argv)

    metrics = eval_ma_run(
        args.run_dir,
        eval_episodes=args.eval_episodes,
        device=args.device,
        seed=args.seed,
        render_mode=args.render_mode,
        deterministic=args.deterministic,
    )
    parts = Path(args.run_dir).resolve().parts
    algo = parts[-2] if len(parts) >= 2 else "unknown"
    env = parts[-3] if len(parts) >= 3 else "unknown"
    line = _format_line(env, algo, metrics, args.eval_episodes)
    _print_eval_path(args.run_dir)
    print(line.strip())
    # Also print explicit rescue-rate block if missing from line (already included)
    _write_eval_summary(
        args.run_dir,
        {
            "EVAL_PATH": os.path.abspath(args.run_dir),
            "env": env,
            "algo": algo,
            "eval_episodes": args.eval_episodes,
            **metrics,
        },
    )


if __name__ == "__main__":
    main()
