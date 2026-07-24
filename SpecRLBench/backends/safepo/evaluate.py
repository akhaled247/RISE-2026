"""Post-train SafePO evaluation for SpecRLBench (mirrors safepo/evaluate.py SA path).

Loads last ``torch_save/*.pt`` + Normalizer ``*.pkl`` from a run directory,
builds CMDP env via the SpecRLBench env hook, rolls deterministic policy,
reports EpRet / EpCost / ep_len / rescue rate.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _itr_from_name(name: str) -> int | None:
    """Trailing digits in ``model99.pt`` / ``state10.pkl`` → 99 / 10."""
    import re

    m = re.search(r"(\d+)(?:\.[^.]+)?$", name)
    return int(m.group(1)) if m else None


def _latest(paths: list[str]) -> str:
    """Pick max numeric iteration (``model100`` > ``model99``), not string sort."""
    if not paths:
        raise FileNotFoundError("No matching checkpoint files")

    def _key(name: str) -> tuple[int, str]:
        itr = _itr_from_name(name)
        return (itr if itr is not None else -1, name)

    return max(paths, key=_key)


def _load_run_artifacts(run_dir: str) -> tuple[dict[str, Any], str, str | None]:
    """Return (config, model_path, norm_path_or_None)."""
    run_dir = os.path.abspath(run_dir)
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Missing config.json in {run_dir}")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    model_dir = os.path.join(run_dir, "torch_save")
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Missing torch_save/ in {run_dir}")
    models = [m for m in os.listdir(model_dir) if m.endswith(".pt")]
    model_path = os.path.join(model_dir, _latest(models))

    pkls = [p for p in os.listdir(run_dir) if p.endswith(".pkl")]
    norm_path = os.path.join(run_dir, _latest(pkls)) if pkls else None
    return config, model_path, norm_path


def _info_dict(info: Any) -> dict[str, Any]:
    """Normalize vec/single info to a flat dict for SpecRL props."""
    if isinstance(info, (list, tuple)):
        info = info[0] if info else {}
    if not isinstance(info, dict):
        return {}
    # Nested agent_0 (native MA) — uncommon on SB3-flat SAR path.
    if "propositions" not in info and "agent_0" in info and isinstance(info["agent_0"], dict):
        return info["agent_0"]
    return info


def _rescued_from_info(info: dict[str, Any]) -> bool:
    prop_keys = info.get("propositions", []) or []
    return any(
        "cost_casualtys_surface" in k or "cost_casualtys_entrapped" in k for k in prop_keys
    )


def _seed_everything(seed: int) -> None:
    """Seed Python / NumPy / Torch for eval (see backends/safepo/SEEDING.md)."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def eval_single_run(
    run_dir: str,
    eval_episodes: int = 50,
    *,
    device: str = "cpu",
    seed: int | None = 0,
    render_mode: str | None = None,
) -> dict[str, float]:
    """Evaluate one SafePO seed folder. Returns metric dict.

    When ``seed`` is not ``None``, seeds ``random``/``numpy``/``torch``, builds
    the eval env with that seed, then each episode uses ``reset(seed=ep_seed)``
    with ``ep_seed`` starting at ``seed`` (WC layout only honors ``reset(seed=)``;
    see ``SEEDING.md``).

    Pass ``render_mode='human'`` to open a live MuJoCo window during rollouts
    (requires a local display; eval always uses ``num_envs=1``).
    """
    from backends.safepo.env_hook import (
        is_specrlbench_env,
        make_specrlbench_sa_env,
        patch_safepo_env_factory,
    )
    from backends.safepo.paths import ensure_specrlbench_paths
    from envs.cmdp.normalize import apply_rms_normalizer
    from safepo.common.model import ActorVCritic

    ensure_specrlbench_paths()
    patch_safepo_env_factory()

    if seed is not None:
        _seed_everything(int(seed))

    config, model_path, norm_path = _load_run_artifacts(run_dir)
    env_id = config.get("task") or config.get("env_name")
    if env_id is None:
        raise KeyError("config.json missing task/env_name")
    if config.get("algorithm_name") in ("macpo", "mappo", "mappolag", "happo"):
        raise NotImplementedError(
            "Multi-agent SafePO eval is out of scope for this SpecRLBench path"
        )
    if not is_specrlbench_env(str(env_id)):
        raise ValueError(
            f"Env {env_id!r} is not a SpecRLBench Point/Car/AntLTL* task; "
            "use upstream safepo/evaluate.py for stock MuJoCo"
        )

    hidden_sizes = config.get("hidden_sizes", [64, 64])
    torch.set_num_threads(int(config.get("torch_threads", 4)))
    device_t = torch.device(device)

    # Always eval with 1 env, frozen RMS updates.
    eval_env, obs_space, act_space = make_specrlbench_sa_env(
        1, str(env_id), seed=seed, training=False, render_mode=render_mode
    )

    if norm_path is not None and os.path.isfile(norm_path):
        try:
            import joblib
        except ImportError as e:
            raise ImportError(
                "joblib required to load SafePO Normalizer *.pkl "
                "(see requirements-safepo.txt)"
            ) from e
        state = joblib.load(norm_path)
        if isinstance(state, dict) and "Normalizer" in state:
            apply_rms_normalizer(eval_env, state["Normalizer"], training=False)
        else:
            apply_rms_normalizer(eval_env, state, training=False)

    model = ActorVCritic(
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
        hidden_sizes=hidden_sizes,
    ).to(device_t)
    state_dict = torch.load(model_path, map_location=device_t, weights_only=False)
    model.actor.load_state_dict(state_dict)
    model.eval()

    rew_deque: deque[float] = deque(maxlen=max(50, eval_episodes))
    cost_deque: deque[float] = deque(maxlen=max(50, eval_episodes))
    len_deque: deque[float] = deque(maxlen=max(50, eval_episodes))
    rescue_count = 0
    ep_seed = seed if seed is not None else 0

    for _ in range(eval_episodes):
        eval_done = False
        eval_obs, info0 = eval_env.reset(seed=ep_seed)
        eval_obs = torch.as_tensor(eval_obs, dtype=torch.float32, device=device_t)
        eval_rew, eval_cost, eval_len = 0.0, 0.0, 0.0
        rescued = _rescued_from_info(_info_dict(info0))
        while not eval_done:
            with torch.no_grad():
                act, _, _, _ = model.step(eval_obs, deterministic=True)
            action = act.detach().squeeze().cpu().numpy()
            eval_obs_np, reward, cost, terminated, truncated, info = eval_env.step(
                action
            )
            eval_obs = torch.as_tensor(eval_obs_np, dtype=torch.float32, device=device_t)
            # Batched (Unsqueeze / vec) or scalar
            r = float(np.asarray(reward).reshape(-1)[0])
            c = float(np.asarray(cost).reshape(-1)[0])
            term = bool(np.asarray(terminated).reshape(-1)[0])
            trunc = bool(np.asarray(truncated).reshape(-1)[0])
            eval_rew += r
            eval_cost += c
            eval_len += 1
            info_d = _info_dict(info)
            if _rescued_from_info(info_d):
                rescued = True
            eval_done = term or trunc
        rew_deque.append(eval_rew)
        cost_deque.append(eval_cost)
        len_deque.append(eval_len)
        if rescued:
            rescue_count += 1
        ep_seed += 1

    eval_env.close()

    metrics = {
        "mean_reward": float(np.mean(rew_deque)),
        "std_reward": float(np.std(rew_deque)),
        "mean_cost": float(np.mean(cost_deque)),
        "std_cost": float(np.std(cost_deque)),
        "mean_ep_len": float(np.mean(len_deque)),
        "std_ep_len": float(np.std(len_deque)),
        "rescue_rate": float(rescue_count) / float(eval_episodes),
        "eval_episodes": float(eval_episodes),
    }
    return metrics


def _iter_run_dirs(benchmark_dir: str) -> list[tuple[str, str, str]]:
    """Yield (env, algo, run_dir) under task/algo/seed-* layout.

    Accepts either:
    - ``.../log_dir`` containing task folders, or
    - ``.../task`` containing algo folders, or
    - a single seed run dir (has config.json) — caller should use --run-dir.
    """
    root = Path(benchmark_dir)
    if not root.is_dir():
        raise NotADirectoryError(benchmark_dir)
    out: list[tuple[str, str, str]] = []
    # Layout from runners: log_dir/task/algo/seed-000-TIMESTAMP
    for task_dir in sorted(root.iterdir()):
        if not task_dir.is_dir():
            continue
        # If this looks like an algo folder (contains seed-* with config.json)
        seed_hits = [
            p
            for p in task_dir.iterdir()
            if p.is_dir() and (p / "config.json").is_file()
        ]
        if seed_hits:
            # benchmark_dir is already task level; task_dir is algo
            for seed_path in seed_hits:
                out.append((root.name, task_dir.name, str(seed_path)))
            continue
        for algo_dir in sorted(task_dir.iterdir()):
            if not algo_dir.is_dir():
                continue
            for seed_path in sorted(algo_dir.iterdir()):
                if seed_path.is_dir() and (seed_path / "config.json").is_file():
                    out.append((task_dir.name, algo_dir.name, str(seed_path)))
    return out


def _format_line(env: str, algo: str, metrics: dict[str, float], n: int) -> str:
    return (
        f"After {n} episodes evaluation, the {algo} in {env} "
        f"reward: {metrics['mean_reward']:.2f}±{metrics['std_reward']:.2f}, "
        f"cost: {metrics['mean_cost']:.2f}±{metrics['std_cost']:.2f}, "
        f"ep_len: {metrics['mean_ep_len']:.2f}±{metrics['std_ep_len']:.2f}, "
        f"rescue: {100 * metrics['rescue_rate']:.1f}%\n"
    )


def _print_eval_path(run_dir: str) -> None:
    """Stdout machine line so pasted evals carry the evaluated seed folder."""
    print(f"EVAL_PATH={os.path.abspath(run_dir)}", flush=True)


def benchmark_eval(
    *,
    benchmark_dir: str | None = None,
    run_dir: str | None = None,
    eval_episodes: int = 50,
    save_dir: str | None = None,
    device: str = "cpu",
    seed: int = 0,
    render_mode: str | None = None,
) -> list[dict[str, Any]]:
    if bool(benchmark_dir) == bool(run_dir):
        raise ValueError("Pass exactly one of --benchmark-dir or --run-dir")

    results: list[dict[str, Any]] = []

    if run_dir is not None:
        metrics = eval_single_run(
            run_dir,
            eval_episodes=eval_episodes,
            device=device,
            seed=seed,
            render_mode=render_mode,
        )
        # Infer env/algo from path: .../task/algo/seed-...
        parts = Path(run_dir).resolve().parts
        algo = parts[-2] if len(parts) >= 2 else "unknown"
        env = parts[-3] if len(parts) >= 3 else "unknown"
        line = _format_line(env, algo, metrics, eval_episodes)
        _print_eval_path(run_dir)
        print(line.strip())
        results.append({"env": env, "algo": algo, "run_dir": run_dir, **metrics})
        if save_dir is None:
            # .../task/algo/seed → .../log_dir/results
            rd = Path(run_dir).resolve()
            save_dir = str(rd.parents[2] / "results") if len(rd.parts) > 3 else str(rd.parent / "results")
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "eval_result.txt"), "a", encoding="utf-8") as f:
            f.write(line)
        return results

    assert benchmark_dir is not None
    if save_dir is None:
        save_dir = benchmark_dir.replace("runs", "results")
        if save_dir == benchmark_dir:
            save_dir = os.path.join(
                os.path.dirname(os.path.abspath(benchmark_dir)), "results"
            )
    os.makedirs(save_dir, exist_ok=True)

    runs = _iter_run_dirs(benchmark_dir)
    if not runs:
        raise FileNotFoundError(
            f"No seed run dirs with config.json under {benchmark_dir}"
        )

    # Group by (env, algo)
    grouped: dict[tuple[str, str], list[str]] = {}
    for env, algo, path in runs:
        grouped.setdefault((env, algo), []).append(path)

    for (env, algo), paths in sorted(grouped.items()):
        print(f"Start evaluating {algo} in {env}")
        rewards, costs = [], []
        rescues = []
        last_metrics: dict[str, float] | None = None
        for path in paths:
            metrics = eval_single_run(
                path,
                eval_episodes=eval_episodes,
                device=device,
                seed=seed,
                render_mode=render_mode,
            )
            _print_eval_path(path)
            last_metrics = metrics
            rewards.append(metrics["mean_reward"])
            costs.append(metrics["mean_cost"])
            rescues.append(metrics["rescue_rate"])
            results.append({"env": env, "algo": algo, "run_dir": path, **metrics})
        assert last_metrics is not None
        agg = {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_cost": float(np.mean(costs)),
            "std_cost": float(np.std(costs)),
            "mean_ep_len": last_metrics["mean_ep_len"],
            "std_ep_len": last_metrics["std_ep_len"],
            "rescue_rate": float(np.mean(rescues)),
        }
        # Aggregate summary only — no fake EVAL_PATH for the group.
        line = _format_line(env, algo, agg, eval_episodes)
        print(line.strip() + f", saved in {save_dir}/eval_result.txt")
        with open(os.path.join(save_dir, "eval_result.txt"), "a", encoding="utf-8") as f:
            f.write(line)

    return results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SpecRLBench SafePO post-train evaluation (SA)",
        epilog=(
            "Example: python eval_safepo_env.py "
            "--run-dir ./_training_logs/safepo/PointLTL1MASAR1WC-v0/ppo/seed-000-... "
            "--eval-episodes 50\n"
            "Stdout (pasteable):\n"
            "  EVAL_PATH=/abs/.../seed-000-...\n"
            "  After 50 episodes evaluation, the ppo in PointLTL… "
            "reward: …, cost: …, ep_len: …, rescue: …%\n"
            "Watch live: add --render-mode human (local display required)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--benchmark-dir",
        type=str,
        default=None,
        help="Parent of task/algo/seed-* runs (SafePO docs layout)",
    )
    p.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Single seed run folder containing config.json + torch_save/",
    )
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--render-mode",
        type=str,
        default=None,
        help="Gymnasium render mode, e.g. human (live window) or rgb_array",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    benchmark_eval(
        benchmark_dir=args.benchmark_dir,
        run_dir=args.run_dir,
        eval_episodes=args.eval_episodes,
        save_dir=args.save_dir,
        device=args.device,
        seed=args.seed,
        render_mode=args.render_mode,
    )


if __name__ == "__main__":
    main()
