#!/usr/bin/env python3
"""Diagnose SafePO train EpRet vs post-train eval gap (obs RMS / det vs stoch).

Does **not** modify production eval. Imports helpers from
``backends.safepo.evaluate`` and mirrors its rollout with an explicit
``deterministic`` flag plus action-norm sanity.

Example (Linux, from SpecRLBench root)::

  python scripts/diag_safepo_train_eval_gap.py \\
    --run-dir ./_training_logs/safepo/PointLTL5MASAR1WC-v0/ppo/seed-000-2026-07-22-07-52-51 \\
    --eval-episodes 20 --device cuda --seed 0

Exit code 2 if no Normalizer ``*.pkl`` in the run dir (hard fail for this diag).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

# SpecRLBench root on sys.path (script lives in scripts/)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _itr_from_name(name: str) -> int | None:
    """Trailing digits in ``model99.pt`` / ``state10.pkl`` → 99 / 10."""
    import re

    m = re.search(r"(\d+)(?:\.[^.]+)?$", name)
    return int(m.group(1)) if m else None


def _list_ckpt_itrs(run_dir: str) -> tuple[list[int], list[int]]:
    """Return sorted model / state iteration lists from a run dir."""
    model_dir = os.path.join(run_dir, "torch_save")
    model_itrs: list[int] = []
    if os.path.isdir(model_dir):
        for m in os.listdir(model_dir):
            if m.endswith(".pt"):
                itr = _itr_from_name(m)
                if itr is not None:
                    model_itrs.append(itr)
    state_itrs: list[int] = []
    for p in os.listdir(run_dir):
        if p.endswith(".pkl"):
            itr = _itr_from_name(p)
            if itr is not None:
                state_itrs.append(itr)
    return sorted(model_itrs), sorted(state_itrs)


def _inspect_pkl(norm_path: str) -> dict[str, Any]:
    """Load joblib Normalizer blob; return RMS summary fields."""
    import joblib

    state = joblib.load(norm_path)
    if isinstance(state, dict) and "Normalizer" in state:
        normalizer = state["Normalizer"]
    else:
        normalizer = state

    if hasattr(normalizer, "mean") and hasattr(normalizer, "var"):
        mean = np.asarray(normalizer.mean, dtype=np.float64)
        var = np.asarray(normalizer.var, dtype=np.float64)
        count = float(getattr(normalizer, "count", float("nan")))
    elif isinstance(normalizer, dict):
        mean = np.asarray(normalizer["mean"], dtype=np.float64)
        var = np.asarray(normalizer["var"], dtype=np.float64)
        count = float(normalizer.get("count", float("nan")))
    else:
        raise TypeError(f"Unsupported Normalizer type: {type(normalizer)}")

    return {
        "count": count,
        "mean_l2": float(np.linalg.norm(mean)),
        "var_mean": float(np.mean(var)),
        "obs_dim": int(mean.size),
    }


def _rollout_metrics(
    *,
    run_dir: str,
    eval_episodes: int,
    device: str,
    seed: int,
    deterministic: bool,
) -> dict[str, float]:
    """Mirror ``eval_single_run`` with controllable ``deterministic`` + action |a|."""
    from backends.safepo.evaluate import (
        _info_dict,
        _load_run_artifacts,
        _rescued_from_info,
        _seed_everything,
    )
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
    _seed_everything(int(seed))

    config, model_path, norm_path = _load_run_artifacts(run_dir)
    env_id = config.get("task") or config.get("env_name")
    if env_id is None:
        raise KeyError("config.json missing task/env_name")
    if not is_specrlbench_env(str(env_id)):
        raise ValueError(f"Env {env_id!r} is not a SpecRLBench Point/Car/AntLTL* task")

    hidden_sizes = config.get("hidden_sizes", [64, 64])
    torch.set_num_threads(int(config.get("torch_threads", 4)))
    device_t = torch.device(device)

    eval_env, obs_space, act_space = make_specrlbench_sa_env(
        1, str(env_id), seed=seed, training=False
    )

    if norm_path is not None and os.path.isfile(norm_path):
        import joblib

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
    abs_act_sums: list[float] = []
    abs_act_counts: list[int] = []
    ep_seed = int(seed)

    for ep_i in range(eval_episodes):
        eval_done = False
        eval_obs, info0 = eval_env.reset(seed=ep_seed)
        eval_obs = torch.as_tensor(eval_obs, dtype=torch.float32, device=device_t)
        eval_rew, eval_cost, eval_len = 0.0, 0.0, 0.0
        rescued = _rescued_from_info(_info_dict(info0))
        abs_sum = 0.0
        abs_n = 0
        while not eval_done:
            with torch.no_grad():
                act, _, _, _ = model.step(eval_obs, deterministic=deterministic)
            action = act.detach().squeeze().cpu().numpy()
            abs_sum += float(np.mean(np.abs(action)))
            abs_n += 1
            eval_obs_np, reward, cost, terminated, truncated, info = eval_env.step(
                action
            )
            eval_obs = torch.as_tensor(eval_obs_np, dtype=torch.float32, device=device_t)
            r = float(np.asarray(reward).reshape(-1)[0])
            c = float(np.asarray(cost).reshape(-1)[0])
            term = bool(np.asarray(terminated).reshape(-1)[0])
            trunc = bool(np.asarray(truncated).reshape(-1)[0])
            eval_rew += r
            eval_cost += c
            eval_len += 1
            if _rescued_from_info(_info_dict(info)):
                rescued = True
            eval_done = term or trunc
        rew_deque.append(eval_rew)
        cost_deque.append(eval_cost)
        len_deque.append(eval_len)
        if rescued:
            rescue_count += 1
        if ep_i == 0 and abs_n > 0:
            abs_act_sums.append(abs_sum / abs_n)
            abs_act_counts.append(abs_n)
        ep_seed += 1

    eval_env.close()

    out: dict[str, float] = {
        "mean_reward": float(np.mean(rew_deque)),
        "std_reward": float(np.std(rew_deque)),
        "mean_cost": float(np.mean(cost_deque)),
        "std_cost": float(np.std(cost_deque)),
        "mean_ep_len": float(np.mean(len_deque)),
        "std_ep_len": float(np.std(len_deque)),
        "rescue_rate": float(rescue_count) / float(eval_episodes),
        "eval_episodes": float(eval_episodes),
    }
    if abs_act_sums:
        out["mean_abs_action_ep0"] = float(abs_act_sums[0])
        out["ep0_steps"] = float(abs_act_counts[0])
    return out


def _print_metrics(label: str, m: dict[str, float]) -> None:
    n = int(m["eval_episodes"])
    print(
        f"  {label}: reward {m['mean_reward']:.4f}±{m['std_reward']:.4f}, "
        f"cost {m['mean_cost']:.4f}±{m['std_cost']:.4f}, "
        f"ep_len {m['mean_ep_len']:.2f}±{m['std_ep_len']:.2f}, "
        f"rescue {100 * m['rescue_rate']:.1f}% (n={n})"
    )
    if "mean_abs_action_ep0" in m:
        print(
            f"    action |a| mean (ep0, {int(m['ep0_steps'])} steps): "
            f"{m['mean_abs_action_ep0']:.6f}"
        )


def main(argv: list[str] | None = None) -> int:
    from backends.safepo.evaluate import _load_run_artifacts

    p = argparse.ArgumentParser(
        description="Diagnose SafePO train EpRet vs eval gap (RMS + det/stoch)"
    )
    p.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Seed run folder with config.json + torch_save/ + *.pkl",
    )
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    run_dir = os.path.abspath(args.run_dir)
    print("=" * 60)
    print("SafePO train↔eval gap diagnosis")
    print(f"run_dir={run_dir}")
    print("=" * 60)

    config, model_path, norm_path = _load_run_artifacts(run_dir)
    task = config.get("task") or config.get("env_name")
    print("\n[1] config.json")
    print(f"  task={task}")
    print(f"  total_steps={config.get('total_steps')}")
    print(f"  steps_per_epoch={config.get('steps_per_epoch')}")
    print(f"  num_envs={config.get('num_envs')}")
    print(f"  gamma={config.get('gamma')}  lam={config.get('lam')}  lam_c={config.get('lam_c')}")
    print(f"  actor_lr={config.get('actor_lr')}  lr_end_factor={config.get('lr_end_factor')}")
    print(f"  target_kl={config.get('target_kl')}  clip_ratio={config.get('clip_ratio')}")

    total_steps = int(config.get("total_steps") or 0)
    steps_per_epoch = int(config.get("steps_per_epoch") or 0)
    num_envs = int(config.get("num_envs") or 1)
    expected_epochs = (
        total_steps // steps_per_epoch if steps_per_epoch > 0 else 0
    )
    expected_last = max(expected_epochs - 1, 0)
    model_itrs, state_itrs = _list_ckpt_itrs(run_dir)
    max_model = max(model_itrs) if model_itrs else -1
    max_state = max(state_itrs) if state_itrs else -1
    max_ckpt = max(max_model, max_state)
    # One epoch of local worker steps ≈ steps_per_epoch / num_envs (async RMS)
    min_rms_count = (
        float(steps_per_epoch) / float(max(num_envs, 1)) if steps_per_epoch > 0 else 0.0
    )

    print("\n[2] Checkpoint cadence")
    print(f"  expected_epochs={expected_epochs} (last itr={expected_last})")
    print(f"  model*.pt itrs={model_itrs}")
    print(f"  state*.pkl itrs={state_itrs}")
    if expected_epochs > 1 and max_ckpt >= 0 and max_ckpt < expected_last * 0.5:
        print(
            f"  WARN: max ckpt itr={max_ckpt} << expected last={expected_last} — "
            "likely %100-only save; final weights may be missing (retrain after cadence fix)."
        )
    elif expected_epochs > 1 and max_ckpt >= 0 and max_ckpt < expected_last:
        print(
            f"  WARN: max ckpt itr={max_ckpt} < expected last={expected_last} — "
            "end-of-run save may be missing."
        )

    print("\n[3] Normalizer *.pkl")
    if norm_path is None or not os.path.isfile(norm_path):
        print("  FAIL: no *.pkl in run dir — eval without frozen RMS is invalid for this diag.")
        print("  Fix train save path / re-run with Normalizer export before trusting eval.")
        return 2
    print(f"  path={norm_path}")
    try:
        rms = _inspect_pkl(norm_path)
    except Exception as e:
        print(f"  FAIL: could not inspect pkl: {e}")
        return 2
    print(f"  count={rms['count']}")
    print(f"  mean_l2={rms['mean_l2']:.6f}")
    print(f"  var_mean={rms['var_mean']:.6f}")
    print(f"  obs_dim={rms['obs_dim']}")
    if not np.isfinite(rms["count"]) or rms["count"] < 1.0:
        print("  WARN: RMS count looks empty/tiny — strong transfer-bug signal.")
    elif min_rms_count > 0 and rms["count"] < min_rms_count * 0.9:
        print(
            f"  WARN: RMS count={rms['count']:.0f} << one-epoch local steps "
            f"~{min_rms_count:.0f} (steps_per_epoch/num_envs) — stale epoch-0 normalizer?"
        )

    print("\n[4] Checkpoint file")
    print(f"  model={model_path}")

    print("\n[5] Eval side-by-side (same seed start, N episodes each)")
    print(f"  episodes={args.eval_episodes}  device={args.device}  seed={args.seed}")

    m_det = _rollout_metrics(
        run_dir=run_dir,
        eval_episodes=args.eval_episodes,
        device=args.device,
        seed=args.seed,
        deterministic=True,
    )
    _print_metrics("deterministic=True ", m_det)

    m_sto = _rollout_metrics(
        run_dir=run_dir,
        eval_episodes=args.eval_episodes,
        device=args.device,
        seed=args.seed,
        deterministic=False,
    )
    _print_metrics("deterministic=False", m_sto)

    print("\n[6] How to read")
    print("  Root cause class: ckpt cadence (%100) → only model0/state0 on short runs.")
    print("  Train last EpRet~0.65 on 07-52-51 was live+unsaved; cannot recover — retrain.")
    print("  If both modes reward~0 + max ckpt<<epochs: weights never on disk.")
    print("  Det wall-death (ep~300,cost~0.9) vs stoch timeout (ep~975,cost~0): known.")
    print("  Tiny mean |a|: possible std collapse / dead actor.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
