"""Deploy a single-agent SafePO checkpoint on a multi-agent SAR env (paper §5.3)."""

from __future__ import annotations

import json
import os
from collections import deque
from typing import Any

import numpy as np
import torch
from tqdm import trange

from rise_training.cmdp.obs_spec import (
    assert_deploy_obs_compatible,
    flatten_ma_agent_for_sa_deploy,
    infer_actor_obs_dim,
    load_rms_from_pkl,
    normalize_obs_vector,
    probe_flatten_keys,
)
from rise_training.safepo.config import SAR_PAPER_PROTOCOL
from rise_training.safepo.evaluate import (
    _casualty_rescued_flags,
    _classify_fail,
    _info_dict,
    _load_run_artifacts,
    _rescued_from_info,
    _seed_everything,
)


def eval_sa_on_ma(
    run_dir: str,
    *,
    eval_env: str | None = None,
    eval_episodes: int = 50,
    seed: int | None = 0,
    device: str = "cpu",
    render_mode: str | None = None,
    sar_ltl_ordering: bool | None = None,
    use_rms: bool = False,
    zero_buildings_visited: bool = True,
) -> dict[str, float]:
    """Roll out shared SA actor on native MA env (flat=False).

    Primary metrics mirror GenZ-LTL MA deploy:
    - ``success_rate`` (S): full team rescue
    - ``violation_rate`` (V): wall-cost termination without success
    - ``mean_success_ep_len`` (AS): mean length of successful episodes only

    Ablation defaults: no RMS; zero ``*_buildings_visited`` (slot kept for obs dim).
    """
    from rise_training.env_utils import make_env
    from rise_training.paths import ensure_specrlbench_paths
    from safepo.common.model import ActorVCritic

    ensure_specrlbench_paths()
    if seed is not None:
        _seed_everything(int(seed))

    config, model_path, norm_path = _load_run_artifacts(run_dir)
    train_env = config.get("task") or config.get("env_name")
    if train_env is None:
        raise KeyError("config.json missing task/env_name")
    eval_env = eval_env or SAR_PAPER_PROTOCOL.get("eval_env", "PointLTL0MASAR2WC-v0")
    if sar_ltl_ordering is None:
        sar_ltl_ordering = bool(config.get("sar_ltl_ordering", False))

    algo_name = str(config.get("algorithm_name") or config.get("algo") or "")
    if algo_name.lower().replace("-", "_") in (
        "macpo", "mappo", "mappolag", "mappo_lag", "happo", "ippo", "ippo_lag",
    ):
        raise NotImplementedError("Use evaluate_ma for MA-trained checkpoints")

    flatten_keys = config.get("flatten_keys") or probe_flatten_keys(
        str(train_env),
        sar_ltl_ordering=sar_ltl_ordering,
    )
    rms = None
    if use_rms and norm_path and os.path.isfile(norm_path):
        rms = load_rms_from_pkl(norm_path)
    print(
        f"[sa_on_ma] use_rms={use_rms and rms is not None} "
        f"zero_buildings_visited={zero_buildings_visited}"
    )

    hidden_sizes = config.get("hidden_sizes", [64, 64])
    device_t = torch.device(device)
    torch.set_num_threads(int(config.get("torch_threads", 4)))

    state_dict = torch.load(model_path, map_location=device_t, weights_only=False)
    obs_dim = int(config.get("obs_dim") or infer_actor_obs_dim(state_dict))

    env = make_env(
        eval_env,
        flat=False,
        render_mode=render_mode,
        sar_ltl_ordering=sar_ltl_ordering,
    )
    agents = list(env.unwrapped.possible_agents)
    obs0, _ = env.reset(seed=seed)
    flat0 = flatten_ma_agent_for_sa_deploy(
        obs0[agents[0]],
        0,
        flatten_keys,
        zero_buildings_visited=zero_buildings_visited,
    )
    assert_deploy_obs_compatible(
        flat0,
        flatten_keys,
        obs_dim,
        train_env=str(train_env),
        eval_env=str(eval_env),
        sar_ltl_ordering=sar_ltl_ordering,
    )
    act_dim = int(np.prod(env.action_space(agents[0]).shape))

    model = ActorVCritic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=hidden_sizes,
    ).to(device_t)
    model.actor.load_state_dict(state_dict)
    model.eval()

    rew_deque: deque[float] = deque(maxlen=max(50, eval_episodes))
    cost_deque: deque[float] = deque(maxlen=max(50, eval_episodes))
    len_deque: deque[float] = deque(maxlen=max(50, eval_episodes))
    success_steps: list[float] = []
    full_count = partial_count = none_count = 0
    violation_count = 0
    total_casualty_rescues = 0
    fail_walls = fail_collision = fail_timeout = fail_other = 0
    collision_events = 0
    casualty_num = 1
    ep_seed = seed if seed is not None else 0

    pbar = trange(eval_episodes)
    for ep_i in pbar:
        obs, info0 = env.reset(seed=ep_seed)
        ep_rew = ep_cost = ep_len = 0.0
        rescued = _rescued_from_info(_info_dict(info0))
        saw_walls = saw_collision = False
        last_trunc = False
        flags0 = _casualty_rescued_flags(env)
        if flags0:
            casualty_num = len(flags0)
        done = False

        while not done:
            actions = {}
            for agent in agents:
                agent_idx = int(agent.rsplit("_", 1)[-1])
                flat = flatten_ma_agent_for_sa_deploy(
                    obs[agent],
                    agent_idx,
                    flatten_keys,
                    zero_buildings_visited=zero_buildings_visited,
                )
                if rms is not None:
                    flat = normalize_obs_vector(flat, rms)
                obs_t = torch.as_tensor(flat, dtype=torch.float32, device=device_t)
                with torch.no_grad():
                    act, _, _, _ = model.step(obs_t, deterministic=True)
                actions[agent] = act.detach().squeeze().cpu().numpy()

            obs, rewards, terminated, truncated, info = env.step(actions)
            ep_len += 1
            if isinstance(rewards, dict):
                # Team sum (not mean) so full-rescue return scale stays comparable to SA.
                ep_rew += float(np.sum([float(rewards[a]) for a in agents]))
            else:
                ep_rew += float(rewards)

            info_d = _info_dict(info)
            if _rescued_from_info(info_d):
                rescued = True
            for agent in agents:
                ai = info.get(agent, {}) if isinstance(info, dict) else {}
                if not isinstance(ai, dict):
                    continue
                if float(ai.get("cost_walls", 0) or 0) > 0:
                    saw_walls = True
                if float(ai.get("cost_collision", 0) or 0) > 0:
                    saw_collision = True
                    collision_events += 1
            # WC wrappers set info["cost"] from walls (and LTL surface-order); do not
            # infer walls from cost when only collision fired.
            if float(info_d.get("cost", 0) or 0) > 0:
                ep_cost += float(info_d.get("cost", 0))

            if isinstance(terminated, dict):
                term = any(bool(terminated[a]) for a in agents)
                trunc = any(bool(truncated[a]) for a in agents)
            else:
                term = bool(terminated)
                trunc = bool(truncated)
            last_trunc = trunc
            done = term or trunc

        flags = _casualty_rescued_flags(env)
        if flags:
            casualty_num = len(flags)
            n_rescued = sum(flags)
        else:
            n_rescued = int(rescued)
            casualty_num = max(casualty_num, 1)
        total_casualty_rescues += n_rescued
        success = n_rescued >= casualty_num
        if success:
            full_count += 1
            rescued = True
            success_steps.append(ep_len)
        elif n_rescued > 0:
            partial_count += 1
        else:
            none_count += 1

        # GenZ-style violation: constraint fail without success (walls on WC).
        if not success and saw_walls:
            violation_count += 1

        fail = _classify_fail(
            success=success,
            saw_walls=saw_walls,
            saw_collision=saw_collision,
            truncated=last_trunc and not rescued,
        )
        if fail == "cost_walls":
            fail_walls += 1
        elif fail == "cost_collision":
            fail_collision += 1
        elif fail == "timeout":
            fail_timeout += 1
        elif fail == "other":
            fail_other += 1

        rew_deque.append(ep_rew)
        cost_deque.append(ep_cost)
        len_deque.append(ep_len)
        ep_seed += 1

        success_rate = float(full_count) / float(ep_i + 1)
        violation_rate = float(violation_count) / float(ep_i + 1)
        as_so_far = float(np.mean(success_steps)) if success_steps else 0.0
        pbar.set_postfix({
            "S": f"{success_rate:.3f}",
            "V": f"{violation_rate:.3f}",
            "AS": f"{as_so_far:.0f}",
        })

    env.close()

    success_rate = float(full_count) / float(eval_episodes)
    violation_rate = float(violation_count) / float(eval_episodes)
    mean_success_ep_len = float(np.mean(success_steps)) if success_steps else float("nan")

    metrics = {
        # GenZ-LTL MA deploy parity
        "success_rate": success_rate,
        "violation_rate": violation_rate,
        "mean_success_ep_len": mean_success_ep_len,
        # Legacy / diagnostic
        "mean_reward": float(np.mean(rew_deque)),
        "std_reward": float(np.std(rew_deque)),
        "mean_cost": float(np.mean(cost_deque)),
        "std_cost": float(np.std(cost_deque)),
        "mean_ep_len": float(np.mean(len_deque)),
        "std_ep_len": float(np.std(len_deque)),
        "rescue_rate": success_rate,
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
        "collision_events": float(collision_events),
        "fail_timeout": float(fail_timeout),
        "fail_other": float(fail_other),
        "train_env": str(train_env),
        "eval_env": str(eval_env),
        "deploy_mode": "sa_on_ma",
        "obs_dim": float(obs_dim),
        "flatten_keys_count": float(len(flatten_keys)),
        "use_rms": float(use_rms and rms is not None),
        "zero_buildings_visited": float(zero_buildings_visited),
    }
    print(
        f"[S={success_rate:.3f}, V={violation_rate:.3f}, "
        f"AS={mean_success_ep_len:.1f}]"
    )
    return metrics


def eval_single_run(
    run_dir: str,
    *,
    eval_env: str | None = None,
    eval_episodes: int = 50,
    seed: int | None = 0,
    device: str = "cpu",
    render_mode: str | None = None,
    sar_ltl_ordering: bool | None = None,
    use_rms: bool = False,
    zero_buildings_visited: bool = True,
) -> str:
    """Evaluate and write ``eval_summary_ma_deploy.json`` next to run dir."""
    metrics = eval_sa_on_ma(
        run_dir,
        eval_env=eval_env,
        eval_episodes=eval_episodes,
        seed=seed,
        device=device,
        sar_ltl_ordering=sar_ltl_ordering,
        render_mode=render_mode,
        use_rms=use_rms,
        zero_buildings_visited=zero_buildings_visited,
    )
    out_path = os.path.join(os.path.abspath(run_dir), "eval_summary_ma_deploy.json")
    payload = {
        "run_dir": os.path.abspath(run_dir),
        "metrics": metrics,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path
