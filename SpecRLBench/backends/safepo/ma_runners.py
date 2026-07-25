"""Launch SafePO multi-agent algorithms on SpecRLBench MASAR tasks."""

from __future__ import annotations

import importlib
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any


_SAFEPO_MA_MODULES = {
    "mappo": "safepo.multi_agent.mappo",
    "happo": "safepo.multi_agent.happo",
    "mappolag": "safepo.multi_agent.mappolag",
    "ippo": "safepo.multi_agent.ippo",
    "ippo_lag": "safepo.multi_agent.ippo_lag",
}


def _resolve_torch_device(device: str, device_id: int) -> str:
    """Normalize to ``cuda:{device_id}`` (e.g. cuda:1) for SafePO cfg_train."""
    if str(device).startswith("cpu"):
        return "cpu"
    if ":" in str(device):
        return str(device)
    return f"cuda:{int(device_id)}"


def _ensure_ma_training_epochs(cfg_train: dict, env_id: str) -> None:
    """Shrink ``episode_length`` when ``num_env_steps`` cannot fit one MAPPO epoch.

    SafePO mamujoco defaults use ``episode_length=1000``; with ``--total-steps 2000``
    and ``--num-envs 8`` that yields ``episodes = 2000//1000//8 = 0`` and immediate exit.
    """
    from backends.safepo.env_hook import is_specrlbench_env

    if not is_specrlbench_env(env_id):
        return

    n_env = int(cfg_train.get("n_rollout_threads", 1))
    ep_len = int(cfg_train.get("episode_length", 1000))
    n_steps = int(cfg_train.get("num_env_steps", 0))
    if n_steps <= 0 or n_env <= 0:
        return

    episodes = n_steps // ep_len // n_env
    if episodes > 0:
        return

    cfg_train["episode_length"] = max(1, n_steps // n_env)


def _ensure_mp_spawn_before_cuda() -> None:
    """Linux default fork + parent CUDA → 'Cannot re-initialize CUDA in forked subprocess'.

    Set spawn while start method still unset. ShareSubprocVecEnv also uses spawn context.
    """
    if mp.get_start_method(allow_none=True) is not None:
        return
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass


def train_with_safepo_ma(
    algo: str,
    env_id: str,
    *,
    seed: int = 0,
    total_steps: int | None = None,
    num_envs: int | None = None,
    cost_limit: float | None = None,
    device: str = "cuda",
    device_id: int = 0,
    log_dir: str = "./_training_logs/safepo",
    experiment: str = "specrlbench",
    write_terminal: bool = True,
    use_tensorboard: bool = True,
    use_eval: bool = False,
    entropy_coef: float | None = None,
    share_policy: bool | None = None,
    model_dir: str = "",
    **_extra: Any,
) -> dict[str, Any]:
    """Patch MA env factory, parse SafePO multi_agent_args, run algo ``train()``."""
    from backends.safepo.env_hook import patch_safepo_ma_env_factory
    from backends.safepo.paths import ensure_specrlbench_paths
    from backends.safepo.registry import MA_ALGO_MODULE, resolve_ma_algo
    from backends.safepo.runners import _patch_epoch_logger_tensorboard

    # PYTHONPATH for spawn workers, spawn before CUDA, farama filter.
    ensure_specrlbench_paths()
    _ensure_mp_spawn_before_cuda()
    from backends.safepo.ma_safepo_patches import apply_ma_safepo_patches

    apply_ma_safepo_patches()
    patch_safepo_ma_env_factory()

    algo_key = resolve_ma_algo(algo)
    safepo_algo = MA_ALGO_MODULE[algo_key]
    if safepo_algo not in _SAFEPO_MA_MODULES:
        raise ValueError(f"No SafePO MA module for {algo_key!r}")

    import torch

    device = _resolve_torch_device(device, device_id)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    elif str(device).startswith("cuda"):
        torch.cuda.set_device(int(str(device).split(":")[1]))

    # multi_agent_args reads sys.argv
    argv = ["ma_train", "--task", env_id, "--seed", str(seed), "--experiment", experiment]
    argv += ["--device", "cuda" if str(device).startswith("cuda") else device, "--device-id", str(device_id)]
    argv += ["--write-terminal", "True" if write_terminal else "False"]
    argv += ["--use-eval", "True" if use_eval else "False"]
    if total_steps is not None:
        argv += ["--total-steps", str(total_steps)]
    if num_envs is not None:
        argv += ["--num-envs", str(num_envs)]
    if cost_limit is not None:
        argv += ["--cost-limit", str(cost_limit)]
    if model_dir:
        argv += ["--model-dir", model_dir]

    old_argv = sys.argv
    sys.argv = argv
    try:
        from safepo.utils.config import multi_agent_args, set_np_formatting, set_seed

        set_np_formatting()
        args, _cfg_env, cfg_train = multi_agent_args(algo=safepo_algo)
    finally:
        sys.argv = old_argv

    set_seed(cfg_train.get("seed", seed), cfg_train.get("torch_deterministic", False))

    cfg_train["device"] = device
    _ensure_ma_training_epochs(cfg_train, env_id)

    # SpecRL log layout: {log_dir}/{task}/{algo}/seed-NNN-TIMESTAMP
    relpath = time.strftime("%Y-%m-%d-%H-%M-%S")
    subfolder = "-".join(["seed", str(args.seed).zfill(3)])
    relpath = "-".join([subfolder, relpath])
    cfg_train["log_dir"] = os.path.join(log_dir, args.task, algo_key, relpath)
    Path(cfg_train["log_dir"]).mkdir(parents=True, exist_ok=True)

    if entropy_coef is not None:
        cfg_train["entropy_coef"] = float(entropy_coef)
    if share_policy is not None:
        cfg_train["share_policy"] = bool(share_policy)
    elif "share_policy" not in cfg_train and safepo_algo.startswith("ippo"):
        cfg_train["share_policy"] = True

    mod = importlib.import_module(_SAFEPO_MA_MODULES[safepo_algo])
    # Ensure patched factory is visible on already-bound names
    import safepo.common.env as safepo_env

    if hasattr(mod, "make_ma_multi_goal_env"):
        mod.make_ma_multi_goal_env = safepo_env.make_ma_multi_goal_env

    # MA algos hardcode EpochLogger(...); inject use_tensorboard like SA path.
    _patch_epoch_logger_tensorboard(bool(use_tensorboard), algo_mod=mod)

    if not write_terminal:
        os.makedirs(cfg_train["log_dir"], exist_ok=True)
        sys.stdout = open(  # noqa: SIM115
            os.path.join(cfg_train["log_dir"], f"seed{args.seed}_terminal.log"),
            "w",
            encoding="utf-8",
        )
        sys.stderr = open(  # noqa: SIM115
            os.path.join(cfg_train["log_dir"], f"seed{args.seed}_error.log"),
            "w",
            encoding="utf-8",
        )

    mod.train(args=args, cfg_train=cfg_train)
    return {"log_dir": cfg_train["log_dir"], "algo": algo_key, "task": env_id}
