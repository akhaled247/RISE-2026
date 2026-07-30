"""Launch installed SafePO single-agent algorithms on SpecRLBench envs.

Algorithms are imported from the ``safepo`` package (``pip install safepo``).
This module patches the env factory, merges SpecRL ``SafePOTrainConfig`` into
SafePO ``default_cfg`` / ``args``, then calls SafePO ``main()``.
"""

from __future__ import annotations

import os
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

from rise_training.paths import default_log_dir

from rise_training.safepo.config import ALGO_DEFAULTS, SAR_PAPER_PROTOCOL, SafePOTrainConfig
from rise_training.safepo.env_hook import patch_safepo_env_factory, set_parallel, set_sar_ltl_ordering
from rise_training.safepo.registry import resolve_algo

# Map SpecRLBench algo names → SafePO single_agent modules
_SAFEPO_MODULES = {
    "ppo": "safepo.single_agent.ppo",
    "trpo": "safepo.single_agent.trpo",
    "ppo_lag": "safepo.single_agent.ppo_lag",
    "trpo_lag": "safepo.single_agent.trpo_lag",
    "cpo": "safepo.single_agent.cpo",
}

_CFG = SafePOTrainConfig()


def _redirect_terminal_logs(log_dir: str, seed: int) -> None:
    """Match SafePO ``if __name__ == '__main__'`` when write_terminal=False."""
    os.makedirs(log_dir, exist_ok=True)
    term = os.path.join(log_dir, f"seed{seed}_terminal.log")
    err = os.path.join(log_dir, f"seed{seed}_error.log")
    # Keep handles open for process lifetime (same pattern as upstream SafePO).
    sys.stdout = open(term, "w", encoding="utf-8")  # noqa: SIM115
    sys.stderr = open(err, "w", encoding="utf-8")  # noqa: SIM115


def _patch_epoch_logger_tensorboard(use_tensorboard: bool, algo_mod: Any = None) -> None:
    """SafePO mains hardcode EpochLogger(...); inject use_tensorboard.

    Algo modules do ``from safepo.common.logger import EpochLogger``, so we must
    rebind both ``safepo.common.logger.EpochLogger`` and ``algo_mod.EpochLogger``.
    """
    import safepo.common.logger as logger_mod

    if not hasattr(logger_mod, "_EpochLoggerOrig"):
        logger_mod._EpochLoggerOrig = logger_mod.EpochLogger
    _Orig = logger_mod._EpochLoggerOrig

    class EpochLogger(_Orig):  # type: ignore[valid-type, misc]
        def __init__(self, *a, **kw):
            kw["use_tensorboard"] = use_tensorboard
            super().__init__(*a, **kw)

    logger_mod.EpochLogger = EpochLogger
    if algo_mod is not None and hasattr(algo_mod, "EpochLogger"):
        algo_mod.EpochLogger = EpochLogger


def _merge_train_kwargs(algo: str, extra: dict[str, Any]) -> dict[str, Any]:
    """Fill missing knobs from SafePOTrainConfig + ALGO_DEFAULTS."""
    base = _CFG.to_dict()
    # Drop non-arg fields
    for k in ("env_id", "algo", "normalize_obs", "clip_obs", "use_eval", "eval_episodes"):
        base.pop(k, None)
    base.update(ALGO_DEFAULTS.get(algo, {}))
    # Caller overrides win
    base.update(extra)
    return base


def _default_args(
    *,
    task: str,
    seed: int = 0,
    total_steps: int = 1_000_000,
    num_envs: int = 8,
    steps_per_epoch: int = 16_384,
    cost_limit: float = 0.0,
    device: str = "cpu",
    device_id: int = 0,
    log_dir: str | None = None,
    experiment: str = "specrlbench",
    use_eval: bool = False,
    write_terminal: bool = True,
    use_tensorboard: bool = True,
    **extra: Any,
) -> Namespace:
    """Build argparse-like Namespace matching SafePO ``single_agent_args`` + SpecRL knobs."""
    if log_dir is None:
        log_dir = default_log_dir()
    args = Namespace(
        task=task,
        seed=seed,
        total_steps=total_steps,
        num_envs=num_envs,
        steps_per_epoch=steps_per_epoch,
        cost_limit=cost_limit,
        device=device,
        device_id=device_id,
        log_dir=log_dir,
        experiment=experiment,
        use_eval=use_eval,
        write_terminal=write_terminal,
        use_tensorboard=use_tensorboard,
        lagrangian_multiplier_init=extra.pop(
            "lagrangian_multiplier_init", _CFG.lagrangian_multiplier_init
        ),
        lagrangian_multiplier_lr=extra.pop(
            "lagrangian_multiplier_lr", _CFG.lagrangian_multiplier_lr
        ),
        parallel=1,
        torch_threads=4,
        # SpecRL → Linux-edited SafePO (getattr fallbacks in clone)
        actor_lr=extra.pop("actor_lr", _CFG.actor_lr),
        critic_lr=extra.pop("critic_lr", _CFG.critic_lr),
        batch_size=extra.pop("batch_size", _CFG.batch_size),
        learning_iters=extra.pop("learning_iters", _CFG.learning_iters),
        target_kl=extra.pop("target_kl", _CFG.target_kl),
        gamma=extra.pop("gamma", _CFG.gamma),
        lam=extra.pop("lam", _CFG.lam),
        lam_c=extra.pop("lam_c", _CFG.lam_c),
        clip_ratio=extra.pop("clip_ratio", _CFG.clip_ratio),
        ent_coef=extra.pop("ent_coef", _CFG.ent_coef),
        max_grad_norm=extra.pop("max_grad_norm", _CFG.max_grad_norm),
        hidden_sizes=extra.pop("hidden_sizes", list(_CFG.hidden_sizes)),
    )
    for k, v in extra.items():
        setattr(args, k, v)
    return args


def _patch_safepo_default_cfg(mod: Any, args: Namespace) -> dict[str, Any]:
    """Update SafePO module ``default_cfg`` from SpecRL args (Phase 8)."""
    if not hasattr(mod, "default_cfg"):
        return {}
    cfg = dict(mod.default_cfg)
    updates = {
        "hidden_sizes": list(getattr(args, "hidden_sizes", _CFG.hidden_sizes)),
        "gamma": float(getattr(args, "gamma", _CFG.gamma)),
        "target_kl": float(getattr(args, "target_kl", _CFG.target_kl)),
        "batch_size": int(getattr(args, "batch_size", _CFG.batch_size)),
        "learning_iters": int(getattr(args, "learning_iters", _CFG.learning_iters)),
        "max_grad_norm": float(getattr(args, "max_grad_norm", _CFG.max_grad_norm)),
    }
    cfg.update(updates)
    mod.default_cfg = cfg
    return updates


def _apply_sar_paper_protocol(env_id: str, merged: dict[str, Any]) -> None:
    """Apply GenZ-adjacent SAR solo-train recipe on MASAR1WC tasks."""
    if "MASAR1" not in env_id or "WC" not in env_id:
        return
    skip = {"train_env", "eval_env", "team_done"}
    for key, value in SAR_PAPER_PROTOCOL.items():
        if key not in skip:
            merged.setdefault(key, value)


def train_with_safepo(
    algo: str,
    env_id: str,
    *,
    seed: int = 0,
    total_steps: int | None = None,
    num_envs: int | None = None,
    steps_per_epoch: int | None = None,
    cost_limit: float | None = None,
    device: str | None = None,
    device_id: int = 0,
    log_dir: str | None = None,
    experiment: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Patch env factory, merge config into SafePO, then run ``main(args)``."""
    algo = resolve_algo(algo)
    if algo not in _SAFEPO_MODULES:
        raise ValueError(f"No SafePO module mapping for {algo!r}")

    merged = _merge_train_kwargs(algo, dict(extra))
    _apply_sar_paper_protocol(env_id, merged)
    if total_steps is not None:
        merged["total_steps"] = total_steps
    if num_envs is not None:
        merged["num_envs"] = num_envs
    if steps_per_epoch is not None:
        merged["steps_per_epoch"] = steps_per_epoch
    if cost_limit is not None:
        merged["cost_limit"] = cost_limit
    if device is not None:
        merged["device"] = device
    if log_dir is not None:
        merged["log_dir"] = log_dir
    if experiment is not None:
        merged["experiment"] = experiment

    # Avoid duplicate kwargs: explicit seed/device_id win over config dict.
    merged.pop("seed", None)
    merged.pop("device_id", None)
    

    from rise_training.paths import ensure_specrlbench_paths
    from rise_training.safepo.torch_compat import patch_linear_lr_verbose

    ensure_specrlbench_paths()
    patch_safepo_env_factory()

    # SpecRL vec parallelism (SafetyAsync); not SafePO MA ``args.parallel``.
    set_parallel(bool(merged.pop("parallel", _CFG.parallel)))
    sar_ltl_ordering = bool(merged.pop("sar_ltl_ordering", False))
    set_sar_ltl_ordering(sar_ltl_ordering)

    import importlib

    mod = importlib.import_module(_SAFEPO_MODULES[algo])
    patch_linear_lr_verbose(mod)

    args = _default_args(
        task=env_id,
        seed=seed,
        device_id=device_id,
        sar_ltl_ordering=sar_ltl_ordering,
        **merged,
    )

    cfg_patch = _patch_safepo_default_cfg(mod, args)

    # Layout: {log_dir}/{task}/{algo}/seed-NNN-TIMESTAMP (experiment is config metadata only)
    relpath = time.strftime("%Y-%m-%d-%H-%M-%S")
    subfolder = "-".join(["seed", str(args.seed).zfill(3)])
    relpath = "-".join([subfolder, relpath])
    args.log_dir = os.path.join(args.log_dir, args.task, algo, relpath)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    _patch_epoch_logger_tensorboard(
        bool(getattr(args, "use_tensorboard", True)),
        algo_mod=mod,
    )
    if not getattr(args, "write_terminal", True):
        _redirect_terminal_logs(args.log_dir, args.seed)

    # SafePO mains expect (args, cfg_env=None) for mujoco path
    mod.main(args, None)

    _augment_run_config(
        args.log_dir,
        env_id=env_id,
        sar_ltl_ordering=sar_ltl_ordering,
    )

    return {
        "log_dir": args.log_dir,
        "algo": algo,
        "env_id": env_id,
        "default_cfg_patch": cfg_patch,
    }


def _augment_run_config(
    log_dir: str,
    *,
    env_id: str,
    sar_ltl_ordering: bool,
) -> None:
    """Merge SA deploy obs metadata into SafePO config.json when missing."""
    import json

    from rise_training.cmdp.obs_spec import probe_flatten_keys, probe_train_obs_dim

    config_path = os.path.join(log_dir, "config.json")
    if not os.path.isfile(config_path):
        return
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    updated = False
    if "sar_ltl_ordering" not in config:
        config["sar_ltl_ordering"] = bool(sar_ltl_ordering)
        updated = True
    if "flatten_keys" not in config:
        config["flatten_keys"] = probe_flatten_keys(
            env_id, sar_ltl_ordering=bool(config.get("sar_ltl_ordering", sar_ltl_ordering))
        )
        updated = True
    if "obs_dim" not in config:
        config["obs_dim"] = probe_train_obs_dim(
            env_id, sar_ltl_ordering=bool(config.get("sar_ltl_ordering", sar_ltl_ordering))
        )
        updated = True
    if updated:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)


def train_algo(algo: str, **overrides: Any) -> dict[str, Any]:
    env_id = overrides.pop("env_id", overrides.pop("task", "PointLTL1MASAR1WC-v0"))
    return train_with_safepo(algo, env_id, **overrides)
