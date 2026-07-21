"""Launch installed SafePO single-agent algorithms on SpecRLBench envs.

Algorithms are imported from the ``safepo`` package (``pip install safepo``).
This module only patches the env factory, then calls SafePO ``main()``.
"""

from __future__ import annotations

import os
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

from backends.safepo.env_hook import patch_safepo_env_factory
from backends.safepo.registry import resolve_algo

# Map SpecRLBench algo names → SafePO single_agent modules
_SAFEPO_MODULES = {
    "ppo": "safepo.single_agent.ppo",
    "trpo": "safepo.single_agent.trpo",
    "ppo_lag": "safepo.single_agent.ppo_lag",
    "trpo_lag": "safepo.single_agent.trpo_lag",
    "cpo": "safepo.single_agent.cpo",
}


def _default_args(
    *,
    task: str,
    seed: int = 0,
    total_steps: int = 1_000_000,
    num_envs: int = 1,
    steps_per_epoch: int = 20000,
    cost_limit: float = 0.0,
    device: str = "cpu",
    device_id: int = 0,
    log_dir: str = "./_training_logs/safepo",
    experiment: str = "specrlbench",
    use_eval: bool = False,
    write_terminal: bool = True,
    **extra: Any,
) -> Namespace:
    """Build argparse-like Namespace matching SafePO ``single_agent_args`` fields."""
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
        # SafePO lag defaults
        lagrangian_multiplier_init=extra.pop("lagrangian_multiplier_init", 0.001),
        lagrangian_multiplier_lr=extra.pop("lagrangian_multiplier_lr", 0.035),
        # misc SafePO flags often present
        parallel=1,
        torch_threads=4,
    )
    for k, v in extra.items():
        setattr(args, k, v)
    return args


def train_with_safepo(
    algo: str,
    env_id: str,
    *,
    seed: int = 0,
    total_steps: int = 1_000_000,
    num_envs: int = 1,
    steps_per_epoch: int = 20000,
    cost_limit: float = 0.0,
    device: str = "cpu",
    device_id: int = 0,
    log_dir: str = "./_training_logs/safepo",
    experiment: str = "specrlbench",
    **extra: Any,
) -> dict[str, Any]:
    """Patch env factory, then run SafePO's ``main(args)`` for ``algo``."""
    algo = resolve_algo(algo)
    if algo == "rnd_ppo":
        raise NotImplementedError(
            "RND-PPO is SpecRLBench-specific; use backends.safepo.rnd_runner "
            "(wraps SafePO PPO buffer/model + local RND module), not a SafePO stock algo."
        )
    if algo not in _SAFEPO_MODULES:
        raise ValueError(f"No SafePO module mapping for {algo!r}")

    from backends.safepo.paths import ensure_specrlbench_paths
    from backends.safepo.torch_compat import patch_linear_lr_verbose

    ensure_specrlbench_paths()
    patch_safepo_env_factory()

    import importlib

    mod = importlib.import_module(_SAFEPO_MODULES[algo])
    patch_linear_lr_verbose(mod)

    args = _default_args(
        task=env_id,
        seed=seed,
        total_steps=total_steps,
        num_envs=num_envs,
        steps_per_epoch=steps_per_epoch,
        cost_limit=cost_limit,
        device=device,
        device_id=device_id,
        log_dir=log_dir,
        experiment=experiment,
        **extra,
    )

    # Match SafePO __main__ log path layout
    relpath = time.strftime("%Y-%m-%d-%H-%M-%S")
    subfolder = "-".join(["seed", str(args.seed).zfill(3)])
    relpath = "-".join([subfolder, relpath])
    args.log_dir = os.path.join(args.log_dir, args.experiment, args.task, algo, relpath)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    # SafePO mains expect (args, cfg_env=None) for mujoco path
    mod.main(args, None)
    return {"log_dir": args.log_dir, "algo": algo, "env_id": env_id}


def train_algo(algo: str, **overrides: Any) -> dict[str, Any]:
    env_id = overrides.pop("env_id", overrides.pop("task", "PointLTL4MASAR1WC-v0"))
    return train_with_safepo(algo, env_id, **overrides)
