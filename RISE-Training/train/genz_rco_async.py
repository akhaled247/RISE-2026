#!/usr/bin/env python
"""GenZ RCO train with safety_async vec (RISE-owned; GenZ BaseAlgo stays SyncEnv)."""
from __future__ import annotations

import sys
import time
import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rise_training.paths import ensure_genz_paths, ensure_specrlbench_paths

ensure_specrlbench_paths()
ensure_genz_paths()

from rise_training.genz_vec.async_train import (  # noqa: E402
    AsyncTrainConfig,
    patch_trainer_for_async,
    set_async_train_config,
)
from rise_training.genz_vec.oom_guard import compensate_steps_per_process, rollout_ref_procs  # noqa: E402


def main() -> None:
    # Checkpoints / experiments live under GenZ-LTL (GenZ ModelStore layout).
    genz_root = Path(__file__).resolve().parents[2] / "GenZ-LTL"
    if genz_root.is_dir():
        import os

        os.chdir(genz_root)

    # Import GenZ trainer only after path bootstrap.
    from train.train_rco import Trainer, parse_arguments  # noqa: WPS433

    args = parse_arguments()
    n_procs = int(args.experiment.num_procs)
    # Preserve frames/update vs historical 24-proc recipe when user lowers procs.
    ref = rollout_ref_procs()
    if n_procs != ref:
        old = int(args.rco.steps_per_process)
        args.rco.steps_per_process = compensate_steps_per_process(ref, n_procs, old)

    set_async_train_config(
        AsyncTrainConfig(
            n_envs=n_procs,
            env_name=args.experiment.env,
            curriculum_name=args.curriculum,
            seed=int(args.experiment.seed),
            sar_env_backend=str(args.experiment.sar_env_backend),
            entr_bldg_obs=False if args.zone_compat else bool(args.entr_bldg_obs),
            zone_compat=bool(args.zone_compat),
            fast_action_bridge=True,
            safety=True,
            sequence=True,
        )
    )
    patch_trainer_for_async(Trainer)

    trainer = Trainer(args)
    start = time.time()
    trainer.train(log_csv=args.log_csv, log_wandb=args.log_wandb)
    print(f"Training took {datetime.timedelta(seconds=int(time.time() - start))}.")


if __name__ == "__main__":
    main()
