"""Enable GenZ RCO/PPO async vec from RISE without modifying GenZ BaseAlgo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class AsyncTrainConfig:
    n_envs: int
    env_name: str
    curriculum_name: str
    seed: int
    max_steps: int = 2500
    safety: bool = True
    sequence: bool = True
    sar_env_backend: str = "specrl"
    entr_bldg_obs: bool = False
    zone_compat: bool = False
    fast_action_bridge: bool = True


_CFG: AsyncTrainConfig | None = None


def async_train_config() -> AsyncTrainConfig | None:
    return _CFG


def set_async_train_config(cfg: AsyncTrainConfig | None) -> None:
    global _CFG
    _CFG = cfg


def _curriculum_stage_from_env(env: Any) -> int:
    sample = getattr(env, "sample_sequence", None)
    if sample is not None and hasattr(sample, "curriculum"):
        return int(getattr(sample.curriculum, "stage_index", 0))
    return 0


def _async_factory_kwargs(probe_env: Any) -> dict[str, Any]:
    cfg = _CFG
    assert cfg is not None
    return {
        "n_envs": cfg.n_envs,
        "env_name": cfg.env_name,
        "curriculum_name": cfg.curriculum_name,
        "curriculum_stage": _curriculum_stage_from_env(probe_env),
        "seed": cfg.seed,
        "max_steps": cfg.max_steps,
        "safety": cfg.safety,
        "sequence": cfg.sequence,
        "sar_env_backend": cfg.sar_env_backend,
        "entr_bldg_obs": cfg.entr_bldg_obs,
        "zone_compat": cfg.zone_compat,
    }


def rebind_algo_to_async_vec(algo: Any, probe_env: Any) -> None:
    """Replace SyncEnv runner + resize rollout buffers for ``n_envs`` workers."""
    from rise_training.genz_vec.genz_async_vec import build_genz_async_vec

    cfg = async_train_config()
    if cfg is None:
        raise RuntimeError("async train config not set")

    old_env = getattr(algo, "env", None)
    if old_env is not None and hasattr(old_env, "close"):
        try:
            old_env.close()
        except Exception:
            pass

    algo.env = build_genz_async_vec(**_async_factory_kwargs(probe_env))
    n = int(cfg.n_envs)
    algo.num_procs = n
    algo.num_steps = algo.num_steps_per_proc * n
    device = algo.device
    shape = (algo.num_steps_per_proc, n)
    act_shape = shape + tuple(algo.action_space_shape)

    algo.obs = algo.env.reset()
    algo.obss = [None] * shape[0]
    if getattr(algo.model, "recurrent", False):
        algo.memory = torch.zeros(n, algo.model.memory_size, device=device)
        algo.memories = torch.zeros(*shape, algo.model.memory_size, device=device)
    algo.mask = torch.ones(n, device=device)
    algo.masks = torch.zeros(*shape, device=device)
    algo.actions = torch.zeros(*act_shape, device=device)
    algo.values = torch.zeros(*shape, device=device)
    algo.qs = torch.zeros(*shape, device=device)
    algo.rewards = torch.zeros(*shape, device=device)
    algo.advantages = torch.zeros(*shape, device=device)
    algo.log_probs = torch.zeros(*shape, device=device)

    # BaseAlgoLag extras
    if hasattr(algo, "cost_values"):
        algo.cost_values = torch.zeros(*shape, device=device)
        algo.cost_qs = torch.zeros(*shape, device=device)
        algo.returnn = torch.zeros(*shape, device=device)
        algo.costs = torch.zeros(*shape, device=device)
        algo.cost_returnn = torch.zeros(*shape, device=device)
        algo.cost_advantages = torch.zeros(*shape, device=device)
        algo.log_episode_cost_return = torch.zeros(n, device=device)
        algo.log_cost_return = [0] * n

    algo.log_episode_return = torch.zeros(n, device=device)
    algo.log_episode_num_steps = torch.zeros(n, device=device)
    algo.log_return = [0] * n
    algo.log_num_steps = [0] * n
    algo.log_success = [0] * n
    algo.log_violation = [0] * n


def patch_trainer_for_async(trainer_cls: type) -> None:
    """Probe-only ``make_envs`` + rebind algo after RCO/PPO construct."""
    _orig_make = trainer_cls.make_envs
    _orig_train = trainer_cls.train

    def make_envs(self, curriculum_stage: int):
        cfg = async_train_config()
        if cfg is None:
            return _orig_make(self, curriculum_stage)
        import utils

        utils.set_seed(self.args.experiment.seed)
        env = self.make_probe_env(curriculum_stage)
        seed_offset = 100 * self.args.experiment.seed
        env.reset(seed=seed_offset)
        self.text_logger.info(
            f"Async vec: probe on main; {cfg.n_envs} workers (safety_async)."
        )
        self.text_logger.info("Environments loaded.")
        return [env]

    def train(self, log_csv: bool = True, log_wandb: bool = False):
        cfg = async_train_config()
        if cfg is None:
            return _orig_train(self, log_csv=log_csv, log_wandb=log_wandb)

        from rise_training.genz_vec.oom_guard import apply_oom_guardrails

        exp = self.args.experiment
        # Mark async so oom_guard allows num_procs>1
        object.__setattr__(exp, "vec_backend", "safety_async")
        object.__setattr__(exp, "parallel", False)
        algo_cfg = getattr(self.args, "rco", getattr(self.args, "ppo", None))
        apply_oom_guardrails(
            exp,
            algo_config=algo_cfg,
            log=lambda msg: self.text_logger.important_info(msg),
        )
        cfg.n_envs = int(exp.num_procs)

        # Intercept algo construction inside original train by wrapping torch_ac.RCO/PPO
        import torch_ac

        def _wrap_ctor(orig_ctor):
            def ctor(envs, model, device, config, preprocess_obss, parallel=False):
                algo = orig_ctor(
                    envs, model, device, config, preprocess_obss, parallel=parallel,
                )
                rebind_algo_to_async_vec(algo, envs[0])
                return algo

            return ctor

        _rco, _ppo = torch_ac.RCO, torch_ac.PPO
        torch_ac.RCO = _wrap_ctor(_rco)  # type: ignore[misc, assignment]
        torch_ac.PPO = _wrap_ctor(_ppo)  # type: ignore[misc, assignment]
        try:
            return _orig_train(self, log_csv=log_csv, log_wandb=log_wandb)
        finally:
            torch_ac.RCO = _rco  # type: ignore[misc, assignment]
            torch_ac.PPO = _ppo  # type: ignore[misc, assignment]

    trainer_cls.make_envs = make_envs  # type: ignore[method-assign]
    trainer_cls.train = train  # type: ignore[method-assign]
