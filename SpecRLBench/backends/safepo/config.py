"""SafePO backend config for SpecRLBench."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SafePOTrainConfig:
    """Shared training knobs for SpecRLBench SafePO runners."""

    env_id: str = "PointLTL4MASAR1WC-v0"
    algo: str = "ppo"
    seed: int = 0
    total_steps: int = 1_000_000
    steps_per_epoch: int = 16_384  # n_envs * local_steps; default 8*2048
    num_envs: int = 8
    parallel: bool = True  # SafetyAsyncVectorEnv when num_envs > 1
    gamma: float = 0.99
    lam: float = 0.97
    lam_c: float = 0.97
    target_kl: float = 0.05
    batch_size: int = 256
    learning_iters: int = 10
    hidden_sizes: list[int] = field(default_factory=lambda: [64, 64])
    max_grad_norm: float = 40.0
    actor_lr: float = 5e-5
    critic_lr: float = 1e-3
    clip_ratio: float = 0.2
    lr_end_factor: float = 1.0
    ent_coef: float = 0.0
    # Constrained
    cost_limit: float = 0.0
    lagrangian_multiplier_init: float = 1.0
    lagrangian_multiplier_lr: float = 1e-2
    # Env
    normalize_obs: bool = True
    clip_obs: float = 10.0
    device: str = "cpu"
    log_dir: str = "./_training_logs/safepo"
    experiment: str = "specrlbench"
    use_eval: bool = False
    eval_episodes: int = 10
    save_model_freq: int = 10  # epochs; SafePO also always saves last epoch
    # RND
    rnd_coef: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ALGO_DEFAULTS: dict[str, dict[str, Any]] = {
    "ppo": {},
    "trpo": {"learning_iters": 1, "target_kl": 0.02},
    "ppo_lag": {},
    "trpo_lag": {"learning_iters": 1, "target_kl": 0.02},
    "cpo": {"learning_iters": 1, "target_kl": 0.01},
    "rnd_ppo": {},
}
