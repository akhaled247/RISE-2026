"""RND hyperparameter configuration (OpenAI RND defaults + project aliases)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RNDConfig:
    """Configuration for Random Network Distillation.

    OpenAI-faithful names take precedence. Legacy aliases are mapped in
    :meth:`resolve` / :meth:`from_dict`.
    """

    use_rnd: bool = True

    # OpenAI advantage / return coefficients (ppo_agent.py / run_atari.py)
    int_coeff: float = 1.0
    ext_coeff: float = 2.0
    # Legacy alias: if set at construction and int_coeff left default, maps to int_coeff
    intrinsic_reward_coef: float | None = None

    # Discounts (OpenAI: gamma for intrinsic, gamma_ext for extrinsic)
    gamma: float = 0.99
    gamma_ext: float = 0.99
    gamma_int: float | None = None  # alias for gamma when set
    lam: float = 0.95

    # Intrinsic episode boundaries (OpenAI default use_news=0 → never truncate int GAE)
    use_news: bool = False

    # Predictor / target
    rnd_rep_size: int = 512
    feature_dim: int | None = None  # alias for rnd_rep_size
    predictor_net_arch: list[int] = field(default_factory=lambda: [512, 512])
    target_net_arch: list[int] = field(default_factory=lambda: [512])
    activation: str = "relu"
    proportion_of_exp_used_for_predictor_update: float = 1.0
    policy_size: str = "normal"  # small|normal|large → enlargement 1|2|4
    use_gru: bool = False

    # Observation / reward normalization (OpenAI)
    obs_norm: bool = True
    return_norm: bool = True
    clip_obs: float = 5.0
    obs_clip: float | None = None  # alias for clip_obs
    epsilon: float = 1e-8
    rms_epsilon: float = 1e-4
    update_ob_stats_every_step: bool = False
    update_ob_stats_from_random_agent: bool = True
    random_obs_init_steps: int = 128 * 50

    # Optimization (OpenAI run_atari defaults; overridden by PPORND ctor when set)
    predictor_learning_rate: float | None = None  # unused in faithful path (shared Adam)
    max_grad_norm: float = 0.0  # OpenAI run_atari: 0.0 → no clipping
    nminibatches: int | None = None
    updates_per_policy_batch: int = 1  # unused in faithful path

    # Dict-obs compatibility (project-specific; not in OpenAI Atari code)
    obs_key: str | None = None
    obs_keys: list[str] | None = None

    # Deprecated local flags kept for checkpoint roundtrip
    reward_clip: float = 0.0  # OpenAI does not clip normalized intrinsic

    def resolve(self) -> "RNDConfig":
        """Apply aliases and return self."""
        if self.intrinsic_reward_coef is not None:
            self.int_coeff = float(self.intrinsic_reward_coef)
        if self.gamma_int is not None:
            self.gamma = float(self.gamma_int)
        if self.feature_dim is not None:
            self.rnd_rep_size = int(self.feature_dim)
        if self.obs_clip is not None:
            self.clip_obs = float(self.obs_clip)
        return self

    @property
    def enlargement(self) -> int:
        return {"small": 1, "normal": 2, "large": 4}[self.policy_size]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RNDConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        cfg = cls(**{k: v for k, v in data.items() if k in known})
        return cfg.resolve()
