"""RND hyperparameter configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RNDConfig:
    """Configuration for Random Network Distillation.

    Defaults follow Burda et al. (2018) where practical, with SB3-friendly
    values for vector observations used in SpecRLBench.
    """

    use_rnd: bool = True
    intrinsic_reward_coef: float = 0.01
    predictor_learning_rate: float = 1e-4
    feature_dim: int = 128
    target_net_arch: list[int] = field(default_factory=lambda: [256, 256])
    predictor_net_arch: list[int] = field(default_factory=lambda: [512, 512, 512])
    gamma_int: float = 0.99
    obs_norm: bool = True
    return_norm: bool = True
    reward_clip: float = 5.0
    obs_clip: float = 5.0
    epsilon: float = 1e-8
    rms_epsilon: float = 1e-4
    max_grad_norm: float = 0.5
    updates_per_policy_batch: int = 1
    # None => flatten+concat all Box keys for Dict spaces; str => single key
    obs_key: str | None = None
    activation: str = "relu"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RNDConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})
