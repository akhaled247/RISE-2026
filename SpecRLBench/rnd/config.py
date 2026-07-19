"""RND hyperparameter configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RNDConfig:
    """Configuration for Random Network Distillation.

    Sparse-navigation defaults (SAR L4/L5): beta=0.5 after intrinsic-return
    normalization, feature_dim=256, building/wall-focused obs via ``obs_keys``.
    """

    use_rnd: bool = True
    intrinsic_reward_coef: float = 0.5
    predictor_learning_rate: float = 1e-4
    feature_dim: int = 256
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
    # None => flatten+concat all Box keys for Dict spaces; str => single key
    obs_key: str | None = None
    # Explicit multi-key list (takes precedence over obs_key when set)
    obs_keys: list[str] | None = None
    activation: str = "relu"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RNDConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})
