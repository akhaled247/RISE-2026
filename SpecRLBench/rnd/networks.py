"""RND target (frozen) and predictor networks."""

from __future__ import annotations

import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import create_mlp


def _activation_from_name(name: str) -> type[nn.Module]:
    name = name.lower()
    if name == "relu":
        return nn.ReLU
    if name == "tanh":
        return nn.Tanh
    if name == "elu":
        return nn.ELU
    raise ValueError(f"Unsupported activation: {name}")


class RNDNetwork(nn.Module):
    """MLP that maps flattened observations to feature embeddings."""

    def __init__(
        self,
        input_dim: int,
        feature_dim: int,
        net_arch: list[int],
        activation: str = "relu",
    ) -> None:
        super().__init__()
        act = _activation_from_name(activation)
        layers = create_mlp(input_dim, feature_dim, net_arch, activation_fn=act)
        self.net = nn.Sequential(*layers)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.net(x)


class RNDModel(nn.Module):
    """Fixed random target + trainable predictor."""

    def __init__(
        self,
        input_dim: int,
        feature_dim: int = 128,
        target_net_arch: list[int] | None = None,
        predictor_net_arch: list[int] | None = None,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        target_net_arch = target_net_arch or [256, 256]
        predictor_net_arch = predictor_net_arch or [512, 512, 512]
        self.target = RNDNetwork(input_dim, feature_dim, target_net_arch, activation)
        self.predictor = RNDNetwork(input_dim, feature_dim, predictor_net_arch, activation)
        self.freeze_target()

    def freeze_target(self) -> None:
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.target.eval()

    def forward(self, x: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """Return (predictor_features, target_features)."""
        pred = self.predictor(x)
        with th.no_grad():
            tgt = self.target(x)
        return pred, tgt

    def prediction_error(self, x: th.Tensor) -> th.Tensor:
        """Per-sample half squared L2 error, shape ``(batch,)``."""
        pred, tgt = self.forward(x)
        return 0.5 * ((pred - tgt) ** 2).sum(dim=-1)

    def predictor_loss(self, x: th.Tensor) -> th.Tensor:
        return self.prediction_error(x).mean()
