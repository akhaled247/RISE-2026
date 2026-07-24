"""RND target (frozen) and predictor networks."""

from __future__ import annotations

import torch as th
from torch import nn


def _activation_from_name(name: str) -> type[nn.Module]:
    name = name.lower()
    if name == "relu":
        return nn.ReLU
    if name == "tanh":
        return nn.Tanh
    if name == "elu":
        return nn.ELU
    raise ValueError(f"Unsupported activation: {name}")


def _mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: list[int],
    activation_fn: type[nn.Module],
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(activation_fn())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


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
        self.net = _mlp(input_dim, feature_dim, net_arch, act)

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
        pred = self.predictor(x)
        with th.no_grad():
            tgt = self.target(x)
        return pred, tgt

    def prediction_error(self, x: th.Tensor) -> th.Tensor:
        pred, tgt = self.forward(x)
        return 0.5 * ((pred - tgt) ** 2).sum(dim=-1)
