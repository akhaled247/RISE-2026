"""Shared test helpers for GenZ deploy tests (no MuJoCo)."""
from __future__ import annotations

import torch


def fake_safety_state_dict(
    *,
    feat_dim: int = 64,
    env_net_layers: list[int] | None = None,
    actor_hidden: list[int] | None = None,
    action_dim: int = 2,
) -> dict[str, torch.Tensor]:
    env_net_layers = env_net_layers or [128, 96]
    actor_hidden = actor_hidden or [64, 64, 64]
    embedding_dim = env_net_layers[-1] if env_net_layers else feat_dim

    state: dict[str, torch.Tensor] = {}
    in_dim = feat_dim
    for i, out_dim in enumerate(env_net_layers):
        state[f"env_net.mlp.{i * 2}.weight"] = torch.zeros(out_dim, in_dim)
        state[f"env_net.mlp.{i * 2}.bias"] = torch.zeros(out_dim)
        in_dim = out_dim

    state["actor.enc.0.weight"] = torch.zeros(actor_hidden[0], embedding_dim)
    state["actor.enc.0.bias"] = torch.zeros(actor_hidden[0])
    prev = actor_hidden[0]
    for j, h in enumerate(actor_hidden[1:], start=1):
        state[f"actor.enc.{j * 2}.weight"] = torch.zeros(h, prev)
        state[f"actor.enc.{j * 2}.bias"] = torch.zeros(h)
        prev = h

    state["actor.mu.0.weight"] = torch.zeros(action_dim, prev)
    state["actor.mu.0.bias"] = torch.zeros(action_dim)
    state["actor.log_std"] = torch.zeros(action_dim)
    return state
