"""Actor-critic networks matching OpenAI ``safe_rl/pg/network.py``.

Scopes mirror OpenAI TF graphs:
* ``pi``  — policy (Gaussian or Categorical)
* ``vf``  — reward value function
* ``vc``  — cost value function
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces

from ppo_lagrangian.utils import EPS

LOG_STD_INIT = -0.5


def mlp(
    sizes: list[int],
    activation: type[nn.Module] = nn.Tanh,
    output_activation: type[nn.Module] | None = None,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        act: type[nn.Module] | None = activation if i < len(sizes) - 2 else output_activation
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if act is not None:
            layers.append(act())
    return nn.Sequential(*layers)


def obs_dim_from_space(observation_space: spaces.Space) -> int:
    if isinstance(observation_space, spaces.Box):
        return int(np.prod(observation_space.shape))
    if isinstance(observation_space, spaces.Dict):
        return int(sum(np.prod(observation_space.spaces[k].shape) for k in sorted(observation_space.spaces.keys())))
    raise NotImplementedError(f"Unsupported observation space: {type(observation_space)}")


def obs_to_tensor(
    obs: np.ndarray | dict[str, np.ndarray],
    device: th.device,
    observation_space: spaces.Space,
) -> th.Tensor:
    """Convert a single or batched observation to a flat float tensor."""
    if isinstance(observation_space, spaces.Dict):
        assert isinstance(obs, dict)
        keys = sorted(observation_space.spaces.keys())
        # Detect batch: first key's array has leading batch dim matching env batch
        sample = np.asarray(obs[keys[0]])
        space_shape = observation_space.spaces[keys[0]].shape
        if sample.shape == space_shape:
            # Single observation
            parts = [np.asarray(obs[k], dtype=np.float32).reshape(-1) for k in keys]
            flat = np.concatenate(parts, axis=0)[None, :]
        else:
            batch = sample.shape[0]
            parts = [np.asarray(obs[k], dtype=np.float32).reshape(batch, -1) for k in keys]
            flat = np.concatenate(parts, axis=-1)
        return th.as_tensor(flat, dtype=th.float32, device=device)

    arr = np.asarray(obs, dtype=np.float32)
    if arr.shape == observation_space.shape:  # type: ignore[union-attr]
        arr = arr.reshape(1, -1)
    else:
        arr = arr.reshape(arr.shape[0], -1)
    return th.as_tensor(arr, dtype=th.float32, device=device)


# ---------------------------------------------------------------------------
# Distributions (OpenAI gaussian_* / categorical_*)
# ---------------------------------------------------------------------------

def gaussian_likelihood(x: th.Tensor, mu: th.Tensor, log_std: th.Tensor) -> th.Tensor:
    pre_sum = -0.5 * (((x - mu) / (th.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))
    return th.sum(pre_sum, dim=-1)


def gaussian_kl(mu0: th.Tensor, log_std0: th.Tensor, mu1: th.Tensor, log_std1: th.Tensor) -> th.Tensor:
    """Mean KL between two batches of diagonal Gaussians (OpenAI formula)."""
    var0, var1 = th.exp(2 * log_std0), th.exp(2 * log_std1)
    pre_sum = 0.5 * (((mu1 - mu0) ** 2 + var0) / (var1 + EPS) - 1) + log_std1 - log_std0
    all_kls = th.sum(pre_sum, dim=-1)
    return th.mean(all_kls)


def gaussian_entropy(log_std: th.Tensor) -> th.Tensor:
    """Average entropy over a batch of diagonal Gaussians."""
    pre_sum = log_std + 0.5 * np.log(2 * np.pi * np.e)
    all_ents = th.sum(pre_sum, dim=-1)
    return th.mean(all_ents)


def categorical_kl(logp0: th.Tensor, logp1: th.Tensor) -> th.Tensor:
    """Mean KL between two categorical batches. OpenAI: exp(logp1)*(logp1-logp0)."""
    all_kls = th.sum(th.exp(logp1) * (logp1 - logp0), dim=-1)
    return th.mean(all_kls)


def categorical_entropy(logp: th.Tensor) -> th.Tensor:
    all_ents = -th.sum(logp * th.exp(logp), dim=-1)
    return th.mean(all_ents)


# ---------------------------------------------------------------------------
# Actor-Critic
# ---------------------------------------------------------------------------

class MLPActorCritic(nn.Module):
    """Three-headed actor-critic: pi / vf / vc (OpenAI mlp_actor_critic)."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        hidden_sizes: tuple[int, ...] = (64, 64),
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_dim = obs_dim_from_space(observation_space)
        self.hidden_sizes = list(hidden_sizes)
        self.is_discrete = isinstance(action_space, spaces.Discrete)
        self.is_box = isinstance(action_space, spaces.Box)

        if self.is_box:
            self.act_dim = int(np.prod(action_space.shape))
            self.pi_net = mlp([self.obs_dim] + self.hidden_sizes + [self.act_dim], activation, None)
            self.log_std = nn.Parameter(LOG_STD_INIT * th.ones(self.act_dim, dtype=th.float32))
            self.pi_info_shapes = {"mu": [self.act_dim], "log_std": [self.act_dim]}
        elif self.is_discrete:
            self.act_dim = int(action_space.n)
            self.pi_net = mlp([self.obs_dim] + self.hidden_sizes + [self.act_dim], activation, None)
            self.log_std = None  # type: ignore[assignment]
            self.pi_info_shapes = {"logp_all": [self.act_dim]}
        else:
            raise NotImplementedError(f"Unsupported action space: {type(action_space)}")

        self.vf = mlp([self.obs_dim] + self.hidden_sizes + [1], activation, None)
        self.vc = mlp([self.obs_dim] + self.hidden_sizes + [1], activation, None)

    # -- forward helpers ----------------------------------------------------

    def _features(self, obs: th.Tensor) -> th.Tensor:
        return obs

    def _mu(self, obs: th.Tensor) -> th.Tensor:
        return self.pi_net(self._features(obs))

    def _logits(self, obs: th.Tensor) -> th.Tensor:
        return self.pi_net(self._features(obs))

    def value(self, obs: th.Tensor) -> th.Tensor:
        return self.vf(self._features(obs)).squeeze(-1)

    def cost_value(self, obs: th.Tensor) -> th.Tensor:
        return self.vc(self._features(obs)).squeeze(-1)

    def step(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, dict[str, th.Tensor]]:
        """Sample action and return (a, v, vc, logp_pi, pi_info)."""
        with th.no_grad():
            v = self.value(obs)
            vc = self.cost_value(obs)
            if self.is_box:
                mu = self._mu(obs)
                log_std = self.log_std.expand_as(mu)
                std = th.exp(log_std)
                if deterministic:
                    pi = mu
                else:
                    pi = mu + th.randn_like(mu) * std
                logp_pi = gaussian_likelihood(pi, mu, log_std)
                pi_info = {"mu": mu.detach(), "log_std": log_std.detach()}
            else:
                logits = self._logits(obs)
                logp_all = th.log_softmax(logits, dim=-1)
                if deterministic:
                    pi = th.argmax(logits, dim=-1)
                else:
                    pi = th.multinomial(th.softmax(logits, dim=-1), num_samples=1).squeeze(-1)
                logp_pi = logp_all.gather(-1, pi.unsqueeze(-1)).squeeze(-1)
                pi_info = {"logp_all": logp_all.detach()}
        return pi.detach(), v.detach(), vc.detach(), logp_pi.detach(), pi_info

    def evaluate_actions(
        self,
        obs: th.Tensor,
        act: th.Tensor,
        old_pi_info: dict[str, th.Tensor],
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """Return logp, entropy, d_kl, v, vc for a batch (OpenAI graph symbols)."""
        v = self.value(obs)
        vc = self.cost_value(obs)

        if self.is_box:
            mu = self._mu(obs)
            log_std = self.log_std.expand_as(mu)
            logp = gaussian_likelihood(act, mu, log_std)
            ent = gaussian_entropy(log_std)
            # OpenAI: d_kl = gaussian_kl(mu, log_std, old_mu, old_log_std)
            # Note argument order: (mu0, log_std0, mu1, log_std1) where 0=new, 1=old
            # Wait — OpenAI network.py:
            #   d_kl = gaussian_kl(mu, log_std, old_mu_ph, old_log_std_ph)
            # and gaussian_kl(mu0, log_std0, mu1, log_std1) with
            #   var0=exp(2*log_std0), var1=exp(2*log_std1)
            #   0.5*((mu1-mu0)^2 + var0)/(var1+EPS) - 1 + log_std1 - log_std0
            # So mu0=current, mu1=old. That is KL(current || old)? Let's check:
            # Standard KL(N0||N1) = log(s1/s0) + (s0^2 + (m0-m1)^2)/(2 s1^2) - 1/2
            # OpenAI formula uses var0 in numerator and log_std1 - log_std0 → KL(pi_old || pi_new)?
            # Actually their Spinning Up code historically computed approx KL as KL(old||new)
            # for early stopping. Looking at the formula with mu0=mu (new), mu1=old:
            #   0.5 * ((mu_old - mu_new)^2 + var_new) / var_old + log_std_old - log_std_new - 0.5
            # That is KL(new || old). Spinning Up early-stop uses mean KL(old||new) via
            # approx from logp ratios often. We'll match OpenAI call signature exactly:
            d_kl = gaussian_kl(mu, log_std, old_pi_info["mu"], old_pi_info["log_std"])
        else:
            logits = self._logits(obs)
            logp_all = th.log_softmax(logits, dim=-1)
            if act.dim() > 1:
                act = act.squeeze(-1)
            act_long = act.long()
            logp = logp_all.gather(-1, act_long.unsqueeze(-1)).squeeze(-1)
            ent = categorical_entropy(logp_all)
            # OpenAI: d_kl = categorical_kl(logp_all, old_logp_all)
            # categorical_kl(logp0, logp1) = sum(exp(logp1)*(logp1-logp0))
            # with logp0=current, logp1=old → KL(old || current)
            d_kl = categorical_kl(logp_all, old_pi_info["logp_all"])

        return logp, ent, d_kl, v, vc

    def pi_parameters(self) -> list[nn.Parameter]:
        params = list(self.pi_net.parameters())
        if self.log_std is not None:
            params.append(self.log_std)
        return params

    def vf_parameters(self) -> list[nn.Parameter]:
        return list(self.vf.parameters()) + list(self.vc.parameters())
