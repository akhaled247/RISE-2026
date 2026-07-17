"""OpenAI-faithful RND networks and dual-value actor-critic policy (PyTorch)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn
from torch.distributions import Categorical, Normal

from rnd.obs_adapter import RNDObsAdapter


def ortho_init_(tensor: th.Tensor, scale: float = 1.0) -> None:
    """Orthogonal init matching OpenAI ``utils.ortho_init``."""
    if tensor.ndimension() < 2:
        nn.init.constant_(tensor, 0.0)
        return
    flat_shape = (tensor.shape[0], int(np.prod(tensor.shape[1:])))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(tensor.shape)
    with th.no_grad():
        tensor.copy_(th.as_tensor(scale * q[: tensor.shape[0]], dtype=tensor.dtype))


def _init_linear(module: nn.Linear, scale: float = 1.0, bias: float = 0.0) -> None:
    ortho_init_(module.weight, scale)
    nn.init.constant_(module.bias, bias)


def _init_conv(module: nn.Conv2d, scale: float = np.sqrt(2)) -> None:
    # OpenAI ortho_init for 4D: flatten (prod(shape[:-1]), shape[-1])
    w = module.weight.data
    shape = tuple(w.shape)  # (out, in, kH, kW) in PyTorch — OpenAI TF is (kH,kW,in,out)
    # Match OpenAI NHWC weight layout ortho by reshaping equivalently
    a = np.random.normal(0.0, 1.0, (int(np.prod(shape[1:])), shape[0]))
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == a.shape else v
    q = q.reshape(shape[1:] + (shape[0],)).transpose(3, 2, 0, 1)  # out,in,kH,kW approx
    # Simpler: use torch orthogonal on flattened out channels
    nn.init.orthogonal_(module.weight, gain=scale)
    nn.init.constant_(module.bias, 0.0)


class OpenAIRNDTarget(nn.Module):
    """Frozen random target. Image: OpenAI CNN; vector: MLP to ``rep_size``."""

    def __init__(
        self,
        input_dim: int | None = None,
        image_shape: tuple[int, int, int] | None = None,
        rep_size: int = 512,
        enlargement: int = 2,
    ) -> None:
        super().__init__()
        self.image_shape = image_shape
        self.rep_size = rep_size
        convfeat = 16 * enlargement
        if image_shape is not None:
            c, h, w = image_shape
            self.conv = nn.Sequential(
                nn.Conv2d(c, convfeat, kernel_size=8, stride=4),
                nn.LeakyReLU(),
                nn.Conv2d(convfeat, convfeat * 2, kernel_size=4, stride=2),
                nn.LeakyReLU(),
                nn.Conv2d(convfeat * 2, convfeat * 2, kernel_size=3, stride=1),
                nn.LeakyReLU(),
            )
            with th.no_grad():
                dummy = th.zeros(1, c, h, w)
                n_flat = int(np.prod(self.conv(dummy).shape[1:]))
            self.fc = nn.Linear(n_flat, rep_size)
            for m in self.conv:
                if isinstance(m, nn.Conv2d):
                    _init_conv(m, np.sqrt(2))
            _init_linear(self.fc, np.sqrt(2))
            self.mlp = None
        else:
            assert input_dim is not None
            self.conv = None
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, rep_size),
                nn.ReLU(),
                nn.Linear(rep_size, rep_size),
            )
            for m in self.mlp:
                if isinstance(m, nn.Linear):
                    _init_linear(m, np.sqrt(2))
            self.fc = None

    def forward(self, x: th.Tensor) -> th.Tensor:
        if self.conv is not None:
            # x: (B, C, H, W) or (B, H, W, C)
            if x.ndim == 4 and x.shape[-1] in (1, 3, 4) and x.shape[1] > 4:
                x = x.permute(0, 3, 1, 2).contiguous()
            h = self.conv(x)
            h = h.reshape(h.shape[0], -1)
            return self.fc(h)  # type: ignore[misc]
        return self.mlp(x)  # type: ignore[misc]


class OpenAIRNDPredictor(nn.Module):
    """Trainable predictor matching OpenAI ``define_self_prediction_rew``."""

    def __init__(
        self,
        input_dim: int | None = None,
        image_shape: tuple[int, int, int] | None = None,
        rep_size: int = 512,
        enlargement: int = 2,
    ) -> None:
        super().__init__()
        self.image_shape = image_shape
        convfeat = 16 * enlargement
        hid = 256 * enlargement
        if image_shape is not None:
            c, h, w = image_shape
            self.conv = nn.Sequential(
                nn.Conv2d(c, convfeat, kernel_size=8, stride=4),
                nn.LeakyReLU(),
                nn.Conv2d(convfeat, convfeat * 2, kernel_size=4, stride=2),
                nn.LeakyReLU(),
                nn.Conv2d(convfeat * 2, convfeat * 2, kernel_size=3, stride=1),
                nn.LeakyReLU(),
            )
            with th.no_grad():
                dummy = th.zeros(1, c, h, w)
                n_flat = int(np.prod(self.conv(dummy).shape[1:]))
            self.head = nn.Sequential(
                nn.Linear(n_flat, hid),
                nn.ReLU(),
                nn.Linear(hid, hid),
                nn.ReLU(),
                nn.Linear(hid, rep_size),
            )
            for m in self.conv:
                if isinstance(m, nn.Conv2d):
                    _init_conv(m, np.sqrt(2))
            for m in self.head:
                if isinstance(m, nn.Linear):
                    _init_linear(m, np.sqrt(2))
            self.mlp = None
        else:
            assert input_dim is not None
            self.conv = None
            self.head = None
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hid),
                nn.ReLU(),
                nn.Linear(hid, hid),
                nn.ReLU(),
                nn.Linear(hid, rep_size),
            )
            for m in self.mlp:
                if isinstance(m, nn.Linear):
                    _init_linear(m, np.sqrt(2))

    def forward(self, x: th.Tensor) -> th.Tensor:
        if self.conv is not None:
            if x.ndim == 4 and x.shape[-1] in (1, 3, 4) and x.shape[1] > 4:
                x = x.permute(0, 3, 1, 2).contiguous()
            h = self.conv(x)
            h = h.reshape(h.shape[0], -1)
            return self.head(h)  # type: ignore[misc]
        return self.mlp(x)  # type: ignore[misc]


class RNDModel(nn.Module):
    """Fixed random target + trainable predictor (OpenAI formulas)."""

    def __init__(
        self,
        input_dim: int | None = None,
        image_shape: tuple[int, int, int] | None = None,
        rep_size: int = 512,
        enlargement: int = 2,
    ) -> None:
        super().__init__()
        self.target = OpenAIRNDTarget(input_dim, image_shape, rep_size, enlargement)
        self.predictor = OpenAIRNDPredictor(input_dim, image_shape, rep_size, enlargement)
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
        """Per-sample OpenAI intrinsic reward: mean over features of squared error."""
        pred, tgt = self.forward(x)
        return th.mean(th.square(tgt.detach() - pred), dim=-1)

    def aux_loss(
        self,
        x: th.Tensor,
        proportion: float = 1.0,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Masked predictor loss (OpenAI ``aux_loss``).

        Returns ``(loss, feat_var, max_feat)``.
        """
        pred = self.predictor(x)
        with th.no_grad():
            tgt = self.target(x)
        per = th.mean(th.square(tgt.detach() - pred), dim=-1)
        if proportion >= 1.0:
            loss = per.mean()
        else:
            mask = (th.rand_like(per) < proportion).float()
            loss = (mask * per).sum() / th.clamp(mask.sum(), min=1.0)
        feat_var = th.mean(th.var(tgt, dim=0, unbiased=False))
        max_feat = th.max(th.abs(tgt))
        return loss, feat_var, max_feat


class DualValuePolicy(nn.Module):
    """Actor-critic with separate intrinsic / extrinsic value heads.

    Supports Box (vector or image) and Dict observations via :class:`RNDObsAdapter`.
    Image path uses OpenAI Nature CNN + dual VF; vector/Dict uses MLP.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        adapter: RNDObsAdapter,
        features_dim: int = 256,
        net_arch: list[int] | None = None,
        log_std_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.adapter = adapter
        net_arch = net_arch or [256, 256]
        self.is_discrete = isinstance(action_space, spaces.Discrete)
        self.is_image = (
            isinstance(observation_space, spaces.Box)
            and len(observation_space.shape) == 3
        )

        if self.is_image:
            c, h, w = self._chw(observation_space.shape)
            self.features = nn.Sequential(
                nn.Conv2d(c, 32, 8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 4, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            with th.no_grad():
                n_flat = int(np.prod(self.features(th.zeros(1, c, h, w)).shape[1:]))
            self.fc = nn.Sequential(
                nn.Linear(n_flat, 512),
                nn.ReLU(),
            )
            feat_out = 512
            for m in self.features:
                if isinstance(m, nn.Conv2d):
                    nn.init.orthogonal_(m.weight, np.sqrt(2))
                    nn.init.constant_(m.bias, 0.0)
            for m in self.fc:
                if isinstance(m, nn.Linear):
                    _init_linear(m, np.sqrt(2))
        else:
            layers: list[nn.Module] = []
            last = adapter.input_dim
            for h in net_arch:
                layers.append(nn.Linear(last, h))
                layers.append(nn.Tanh())
                last = h
            self.features = nn.Sequential(*layers)
            self.fc = nn.Identity()
            feat_out = last
            for m in self.features:
                if isinstance(m, nn.Linear):
                    _init_linear(m, np.sqrt(2))

        self.vf_int = nn.Linear(feat_out, 1)
        self.vf_ext = nn.Linear(feat_out, 1)
        _init_linear(self.vf_int, 0.01)
        _init_linear(self.vf_ext, 0.01)

        if self.is_discrete:
            self.action_net = nn.Linear(feat_out, int(action_space.n))
            _init_linear(self.action_net, 0.01)
            self.log_std = None
        else:
            n_act = int(np.prod(action_space.shape))
            self.action_net = nn.Linear(feat_out, n_act)
            _init_linear(self.action_net, 0.01)
            self.log_std = nn.Parameter(th.ones(n_act) * log_std_init)

    @staticmethod
    def _chw(shape: tuple[int, ...]) -> tuple[int, int, int]:
        if shape[0] in (1, 3, 4) and shape[-1] not in (1, 3, 4):
            return shape[0], shape[1], shape[2]
        return shape[2], shape[0], shape[1]

    def _encode(self, obs: Any) -> th.Tensor:
        if self.is_image:
            x = th.as_tensor(obs, dtype=th.float32, device=self.vf_int.weight.device)
            if x.ndim == 3:
                x = x.unsqueeze(0)
            if x.dtype != th.float32:
                x = x.float()
            if float(x.max()) > 1.5:
                x = x / 255.0
            if x.shape[-1] in (1, 3, 4) and x.shape[1] > 4:
                x = x.permute(0, 3, 1, 2).contiguous()
            return self.fc(self.features(x))
        # Accept raw env obs (dict/box) or already-flattened (batch, input_dim)
        if isinstance(obs, np.ndarray) and obs.ndim == 2 and obs.shape[-1] == self.adapter.input_dim:
            flat = obs.astype(np.float32, copy=False)
        elif isinstance(obs, th.Tensor) and obs.ndim == 2 and obs.shape[-1] == self.adapter.input_dim:
            return self.fc(self.features(obs.to(device=self.vf_int.weight.device, dtype=th.float32)))
        else:
            flat = self.adapter.to_numpy(obs)
        x = th.as_tensor(flat, dtype=th.float32, device=self.vf_int.weight.device)
        return self.fc(self.features(x))

    def forward(
        self,
        obs: Any,
        deterministic: bool = False,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """Return ``actions, vpred_int, vpred_ext, neglogp, entropy``."""
        latent = self._encode(obs)
        v_int = self.vf_int(latent).squeeze(-1)
        v_ext = self.vf_ext(latent).squeeze(-1)
        if self.is_discrete:
            logits = self.action_net(latent)
            dist = Categorical(logits=logits)
            actions = th.argmax(logits, dim=-1) if deterministic else dist.sample()
            neglogp = -dist.log_prob(actions)
            entropy = dist.entropy()
        else:
            mean = self.action_net(latent)
            std = th.exp(self.log_std)
            dist = Normal(mean, std)
            actions = mean if deterministic else dist.rsample()
            neglogp = -dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        return actions, v_int, v_ext, neglogp, entropy

    def evaluate_actions(
        self,
        obs: Any,
        actions: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """Return ``neglogp, entropy, vpred_int, vpred_ext``."""
        latent = self._encode(obs)
        v_int = self.vf_int(latent).squeeze(-1)
        v_ext = self.vf_ext(latent).squeeze(-1)
        if self.is_discrete:
            logits = self.action_net(latent)
            dist = Categorical(logits=logits)
            neglogp = -dist.log_prob(actions.long().view(-1))
            entropy = dist.entropy()
        else:
            mean = self.action_net(latent)
            std = th.exp(self.log_std)
            dist = Normal(mean, std)
            if actions.ndim == 1:
                actions = actions.unsqueeze(-1)
            neglogp = -dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        return neglogp, entropy, v_int, v_ext

    def predict_values(self, obs: Any) -> tuple[th.Tensor, th.Tensor]:
        latent = self._encode(obs)
        return self.vf_int(latent).squeeze(-1), self.vf_ext(latent).squeeze(-1)
