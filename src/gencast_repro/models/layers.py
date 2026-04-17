from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_hidden_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers = []
        current = in_dim
        hidden_layers = max(1, num_hidden_layers)
        for _ in range(hidden_layers):
            layers.append(nn.Linear(current, hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            current = hidden_dim
        layers.append(nn.Linear(current, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FourierNoiseEncoder(nn.Module):
    def __init__(
        self,
        base_period: float,
        num_frequencies: int,
        hidden_dim: int,
        out_dim: int,
        apply_log_first: bool = True,
    ) -> None:
        super().__init__()
        self.base_period = base_period
        self.num_frequencies = num_frequencies
        self.apply_log_first = apply_log_first
        self.projection = MLP(
            in_dim=2 * num_frequencies,
            hidden_dim=max(hidden_dim, out_dim),
            out_dim=out_dim,
            num_hidden_layers=2,
        )

    def forward(self, sigmas: torch.Tensor) -> torch.Tensor:
        values = sigmas
        if self.apply_log_first:
            values = torch.log(torch.clamp(values, min=1e-6))
        frequencies = torch.arange(
            1,
            self.num_frequencies + 1,
            device=values.device,
            dtype=values.dtype,
        )
        angles = (2.0 * math.pi / self.base_period) * values[:, None] * frequencies[None, :]
        features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.projection(features)


class AdaLayerNorm(nn.Module):
    def __init__(self, dim: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.modulation = nn.Linear(cond_dim, 2 * dim)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        scale, shift = self.modulation(conditioning).chunk(2, dim=-1)
        normalized = self.norm(x)
        return normalized * (1.0 + scale[:, None, :]) + shift[:, None, :]

