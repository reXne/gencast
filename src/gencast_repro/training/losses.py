from __future__ import annotations

import numpy as np
import torch


def latitude_weights(latitudes: torch.Tensor) -> torch.Tensor:
    if latitudes.ndim > 1:
        latitudes = latitudes[0]
    weights = torch.cos(torch.deg2rad(latitudes)).clamp_min(1e-6)
    return weights / weights.mean()


def weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    latitudes: torch.Tensor,
    channel_weights: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    lat_weight = latitude_weights(latitudes).view(1, 1, -1, 1)
    channel_weight = channel_weights.view(1, -1, 1, 1)
    sample_weight = sample_weights.view(-1, 1, 1, 1)
    loss = ((prediction - target) ** 2) * lat_weight * channel_weight * sample_weight
    return loss.mean()
