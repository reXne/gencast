from __future__ import annotations

import torch

from .losses import latitude_weights


def rmse(prediction: torch.Tensor, target: torch.Tensor, latitudes: torch.Tensor) -> torch.Tensor:
    weights = latitude_weights(latitudes).view(1, 1, -1, 1)
    return torch.sqrt((((prediction - target) ** 2) * weights).mean())


def bias(prediction: torch.Tensor, target: torch.Tensor, latitudes: torch.Tensor) -> torch.Tensor:
    weights = latitude_weights(latitudes).view(1, 1, -1, 1)
    return ((prediction - target) * weights).mean()


def crps_ensemble(
    ensemble: torch.Tensor,
    target: torch.Tensor,
    latitudes: torch.Tensor,
) -> torch.Tensor:
    weights = latitude_weights(latitudes).view(1, 1, 1, -1, 1)
    obs_term = (ensemble - target[:, None, :, :, :]).abs().mean(dim=1)
    pairwise = (ensemble[:, :, None] - ensemble[:, None, :]).abs().mean(dim=(1, 2))
    crps = obs_term - 0.5 * pairwise
    return (crps * weights[:, 0]).mean()


def spread_skill_ratio(
    ensemble: torch.Tensor,
    target: torch.Tensor,
    latitudes: torch.Tensor,
) -> torch.Tensor:
    mean = ensemble.mean(dim=1)
    spread = ensemble.std(dim=1)
    skill = ((mean - target) ** 2).mean().sqrt()
    spread_value = (spread**2).mean().sqrt()
    return spread_value / skill.clamp_min(1e-6)

