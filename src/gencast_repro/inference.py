from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from .data.dataset import build_forcing_array
from .data.normalization import WeatherStatistics
from .data.variables import ChannelLayout
from .models.diffusion import EDMSampler
from .models.gencast import GenCastDenoiser


@dataclass
class RolloutResult:
    normalized_residuals: torch.Tensor
    states: torch.Tensor
    target_times: List[str]


@torch.no_grad()
def sample_next_residual(
    model: GenCastDenoiser,
    sampler: EDMSampler,
    conditioning_states: torch.Tensor,
    forcings: torch.Tensor,
    latitudes: torch.Tensor,
) -> torch.Tensor:
    batch_size, _, channels, height, width = conditioning_states.shape

    def denoiser(noisy_target: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
        return model(conditioning_states, forcings, noisy_target, sigmas)

    return sampler.sample(
        denoiser=denoiser,
        shape=torch.Size([batch_size, channels, height, width]),
        latitudes=latitudes,
        device=conditioning_states.device,
    )


@torch.no_grad()
def autoregressive_rollout(
    model: GenCastDenoiser,
    sampler: EDMSampler,
    conditioning_states: torch.Tensor,
    initial_current_state: torch.Tensor,
    latitudes: torch.Tensor,
    longitudes: torch.Tensor,
    layout: ChannelLayout,
    statistics: WeatherStatistics,
    static_fields: Dict[str, np.ndarray],
    start_time: str,
    steps: int,
    step_hours: int,
    precipitation_mask: torch.Tensor,
) -> RolloutResult:
    device = conditioning_states.device
    current_normalized = conditioning_states.clone()
    current_physical = initial_current_state.clone()
    all_residuals = []
    all_states = []
    all_target_times: List[str] = []

    timestamp = np.datetime64(start_time)
    delta = np.timedelta64(step_hours, "h")

    for step in range(steps):
        target_timestamp = timestamp + delta
        forcing_np = build_forcing_array(
            target_timestamp,
            latitudes.cpu().numpy(),
            longitudes.cpu().numpy(),
            static_fields,
            layout,
        )
        forcing = torch.from_numpy(forcing_np).to(device=device, dtype=current_normalized.dtype)
        forcing = statistics.normalize_forcings_torch(forcing[None, ...], channel_dim=1)
        sampled_residual = sample_next_residual(
            model=model,
            sampler=sampler,
            conditioning_states=current_normalized,
            forcings=forcing,
            latitudes=latitudes,
        )
        residual_physical = statistics.denormalize_residuals_torch(
            sampled_residual, channel_dim=1
        )
        next_state = current_physical + residual_physical
        next_state[:, precipitation_mask, :, :] = residual_physical[
            :, precipitation_mask, :, :
        ]
        next_state_normalized = statistics.normalize_states_torch(next_state, channel_dim=1)
        current_normalized = torch.stack(
            [current_normalized[:, 1], next_state_normalized], dim=1
        )
        current_physical = next_state
        all_residuals.append(sampled_residual)
        all_states.append(next_state)
        all_target_times.append(str(target_timestamp))
        timestamp = target_timestamp

    return RolloutResult(
        normalized_residuals=torch.stack(all_residuals, dim=1),
        states=torch.stack(all_states, dim=1),
        target_times=all_target_times,
    )
