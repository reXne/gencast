from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch


def karras_schedule(
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    device: torch.device,
) -> torch.Tensor:
    ramp = torch.linspace(0.0, 1.0, num_steps, device=device)
    min_inv = sigma_min ** (1.0 / rho)
    max_inv = sigma_max ** (1.0 / rho)
    sigmas = (max_inv + ramp * (min_inv - max_inv)) ** rho
    return torch.cat([sigmas, torch.zeros(1, device=device)], dim=0)


def sample_training_sigmas(
    batch_size: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    device: torch.device,
) -> torch.Tensor:
    uniform = torch.rand(batch_size, device=device)
    min_inv = sigma_min ** (1.0 / rho)
    max_inv = sigma_max ** (1.0 / rho)
    return (max_inv + uniform * (min_inv - max_inv)) ** rho


def approximate_spherical_noise(
    reference: torch.Tensor,
    latitudes: torch.Tensor,
) -> torch.Tensor:
    if latitudes.ndim > 1:
        latitudes = latitudes[0]
    noise = torch.randn_like(reference)
    weights = torch.cos(torch.deg2rad(latitudes)).clamp_min(1e-6).rsqrt()
    return noise * weights.view(1, 1, -1, 1)


@dataclass
class EDMCoefficients:
    c_in: torch.Tensor
    c_out: torch.Tensor
    c_skip: torch.Tensor
    loss_weight: torch.Tensor


def edm_coefficients(sigmas: torch.Tensor, sigma_data: float) -> EDMCoefficients:
    sigma_data_tensor = torch.as_tensor(
        sigma_data, device=sigmas.device, dtype=sigmas.dtype
    )
    denom = torch.sqrt(sigmas**2 + sigma_data_tensor**2)
    c_in = 1.0 / denom
    c_skip = sigma_data_tensor**2 / (sigmas**2 + sigma_data_tensor**2)
    c_out = sigmas * sigma_data_tensor / denom
    loss_weight = (sigmas**2 + sigma_data_tensor**2) / (
        (sigmas * sigma_data_tensor).clamp_min(1e-6) ** 2
    )
    return EDMCoefficients(c_in=c_in, c_out=c_out, c_skip=c_skip, loss_weight=loss_weight)


class EDMSampler:
    def __init__(
        self,
        sigma_min: float,
        sigma_max: float,
        rho: float,
        num_steps: int,
        stochastic_churn: float,
        churn_min_sigma: float,
        churn_max_sigma: float,
        noise_inflation: float,
    ) -> None:
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.num_steps = num_steps
        self.stochastic_churn = stochastic_churn
        self.churn_min_sigma = churn_min_sigma
        self.churn_max_sigma = churn_max_sigma
        self.noise_inflation = noise_inflation

    @torch.no_grad()
    def sample(
        self,
        denoiser: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        shape: torch.Size,
        latitudes: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        sigmas = karras_schedule(
            num_steps=self.num_steps,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=self.rho,
            device=device,
        )
        x = approximate_spherical_noise(
            torch.empty(shape, device=device),
            latitudes=latitudes.to(device),
        ) * sigmas[0]

        for index in range(len(sigmas) - 1):
            sigma = sigmas[index]
            sigma_next = sigmas[index + 1]
            gamma = 0.0
            if self.churn_min_sigma <= float(sigma) <= self.churn_max_sigma:
                gamma = min(
                    self.stochastic_churn / max(self.num_steps - 1, 1),
                    2.0**0.5 - 1.0,
                )

            sigma_hat = sigma * (1.0 + gamma)
            if gamma > 0.0:
                eps = approximate_spherical_noise(x, latitudes=latitudes.to(device))
                x = x + eps * self.noise_inflation * torch.sqrt(
                    torch.clamp(sigma_hat**2 - sigma**2, min=0.0)
                )

            denoised = denoiser(x, sigma_hat.expand(shape[0]))
            derivative = (x - denoised) / sigma_hat.clamp_min(1e-6)
            x_euler = x + (sigma_next - sigma_hat) * derivative

            if sigma_next > 0:
                denoised_next = denoiser(x_euler, sigma_next.expand(shape[0]))
                derivative_next = (x_euler - denoised_next) / sigma_next.clamp_min(1e-6)
                x = x + (sigma_next - sigma_hat) * 0.5 * (derivative + derivative_next)
            else:
                x = x_euler
        return x
