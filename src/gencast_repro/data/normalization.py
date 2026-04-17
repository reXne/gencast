from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class WeatherStatistics:
    state_mean: np.ndarray
    state_std: np.ndarray
    residual_mean: np.ndarray
    residual_std: np.ndarray
    forcing_mean: np.ndarray
    forcing_std: np.ndarray
    state_channel_names: np.ndarray
    forcing_channel_names: np.ndarray

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            state_mean=self.state_mean,
            state_std=self.state_std,
            residual_mean=self.residual_mean,
            residual_std=self.residual_std,
            forcing_mean=self.forcing_mean,
            forcing_std=self.forcing_std,
            state_channel_names=self.state_channel_names,
            forcing_channel_names=self.forcing_channel_names,
        )

    @classmethod
    def load(cls, path: str) -> "WeatherStatistics":
        raw = np.load(path, allow_pickle=True)
        return cls(
            state_mean=raw["state_mean"].astype(np.float32),
            state_std=raw["state_std"].astype(np.float32),
            residual_mean=raw["residual_mean"].astype(np.float32),
            residual_std=raw["residual_std"].astype(np.float32),
            forcing_mean=raw["forcing_mean"].astype(np.float32),
            forcing_std=raw["forcing_std"].astype(np.float32),
            state_channel_names=raw["state_channel_names"],
            forcing_channel_names=raw["forcing_channel_names"],
        )

    @staticmethod
    def _reshape_stats(stats: np.ndarray, x: np.ndarray, channel_dim: int) -> np.ndarray:
        shape = [1] * x.ndim
        shape[channel_dim] = stats.shape[0]
        return stats.reshape(shape)

    @staticmethod
    def _reshape_stats_torch(
        stats: np.ndarray, x: torch.Tensor, channel_dim: int
    ) -> torch.Tensor:
        shape = [1] * x.ndim
        shape[channel_dim] = stats.shape[0]
        return torch.as_tensor(stats, device=x.device, dtype=x.dtype).view(*shape)

    def normalize_states(self, x: np.ndarray, channel_dim: int = 1) -> np.ndarray:
        mean = self._reshape_stats(self.state_mean, x, channel_dim)
        std = self._reshape_stats(self.state_std, x, channel_dim)
        return (x - mean) / std

    def denormalize_states(self, x: np.ndarray, channel_dim: int = 1) -> np.ndarray:
        mean = self._reshape_stats(self.state_mean, x, channel_dim)
        std = self._reshape_stats(self.state_std, x, channel_dim)
        return x * std + mean

    def normalize_residuals(self, x: np.ndarray, channel_dim: int = 1) -> np.ndarray:
        mean = self._reshape_stats(self.residual_mean, x, channel_dim)
        std = self._reshape_stats(self.residual_std, x, channel_dim)
        return (x - mean) / std

    def denormalize_residuals(self, x: np.ndarray, channel_dim: int = 1) -> np.ndarray:
        mean = self._reshape_stats(self.residual_mean, x, channel_dim)
        std = self._reshape_stats(self.residual_std, x, channel_dim)
        return x * std + mean

    def normalize_forcings(self, x: np.ndarray, channel_dim: int = 0) -> np.ndarray:
        mean = self._reshape_stats(self.forcing_mean, x, channel_dim)
        std = self._reshape_stats(self.forcing_std, x, channel_dim)
        return (x - mean) / std

    def normalize_states_torch(
        self, x: torch.Tensor, channel_dim: int = 1
    ) -> torch.Tensor:
        mean = self._reshape_stats_torch(self.state_mean, x, channel_dim)
        std = self._reshape_stats_torch(self.state_std, x, channel_dim)
        return (x - mean) / std

    def denormalize_states_torch(
        self, x: torch.Tensor, channel_dim: int = 1
    ) -> torch.Tensor:
        mean = self._reshape_stats_torch(self.state_mean, x, channel_dim)
        std = self._reshape_stats_torch(self.state_std, x, channel_dim)
        return x * std + mean

    def normalize_residuals_torch(
        self, x: torch.Tensor, channel_dim: int = 1
    ) -> torch.Tensor:
        mean = self._reshape_stats_torch(self.residual_mean, x, channel_dim)
        std = self._reshape_stats_torch(self.residual_std, x, channel_dim)
        return (x - mean) / std

    def denormalize_residuals_torch(
        self, x: torch.Tensor, channel_dim: int = 1
    ) -> torch.Tensor:
        mean = self._reshape_stats_torch(self.residual_mean, x, channel_dim)
        std = self._reshape_stats_torch(self.residual_std, x, channel_dim)
        return x * std + mean

    def normalize_forcings_torch(
        self, x: torch.Tensor, channel_dim: int = 1
    ) -> torch.Tensor:
        mean = self._reshape_stats_torch(self.forcing_mean, x, channel_dim)
        std = self._reshape_stats_torch(self.forcing_std, x, channel_dim)
        return (x - mean) / std

