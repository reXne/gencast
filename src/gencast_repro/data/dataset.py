from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

from .normalization import WeatherStatistics
from .variables import ChannelLayout


DIM_RENAMES = {
    "latitude": "lat",
    "longitude": "lon",
    "isobaricInhPa": "level",
    "pressure_level": "level",
}


def standardize_dataset(ds: xr.Dataset) -> xr.Dataset:
    rename_map = {name: DIM_RENAMES[name] for name in ds.dims if name in DIM_RENAMES}
    rename_map.update(
        {name: DIM_RENAMES[name] for name in ds.coords if name in DIM_RENAMES}
    )
    if rename_map:
        ds = ds.rename(rename_map)
    if "lat" not in ds.coords or "lon" not in ds.coords:
        raise ValueError("El dataset debe contener coordenadas lat/lon.")
    return ds


def open_weather_source(path: str, source_format: str = "zarr") -> xr.Dataset:
    if source_format == "zarr" or path.endswith(".zarr"):
        return standardize_dataset(xr.open_zarr(path))
    if any(token in path for token in ("*", "?", "[")):
        return standardize_dataset(xr.open_mfdataset(path, combine="by_coords"))
    return standardize_dataset(xr.open_dataset(path))


def slice_time(ds: xr.Dataset, time_range: Tuple[str, str]) -> xr.Dataset:
    start, end = time_range
    return ds.sel(time=slice(start, end))


def _timestamp_progress(timestamp: np.datetime64) -> Tuple[float, float]:
    timestamp = np.datetime64(timestamp, "s")
    year = timestamp.astype("datetime64[Y]")
    year_start = year.astype("datetime64[s]")
    next_year = (year + np.timedelta64(1, "Y")).astype("datetime64[s]")
    day = timestamp.astype("datetime64[D]")
    year_progress = float(
        (timestamp - year_start) / np.maximum(next_year - year_start, np.timedelta64(1, "s"))
    )
    day_progress = float((timestamp - day) / np.timedelta64(1, "D"))
    return year_progress, day_progress


def build_forcing_array(
    timestamp: np.datetime64,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    static_fields: Dict[str, np.ndarray],
    layout: ChannelLayout,
) -> np.ndarray:
    height = latitudes.shape[0]
    width = longitudes.shape[0]
    year_progress, day_progress = _timestamp_progress(timestamp)

    forcing_channels: List[np.ndarray] = []
    for variable in layout.forcing_variables:
        if variable == "year_progress_sin":
            array = np.full((height, width), np.sin(2.0 * np.pi * year_progress), dtype=np.float32)
        elif variable == "year_progress_cos":
            array = np.full((height, width), np.cos(2.0 * np.pi * year_progress), dtype=np.float32)
        elif variable == "day_progress_sin":
            array = np.full((height, width), np.sin(2.0 * np.pi * day_progress), dtype=np.float32)
        elif variable == "day_progress_cos":
            array = np.full((height, width), np.cos(2.0 * np.pi * day_progress), dtype=np.float32)
        else:
            raise KeyError("Forcing no soportado: %s" % variable)
        forcing_channels.append(array)

    for variable in layout.static_variables:
        forcing_channels.append(static_fields[variable].astype(np.float32))
    return np.stack(forcing_channels, axis=0).astype(np.float32)


def _extract_array(ds: xr.Dataset, variable: str, time_index: Optional[int] = None) -> xr.DataArray:
    data = ds[variable]
    if time_index is not None and "time" in data.dims:
        data = data.isel(time=time_index)
    return data


def stack_state_channels(ds: xr.Dataset, layout: ChannelLayout, time_index: int) -> np.ndarray:
    channels: List[np.ndarray] = []
    for variable in layout.surface_variables:
        array = _extract_array(ds, variable, time_index).transpose("lat", "lon").values
        channels.append(array.astype(np.float32)[None, ...])
    for variable in layout.atmospheric_variables:
        array = (
            _extract_array(ds, variable, time_index)
            .sel(level=list(layout.pressure_levels))
            .transpose("level", "lat", "lon")
            .values
        )
        channels.append(array.astype(np.float32))
    return np.concatenate(channels, axis=0).astype(np.float32)


def stack_static_fields(ds: xr.Dataset, layout: ChannelLayout) -> Dict[str, np.ndarray]:
    fields: Dict[str, np.ndarray] = {}
    for variable in layout.static_variables:
        data = ds[variable]
        if "time" in data.dims:
            data = data.isel(time=0)
        fields[variable] = data.transpose("lat", "lon").values.astype(np.float32)
    return fields


def compute_weather_statistics(
    ds: xr.Dataset,
    layout: ChannelLayout,
    model_timestep_hours: int,
    native_timestep_hours: int,
    max_samples: Optional[int] = None,
) -> WeatherStatistics:
    ds = standardize_dataset(ds)
    gap = max(1, model_timestep_hours // native_timestep_hours)
    total_samples = max(0, ds.sizes["time"] - 2 * gap)
    if max_samples is not None:
        total_samples = min(total_samples, max_samples)

    state_sum = np.zeros(layout.num_state_channels, dtype=np.float64)
    state_sq_sum = np.zeros(layout.num_state_channels, dtype=np.float64)
    residual_sum = np.zeros(layout.num_state_channels, dtype=np.float64)
    residual_sq_sum = np.zeros(layout.num_state_channels, dtype=np.float64)
    forcing_sum = np.zeros(layout.num_forcing_channels, dtype=np.float64)
    forcing_sq_sum = np.zeros(layout.num_forcing_channels, dtype=np.float64)

    precipitation_mask = layout.precipitation_channel_mask()
    static_fields = stack_static_fields(ds, layout)
    latitudes = ds["lat"].values.astype(np.float32)
    longitudes = ds["lon"].values.astype(np.float32)

    num_state_values = 0
    num_residual_values = 0
    num_forcing_values = 0

    for start in range(total_samples):
        previous_state = stack_state_channels(ds, layout, start)
        current_state = stack_state_channels(ds, layout, start + gap)
        target_state = stack_state_channels(ds, layout, start + 2 * gap)
        forcing = build_forcing_array(
            ds["time"].values[start + 2 * gap],
            latitudes,
            longitudes,
            static_fields,
            layout,
        )

        residual = target_state - current_state
        residual[precipitation_mask] = target_state[precipitation_mask]

        state_sum += previous_state.sum(axis=(1, 2)) + current_state.sum(axis=(1, 2))
        state_sq_sum += (previous_state**2).sum(axis=(1, 2)) + (current_state**2).sum(axis=(1, 2))
        residual_sum += residual.sum(axis=(1, 2))
        residual_sq_sum += (residual**2).sum(axis=(1, 2))
        forcing_sum += forcing.sum(axis=(1, 2))
        forcing_sq_sum += (forcing**2).sum(axis=(1, 2))

        pixels = previous_state.shape[1] * previous_state.shape[2]
        num_state_values += 2 * pixels
        num_residual_values += pixels
        num_forcing_values += pixels

    if num_state_values == 0:
        raise ValueError("No hay suficientes timestamps para calcular estadisticas.")

    state_mean = state_sum / num_state_values
    residual_mean = residual_sum / num_residual_values
    forcing_mean = forcing_sum / num_forcing_values

    state_var = state_sq_sum / num_state_values - state_mean**2
    residual_var = residual_sq_sum / num_residual_values - residual_mean**2
    forcing_var = forcing_sq_sum / num_forcing_values - forcing_mean**2

    return WeatherStatistics(
        state_mean=state_mean.astype(np.float32),
        state_std=np.sqrt(np.maximum(state_var, 1e-6)).astype(np.float32),
        residual_mean=residual_mean.astype(np.float32),
        residual_std=np.sqrt(np.maximum(residual_var, 1e-6)).astype(np.float32),
        forcing_mean=forcing_mean.astype(np.float32),
        forcing_std=np.sqrt(np.maximum(forcing_var, 1e-6)).astype(np.float32),
        state_channel_names=np.asarray(layout.state_channel_names),
        forcing_channel_names=np.asarray(layout.forcing_channel_names),
    )


@dataclass
class SampleMetadata:
    input_time: str
    target_time: str


class WeatherDataset(Dataset):
    def __init__(
        self,
        ds: xr.Dataset,
        layout: ChannelLayout,
        model_timestep_hours: int,
        native_timestep_hours: int,
        statistics: Optional[WeatherStatistics] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.ds = standardize_dataset(ds)
        self.layout = layout
        self.statistics = statistics
        self.gap = max(1, model_timestep_hours // native_timestep_hours)
        self.latitudes = self.ds["lat"].values.astype(np.float32)
        self.longitudes = self.ds["lon"].values.astype(np.float32)
        self.static_fields = stack_static_fields(self.ds, layout)
        total = max(0, self.ds.sizes["time"] - 2 * self.gap)
        self.sample_starts = list(range(total))
        if max_samples is not None:
            self.sample_starts = self.sample_starts[:max_samples]
        self.precipitation_mask = layout.precipitation_channel_mask()

    def __len__(self) -> int:
        return len(self.sample_starts)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        start = self.sample_starts[index]
        previous_state = stack_state_channels(self.ds, self.layout, start)
        current_state = stack_state_channels(self.ds, self.layout, start + self.gap)
        target_state = stack_state_channels(self.ds, self.layout, start + 2 * self.gap)
        forcing = build_forcing_array(
            self.ds["time"].values[start + 2 * self.gap],
            self.latitudes,
            self.longitudes,
            self.static_fields,
            self.layout,
        )

        conditioning = np.stack([previous_state, current_state], axis=0).astype(np.float32)
        residual = (target_state - current_state).astype(np.float32)
        residual[self.precipitation_mask] = target_state[self.precipitation_mask]

        if self.statistics is not None:
            conditioning = self.statistics.normalize_states(conditioning, channel_dim=1)
            residual = self.statistics.normalize_residuals(residual, channel_dim=0)
            forcing = self.statistics.normalize_forcings(forcing, channel_dim=0)

        return {
            "conditioning": torch.from_numpy(conditioning.astype(np.float32)),
            "forcings": torch.from_numpy(forcing.astype(np.float32)),
            "target": torch.from_numpy(residual.astype(np.float32)),
            "current_state": torch.from_numpy(current_state.astype(np.float32)),
            "target_state": torch.from_numpy(target_state.astype(np.float32)),
            "latitudes": torch.from_numpy(self.latitudes.copy()),
            "longitudes": torch.from_numpy(self.longitudes.copy()),
            "input_time": str(self.ds["time"].values[start + self.gap]),
            "target_time": str(self.ds["time"].values[start + 2 * self.gap]),
        }
