from __future__ import annotations

from typing import Tuple

import numpy as np
import xarray as xr

from ..geometry.spherical import make_global_grid


def create_synthetic_weather_dataset(
    layout: object,
    resolution: float,
    num_time_steps: int = 96,
    native_timestep_hours: int = 6,
    seed: int = 42,
) -> xr.Dataset:
    rng = np.random.default_rng(seed)
    latitudes, longitudes = make_global_grid(resolution)
    times = np.arange(num_time_steps, dtype=np.int64)
    time_coord = np.datetime64("2019-01-01T00:00:00") + times * np.timedelta64(
        native_timestep_hours, "h"
    )
    time_coord = time_coord.astype("datetime64[ns]")
    pressure_levels = np.asarray(layout.pressure_levels, dtype=np.int32)

    lat_grid, lon_grid = np.meshgrid(latitudes, longitudes, indexing="ij")
    base_wave = (
        np.sin(np.deg2rad(lat_grid))
        + 0.5 * np.cos(np.deg2rad(lon_grid))
        + 0.25 * np.sin(np.deg2rad(lat_grid + lon_grid))
    ).astype(np.float32)

    data_vars = {}
    for variable in layout.surface_variables:
        frames = []
        for step in range(num_time_steps):
            phase = 2.0 * np.pi * step / max(num_time_steps, 1)
            frame = base_wave + 0.15 * np.sin(phase + np.deg2rad(lon_grid))
            frame += 0.05 * rng.standard_normal(size=base_wave.shape)
            if "pressure" in variable:
                frame = 1013.0 + 5.0 * frame
            elif "temperature" in variable:
                frame = 280.0 + 12.0 * frame
            elif "precipitation" in variable:
                frame = np.maximum(0.0, frame + 0.15)
            elif "sea_surface_temperature" in variable:
                frame = 290.0 + 4.0 * frame
            else:
                frame = 5.0 * frame
            frames.append(frame.astype(np.float32))
        data_vars[variable] = xr.DataArray(
            np.stack(frames, axis=0),
            dims=("time", "lat", "lon"),
            coords={"time": time_coord, "lat": latitudes, "lon": longitudes},
        )

    for variable in layout.atmospheric_variables:
        frames = []
        for step in range(num_time_steps):
            phase = 2.0 * np.pi * step / max(num_time_steps, 1)
            level_frames = []
            for level in layout.pressure_levels:
                level_scale = 1.0 + 0.001 * level
                frame = level_scale * base_wave + 0.1 * np.cos(
                    phase + np.deg2rad(lon_grid * 0.5)
                )
                frame += 0.05 * rng.standard_normal(size=base_wave.shape)
                if variable == "temperature":
                    frame = 220.0 + 30.0 * frame
                elif variable == "geopotential":
                    frame = 1000.0 + 60.0 * frame
                elif variable in ("u_component_of_wind", "v_component_of_wind"):
                    frame = 15.0 * frame
                elif variable == "vertical_velocity":
                    frame = 0.2 * frame
                elif variable == "specific_humidity":
                    frame = np.maximum(0.0, 0.01 + 0.005 * frame)
                else:
                    frame = frame
                level_frames.append(frame.astype(np.float32))
            frames.append(np.stack(level_frames, axis=0))
        data_vars[variable] = xr.DataArray(
            np.stack(frames, axis=0),
            dims=("time", "level", "lat", "lon"),
            coords={
                "time": time_coord,
                "level": pressure_levels,
                "lat": latitudes,
                "lon": longitudes,
            },
        )

    data_vars["geopotential_at_surface"] = xr.DataArray(
        50.0 * np.sin(np.deg2rad(lat_grid)).astype(np.float32),
        dims=("lat", "lon"),
        coords={"lat": latitudes, "lon": longitudes},
    )
    data_vars["land_sea_mask"] = xr.DataArray(
        (np.sin(np.deg2rad(lon_grid * 2.0)) > 0).astype(np.float32),
        dims=("lat", "lon"),
        coords={"lat": latitudes, "lon": longitudes},
    )

    return xr.Dataset(data_vars=data_vars)
