from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def make_global_grid(resolution: float) -> Tuple[np.ndarray, np.ndarray]:
    latitudes = np.arange(90.0, -90.0 - 1e-6, -resolution, dtype=np.float32)
    longitudes = np.arange(0.0, 360.0, resolution, dtype=np.float32)
    return latitudes, longitudes


def lat_lon_to_cartesian(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    lat_radians = np.deg2rad(latitudes)
    lon_radians = np.deg2rad(longitudes)
    x = np.cos(lat_radians) * np.cos(lon_radians)
    y = np.cos(lat_radians) * np.sin(lon_radians)
    z = np.sin(lat_radians)
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def grid_lat_lon_to_cartesian(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes, indexing="xy")
    return lat_lon_to_cartesian(lat_grid.reshape(-1), lon_grid.reshape(-1))


def cartesian_to_lat_lon(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    latitudes = np.rad2deg(np.arcsin(np.clip(z, -1.0, 1.0))).astype(np.float32)
    longitudes = np.rad2deg(np.arctan2(y, x)).astype(np.float32)
    longitudes = np.mod(longitudes + 360.0, 360.0)
    return latitudes, longitudes


def wrap_longitude_delta(delta: np.ndarray) -> np.ndarray:
    return ((delta + 180.0) % 360.0) - 180.0


def cell_area_weights(latitudes: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(latitudes)).astype(np.float32)
    weights = np.clip(weights, 1e-6, None)
    return (weights / weights.mean()).astype(np.float32)

