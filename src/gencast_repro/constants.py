from typing import Dict, Tuple


DEFAULT_PRESSURE_LEVELS = (
    50,
    100,
    150,
    200,
    250,
    300,
    400,
    500,
    600,
    700,
    850,
    925,
    1000,
)

DEFAULT_SURFACE_VARS = (
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_v_component_of_wind",
    "10m_u_component_of_wind",
    "total_precipitation_12hr",
    "sea_surface_temperature",
)

DEFAULT_ATMOSPHERIC_VARS = (
    "temperature",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "specific_humidity",
)

DEFAULT_FORCING_VARS = (
    "year_progress_sin",
    "year_progress_cos",
    "day_progress_sin",
    "day_progress_cos",
)

DEFAULT_STATIC_VARS = (
    "geopotential_at_surface",
    "land_sea_mask",
)

DEFAULT_PRECIPITATION_VARS = ("total_precipitation_12hr",)

DEFAULT_VARIABLE_WEIGHTS: Dict[str, float] = {
    "2m_temperature": 1.0,
    "10m_u_component_of_wind": 0.1,
    "10m_v_component_of_wind": 0.1,
    "mean_sea_level_pressure": 0.1,
    "sea_surface_temperature": 0.1,
    "total_precipitation_12hr": 0.1,
}

