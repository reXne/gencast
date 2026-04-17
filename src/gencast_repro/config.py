from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Optional, Tuple, Type, TypeVar, get_type_hints

import yaml

from .constants import (
    DEFAULT_ATMOSPHERIC_VARS,
    DEFAULT_FORCING_VARS,
    DEFAULT_PRECIPITATION_VARS,
    DEFAULT_PRESSURE_LEVELS,
    DEFAULT_STATIC_VARS,
    DEFAULT_SURFACE_VARS,
)


T = TypeVar("T")


@dataclass
class DataConfig:
    source: str = ""
    source_format: str = "zarr"
    train_split: Tuple[str, str] = ("1979-01-01", "2017-12-31")
    valid_split: Tuple[str, str] = ("2018-01-01", "2018-12-31")
    test_split: Tuple[str, str] = ("2019-01-01", "2019-12-31")
    surface_variables: Tuple[str, ...] = DEFAULT_SURFACE_VARS
    atmospheric_variables: Tuple[str, ...] = DEFAULT_ATMOSPHERIC_VARS
    pressure_levels: Tuple[int, ...] = DEFAULT_PRESSURE_LEVELS
    forcing_variables: Tuple[str, ...] = DEFAULT_FORCING_VARS
    static_variables: Tuple[str, ...] = DEFAULT_STATIC_VARS
    precipitation_variables: Tuple[str, ...] = DEFAULT_PRECIPITATION_VARS
    batch_size: int = 1
    model_timestep_hours: int = 12
    native_timestep_hours: int = 6
    num_workers: int = 0
    pin_memory: bool = False
    shuffle: bool = True
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    stats_path: str = "artifacts/stats.npz"


@dataclass
class ModelConfig:
    resolution: float = 1.0
    mesh_refinement: int = 4
    k_grid_to_mesh: int = 4
    k_mesh_to_grid: int = 3
    attention_k_hop: int = 8
    latent_dim: int = 256
    processor_layers: int = 8
    processor_heads: int = 4
    ffw_hidden_dim: int = 1024
    dropout: float = 0.0
    input_steps: int = 2
    noise_fourier_base_period: float = 16.0
    noise_fourier_features: int = 32
    noise_embedding_dim: int = 16


@dataclass
class DiffusionConfig:
    sigma_data: float = 1.0
    sigma_min: float = 0.02
    sigma_max: float = 88.0
    rho: float = 7.0
    num_sampling_steps: int = 20
    stochastic_churn: float = 2.5
    churn_min_sigma: float = 0.75
    churn_max_sigma: float = 999999.0
    noise_inflation: float = 1.05


@dataclass
class TrainingConfig:
    device: str = "cpu"
    epochs: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    log_every: int = 10
    valid_every: int = 1
    num_ensemble_members: int = 8
    checkpoint_dir: str = "artifacts/checkpoints"
    resume_path: str = ""
    seed: int = 42


@dataclass
class RolloutConfig:
    forecast_steps: int = 30
    ensemble_size: int = 8
    output_path: str = "artifacts/forecast.npz"


@dataclass
class SyntheticConfig:
    num_time_steps: int = 96
    random_seed: int = 42


@dataclass
class ExperimentConfig:
    name: str = "gencast-repro"
    use_synthetic_data: bool = False
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)


def _coerce_dataclass(cls: Type[T], raw: Dict[str, Any]) -> T:
    kwargs: Dict[str, Any] = {}
    type_hints = get_type_hints(cls)
    for item in fields(cls):
        if item.name not in raw:
            continue
        value = raw[item.name]
        field_type = type_hints.get(item.name, item.type)
        if dataclasses.is_dataclass(field_type):
            kwargs[item.name] = _coerce_dataclass(field_type, value)
        elif isinstance(value, list) and getattr(field_type, "__origin__", None) is tuple:
            kwargs[item.name] = tuple(value)
        else:
            kwargs[item.name] = value
    return cls(**kwargs)


def load_experiment_config(path: str) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return _coerce_dataclass(ExperimentConfig, raw)
