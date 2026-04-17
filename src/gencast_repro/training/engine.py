from __future__ import annotations

import os
import random
from dataclasses import asdict
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import ExperimentConfig
from ..data.dataset import (
    WeatherDataset,
    compute_weather_statistics,
    open_weather_source,
    slice_time,
)
from ..data.normalization import WeatherStatistics
from ..data.synthetic import create_synthetic_weather_dataset
from ..data.variables import ChannelLayout
from ..inference import autoregressive_rollout, sample_next_residual
from ..models.diffusion import EDMSampler, sample_training_sigmas, approximate_spherical_noise
from ..models.gencast import GenCastDenoiser
from .losses import weighted_mse
from .metrics import bias, crps_ensemble, rmse, spread_skill_ratio


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_dataset_bundle(config: ExperimentConfig) -> Tuple[object, ChannelLayout, WeatherStatistics]:
    layout = ChannelLayout.from_config(config.data, input_steps=config.model.input_steps)
    if config.use_synthetic_data:
        ds = create_synthetic_weather_dataset(
            layout=layout,
            resolution=config.model.resolution,
            num_time_steps=config.synthetic.num_time_steps,
            native_timestep_hours=config.data.native_timestep_hours,
            seed=config.synthetic.random_seed,
        )
    else:
        ds = open_weather_source(config.data.source, config.data.source_format)

    stats_path = config.data.stats_path
    if os.path.exists(stats_path):
        statistics = WeatherStatistics.load(stats_path)
    else:
        ensure_dir(os.path.dirname(stats_path) or ".")
        statistics = compute_weather_statistics(
            ds=slice_time(ds, config.data.train_split),
            layout=layout,
            model_timestep_hours=config.data.model_timestep_hours,
            native_timestep_hours=config.data.native_timestep_hours,
            max_samples=config.data.max_train_samples,
        )
        statistics.save(stats_path)
    return ds, layout, statistics


def _build_dataloaders(
    config: ExperimentConfig,
    ds: object,
    layout: ChannelLayout,
    statistics: WeatherStatistics,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = WeatherDataset(
        ds=slice_time(ds, config.data.train_split),
        layout=layout,
        model_timestep_hours=config.data.model_timestep_hours,
        native_timestep_hours=config.data.native_timestep_hours,
        statistics=statistics,
        max_samples=config.data.max_train_samples,
    )
    valid_dataset = WeatherDataset(
        ds=slice_time(ds, config.data.valid_split),
        layout=layout,
        model_timestep_hours=config.data.model_timestep_hours,
        native_timestep_hours=config.data.native_timestep_hours,
        statistics=statistics,
        max_samples=config.data.max_eval_samples,
    )
    test_dataset = WeatherDataset(
        ds=slice_time(ds, config.data.test_split),
        layout=layout,
        model_timestep_hours=config.data.model_timestep_hours,
        native_timestep_hours=config.data.native_timestep_hours,
        statistics=statistics,
        max_samples=config.data.max_eval_samples,
    )
    loader_kwargs = dict(
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=config.data.shuffle,
        **loader_kwargs,
    )
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, valid_loader, test_loader


def _build_model_and_sampler(
    config: ExperimentConfig,
    layout: ChannelLayout,
    device: torch.device,
) -> Tuple[GenCastDenoiser, EDMSampler]:
    model = GenCastDenoiser(layout, config.model, config.diffusion).to(device)
    sampler = EDMSampler(
        sigma_min=config.diffusion.sigma_min,
        sigma_max=config.diffusion.sigma_max,
        rho=config.diffusion.rho,
        num_steps=config.diffusion.num_sampling_steps,
        stochastic_churn=config.diffusion.stochastic_churn,
        churn_min_sigma=config.diffusion.churn_min_sigma,
        churn_max_sigma=config.diffusion.churn_max_sigma,
        noise_inflation=config.diffusion.noise_inflation,
    )
    return model, sampler


def _move_batch(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _run_diffusion_loss(
    model: GenCastDenoiser,
    batch: Dict[str, object],
    config: ExperimentConfig,
    channel_weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    conditioning = batch["conditioning"]
    forcings = batch["forcings"]
    target = batch["target"]
    latitudes = batch["latitudes"]
    sigmas = sample_training_sigmas(
        batch_size=conditioning.shape[0],
        sigma_min=config.diffusion.sigma_min,
        sigma_max=config.diffusion.sigma_max,
        rho=config.diffusion.rho,
        device=conditioning.device,
    )
    noise = approximate_spherical_noise(target, latitudes) * sigmas[:, None, None, None]
    noisy_target = target + noise
    prediction = model(conditioning, forcings, noisy_target, sigmas)
    coeffs = model.diffusion_config
    sample_weight = (
        (sigmas**2 + coeffs.sigma_data**2)
        / ((sigmas * coeffs.sigma_data).clamp_min(1e-6) ** 2)
    )
    loss = weighted_mse(prediction, target, latitudes, channel_weights, sample_weight)
    return loss, prediction


def fit_experiment(config: ExperimentConfig) -> str:
    seed_everything(config.training.seed)
    device = torch.device(config.training.device)
    ds, layout, statistics = _load_dataset_bundle(config)
    train_loader, valid_loader, _ = _build_dataloaders(config, ds, layout, statistics)
    model, _ = _build_model_and_sampler(config, layout, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    ensure_dir(config.training.checkpoint_dir)
    channel_weights = torch.from_numpy(layout.variable_weight_vector()).to(device)

    best_valid = float("inf")
    best_path = os.path.join(config.training.checkpoint_dir, "best.pt")
    last_path = os.path.join(config.training.checkpoint_dir, "last.pt")

    for epoch in range(config.training.epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _run_diffusion_loss(model, batch, config, channel_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
            optimizer.step()
            if step % config.training.log_every == 0:
                print(
                    "epoch=%d step=%d train_loss=%.6f"
                    % (epoch, step, float(loss.detach().cpu()))
                )

        if epoch % config.training.valid_every == 0:
            model.eval()
            losses = []
            with torch.no_grad():
                for batch in valid_loader:
                    batch = _move_batch(batch, device)
                    loss, _ = _run_diffusion_loss(model, batch, config, channel_weights)
                    losses.append(float(loss.detach().cpu()))
            valid_loss = float(np.mean(losses)) if losses else float("inf")
            print("epoch=%d valid_loss=%.6f" % (epoch, valid_loss))
            payload = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": asdict(config),
            }
            torch.save(payload, last_path)
            if valid_loss < best_valid:
                best_valid = valid_loss
                torch.save(payload, best_path)

    return best_path if os.path.exists(best_path) else last_path


def _load_checkpoint(model: GenCastDenoiser, checkpoint_path: str, device: torch.device) -> None:
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])


def evaluate_experiment(config: ExperimentConfig, checkpoint_path: str) -> Dict[str, float]:
    device = torch.device(config.training.device)
    ds, layout, statistics = _load_dataset_bundle(config)
    _, _, test_loader = _build_dataloaders(config, ds, layout, statistics)
    model, sampler = _build_model_and_sampler(config, layout, device)
    _load_checkpoint(model, checkpoint_path, device)
    model.eval()

    metrics = {"rmse": [], "bias": [], "crps": [], "spread_skill": []}
    precipitation_mask = model.precipitation_mask.to(device)

    for batch in test_loader:
        batch = _move_batch(batch, device)
        conditioning = batch["conditioning"]
        forcings = batch["forcings"]
        target = batch["target"]
        current_state = batch["current_state"]
        latitudes = batch["latitudes"]

        members = []
        for _ in range(config.training.num_ensemble_members):
            sampled_residual = sample_next_residual(
                model=model,
                sampler=sampler,
                conditioning_states=conditioning,
                forcings=forcings,
                latitudes=latitudes,
            )
            residual_physical = statistics.denormalize_residuals_torch(
                sampled_residual, channel_dim=1
            )
            prediction = current_state + residual_physical
            prediction[:, precipitation_mask, :, :] = residual_physical[
                :, precipitation_mask, :, :
            ]
            members.append(prediction)

        ensemble = torch.stack(members, dim=1)
        target_physical = batch["target_state"]
        ensemble_mean = ensemble.mean(dim=1)

        metrics["rmse"].append(float(rmse(ensemble_mean, target_physical, latitudes).cpu()))
        metrics["bias"].append(float(bias(ensemble_mean, target_physical, latitudes).cpu()))
        metrics["crps"].append(float(crps_ensemble(ensemble, target_physical, latitudes).cpu()))
        metrics["spread_skill"].append(
            float(spread_skill_ratio(ensemble, target_physical, latitudes).cpu())
        )

    summary = {key: float(np.mean(values)) for key, values in metrics.items()}
    for key, value in summary.items():
        print("%s=%.6f" % (key, value))
    return summary


def sample_experiment(
    config: ExperimentConfig,
    checkpoint_path: str,
    split: str = "test",
    index: int = 0,
) -> str:
    device = torch.device(config.training.device)
    ds, layout, statistics = _load_dataset_bundle(config)
    train_loader, valid_loader, test_loader = _build_dataloaders(config, ds, layout, statistics)
    model, sampler = _build_model_and_sampler(config, layout, device)
    _load_checkpoint(model, checkpoint_path, device)
    model.eval()

    datasets = {
        "train": train_loader.dataset,
        "valid": valid_loader.dataset,
        "test": test_loader.dataset,
    }
    dataset = datasets[split]
    sample = dataset[index]
    conditioning = sample["conditioning"][None].to(device)
    current_state = sample["current_state"][None].to(device)
    latitudes = sample["latitudes"].to(device)
    longitudes = sample["longitudes"].to(device)
    static_fields = dataset.static_fields
    rollout = autoregressive_rollout(
        model=model,
        sampler=sampler,
        conditioning_states=conditioning,
        initial_current_state=current_state,
        latitudes=latitudes,
        longitudes=longitudes,
        layout=layout,
        statistics=statistics,
        static_fields=static_fields,
        start_time=sample["input_time"],
        steps=config.rollout.forecast_steps,
        step_hours=config.data.model_timestep_hours,
        precipitation_mask=model.precipitation_mask.to(device),
    )

    ensure_dir(os.path.dirname(config.rollout.output_path) or ".")
    np.savez_compressed(
        config.rollout.output_path,
        forecast=rollout.states.squeeze(0).cpu().numpy(),
        target_times=np.asarray(rollout.target_times),
        latitudes=sample["latitudes"].cpu().numpy(),
        longitudes=sample["longitudes"].cpu().numpy(),
        channels=np.asarray(layout.state_channel_names),
    )
    print("saved_sample=%s" % config.rollout.output_path)
    return config.rollout.output_path
