from __future__ import annotations

import argparse
import os

from .config import load_experiment_config
from .training.engine import evaluate_experiment, fit_experiment, sample_experiment
from .data.dataset import compute_weather_statistics, open_weather_source, slice_time
from .data.synthetic import create_synthetic_weather_dataset
from .data.variables import ChannelLayout


def _command_fit_normalizer(args: argparse.Namespace) -> None:
    config = load_experiment_config(args.config)
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
    stats = compute_weather_statistics(
        ds=slice_time(ds, config.data.train_split),
        layout=layout,
        model_timestep_hours=config.data.model_timestep_hours,
        native_timestep_hours=config.data.native_timestep_hours,
        max_samples=config.data.max_train_samples,
    )
    os.makedirs(os.path.dirname(config.data.stats_path) or ".", exist_ok=True)
    stats.save(config.data.stats_path)
    print("saved_stats=%s" % config.data.stats_path)


def _command_train(args: argparse.Namespace) -> None:
    config = load_experiment_config(args.config)
    checkpoint = fit_experiment(config)
    print("best_checkpoint=%s" % checkpoint)


def _command_evaluate(args: argparse.Namespace) -> None:
    config = load_experiment_config(args.config)
    evaluate_experiment(config, args.checkpoint)


def _command_sample(args: argparse.Namespace) -> None:
    config = load_experiment_config(args.config)
    sample_experiment(config, args.checkpoint, split=args.split, index=args.index)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GenCast-style weather forecasting repro.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_norm = subparsers.add_parser("fit-normalizer")
    fit_norm.add_argument("--config", required=True)
    fit_norm.set_defaults(func=_command_fit_normalizer)

    train = subparsers.add_parser("train")
    train.add_argument("--config", required=True)
    train.set_defaults(func=_command_train)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.set_defaults(func=_command_evaluate)

    sample = subparsers.add_parser("sample")
    sample.add_argument("--config", required=True)
    sample.add_argument("--checkpoint", required=True)
    sample.add_argument("--split", default="test")
    sample.add_argument("--index", type=int, default=0)
    sample.set_defaults(func=_command_sample)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
