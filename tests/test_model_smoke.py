import torch

from gencast_repro.config import load_experiment_config
from gencast_repro.data.variables import ChannelLayout
from gencast_repro.models.gencast import GenCastDenoiser


def test_model_forward_smoke():
    config = load_experiment_config("configs/mini.yaml")
    layout = ChannelLayout.from_config(config.data, input_steps=config.model.input_steps)
    model = GenCastDenoiser(layout, config.model, config.diffusion)
    lat, lon = 19, 36
    conditioning = torch.randn(2, 2, layout.num_state_channels, lat, lon)
    forcings = torch.randn(2, layout.num_forcing_channels, lat, lon)
    noisy_target = torch.randn(2, layout.num_state_channels, lat, lon)
    sigmas = torch.full((2,), 1.0)
    output = model(conditioning, forcings, noisy_target, sigmas)
    assert output.shape == noisy_target.shape
