from __future__ import annotations

import torch
from torch import nn

from ..config import DiffusionConfig, ModelConfig
from ..data.variables import ChannelLayout
from ..geometry.connectivity import GraphConnectivity, build_connectivity
from ..geometry.icosphere import create_icosphere
from ..geometry.spherical import make_global_grid
from .diffusion import edm_coefficients
from .layers import FourierNoiseEncoder, MLP
from .message_passing import BipartiteGraphBlock
from .sparse_transformer import SparseTransformer


class GenCastCore(nn.Module):
    def __init__(
        self,
        layout: ChannelLayout,
        model_config: ModelConfig,
        connectivity: GraphConnectivity,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.model_config = model_config
        self.connectivity = connectivity
        self.hidden_dim = model_config.latent_dim
        grid_input_dim = (
            layout.input_steps * layout.num_state_channels
            + layout.num_state_channels
            + layout.num_forcing_channels
            + 5
        )
        mesh_input_dim = 5
        edge_dim = connectivity.grid_to_mesh_features.shape[-1]
        cond_dim = model_config.noise_embedding_dim

        self.noise_encoder = FourierNoiseEncoder(
            base_period=model_config.noise_fourier_base_period,
            num_frequencies=model_config.noise_fourier_features,
            hidden_dim=max(model_config.latent_dim // 2, cond_dim),
            out_dim=cond_dim,
        )
        self.grid_embed = MLP(
            in_dim=grid_input_dim,
            hidden_dim=model_config.latent_dim,
            out_dim=model_config.latent_dim,
            num_hidden_layers=2,
            dropout=model_config.dropout,
        )
        self.mesh_embed = MLP(
            in_dim=mesh_input_dim,
            hidden_dim=model_config.latent_dim,
            out_dim=model_config.latent_dim,
            num_hidden_layers=2,
            dropout=model_config.dropout,
        )
        self.grid_to_mesh = BipartiteGraphBlock(
            sender_dim=model_config.latent_dim,
            receiver_dim=model_config.latent_dim,
            edge_dim=edge_dim,
            hidden_dim=model_config.latent_dim,
            cond_dim=cond_dim,
            dropout=model_config.dropout,
        )
        self.processor = SparseTransformer(
            dim=model_config.latent_dim,
            cond_dim=cond_dim,
            num_layers=model_config.processor_layers,
            num_heads=model_config.processor_heads,
            ffw_hidden_dim=model_config.ffw_hidden_dim,
            dropout=model_config.dropout,
        )
        self.mesh_to_grid = BipartiteGraphBlock(
            sender_dim=model_config.latent_dim,
            receiver_dim=model_config.latent_dim,
            edge_dim=connectivity.mesh_to_grid_features.shape[-1],
            hidden_dim=model_config.latent_dim,
            cond_dim=cond_dim,
            dropout=model_config.dropout,
        )
        self.output_head = MLP(
            in_dim=model_config.latent_dim,
            hidden_dim=model_config.latent_dim,
            out_dim=layout.num_state_channels,
            num_hidden_layers=2,
            dropout=model_config.dropout,
        )

        grid_positions = torch.from_numpy(
            torch.stack(
                (
                    torch.from_numpy(connectivity.grid_xyz[:, 0]),
                    torch.from_numpy(connectivity.grid_xyz[:, 1]),
                    torch.from_numpy(connectivity.grid_xyz[:, 2]),
                    torch.from_numpy(connectivity.grid_latitudes / 90.0),
                    torch.from_numpy(connectivity.grid_longitudes / 180.0),
                ),
                dim=-1,
            )
            .cpu()
            .numpy()
        )
        mesh_positions = torch.from_numpy(
            torch.stack(
                (
                    torch.from_numpy(connectivity.mesh_xyz[:, 0]),
                    torch.from_numpy(connectivity.mesh_xyz[:, 1]),
                    torch.from_numpy(connectivity.mesh_xyz[:, 2]),
                    torch.from_numpy(connectivity.mesh_latitudes / 90.0),
                    torch.from_numpy(connectivity.mesh_longitudes / 180.0),
                ),
                dim=-1,
            )
            .cpu()
            .numpy()
        )

        self.register_buffer("grid_positions", grid_positions.float(), persistent=False)
        self.register_buffer("mesh_positions", mesh_positions.float(), persistent=False)
        self.register_buffer(
            "grid_to_mesh_senders",
            torch.from_numpy(connectivity.grid_to_mesh_senders.astype("int64")),
            persistent=False,
        )
        self.register_buffer(
            "grid_to_mesh_receivers",
            torch.from_numpy(connectivity.grid_to_mesh_receivers.astype("int64")),
            persistent=False,
        )
        self.register_buffer(
            "grid_to_mesh_features",
            torch.from_numpy(connectivity.grid_to_mesh_features.astype("float32")),
            persistent=False,
        )
        self.register_buffer(
            "mesh_to_grid_senders",
            torch.from_numpy(connectivity.mesh_to_grid_senders.astype("int64")),
            persistent=False,
        )
        self.register_buffer(
            "mesh_to_grid_receivers",
            torch.from_numpy(connectivity.mesh_to_grid_receivers.astype("int64")),
            persistent=False,
        )
        self.register_buffer(
            "mesh_to_grid_features",
            torch.from_numpy(connectivity.mesh_to_grid_features.astype("float32")),
            persistent=False,
        )
        self.register_buffer(
            "mesh_neighbors",
            torch.from_numpy(connectivity.mesh_neighbors.astype("int64")),
            persistent=False,
        )

    def forward(
        self,
        conditioning_states: torch.Tensor,
        forcings: torch.Tensor,
        scaled_noisy_target: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, input_steps, channels, height, width = conditioning_states.shape
        conditioning_flat = conditioning_states.reshape(
            batch_size, input_steps * channels, height, width
        )
        grid_inputs = torch.cat([conditioning_flat, scaled_noisy_target, forcings], dim=1)
        grid_nodes = grid_inputs.flatten(2).transpose(1, 2)
        grid_nodes = torch.cat(
            [grid_nodes, self.grid_positions[None, :, :].expand(batch_size, -1, -1)],
            dim=-1,
        )
        mesh_nodes = self.mesh_positions[None, :, :].expand(batch_size, -1, -1)

        noise_embedding = self.noise_encoder(sigmas)
        grid_latent = self.grid_embed(grid_nodes)
        mesh_latent = self.mesh_embed(mesh_nodes)
        mesh_latent = self.grid_to_mesh(
            sender_features=grid_latent,
            receiver_features=mesh_latent,
            edge_features=self.grid_to_mesh_features,
            senders=self.grid_to_mesh_senders,
            receivers=self.grid_to_mesh_receivers,
            conditioning=noise_embedding,
        )
        mesh_latent = self.processor(mesh_latent, noise_embedding, self.mesh_neighbors)
        grid_latent = self.mesh_to_grid(
            sender_features=mesh_latent,
            receiver_features=grid_latent,
            edge_features=self.mesh_to_grid_features,
            senders=self.mesh_to_grid_senders,
            receivers=self.mesh_to_grid_receivers,
            conditioning=noise_embedding,
        )
        output = self.output_head(grid_latent).transpose(1, 2).reshape(
            batch_size, channels, height, width
        )
        return output


class GenCastDenoiser(nn.Module):
    def __init__(
        self,
        layout: ChannelLayout,
        model_config: ModelConfig,
        diffusion_config: DiffusionConfig,
    ) -> None:
        super().__init__()
        latitudes, longitudes = make_global_grid(model_config.resolution)
        mesh_vertices, mesh_faces = create_icosphere(model_config.mesh_refinement)
        connectivity = build_connectivity(
            latitudes=latitudes,
            longitudes=longitudes,
            mesh_vertices=mesh_vertices,
            mesh_faces=mesh_faces,
            k_grid_to_mesh=model_config.k_grid_to_mesh,
            k_mesh_to_grid=model_config.k_mesh_to_grid,
            attention_k_hop=model_config.attention_k_hop,
        )
        self.layout = layout
        self.model_config = model_config
        self.diffusion_config = diffusion_config
        self.core = GenCastCore(layout, model_config, connectivity)
        self.register_buffer(
            "precipitation_mask",
            torch.from_numpy(layout.precipitation_channel_mask()),
            persistent=False,
        )

    def denoise(
        self,
        conditioning_states: torch.Tensor,
        forcings: torch.Tensor,
        noisy_target: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> torch.Tensor:
        coeffs = edm_coefficients(sigmas, self.diffusion_config.sigma_data)
        scaled_target = noisy_target * coeffs.c_in[:, None, None, None]
        raw = self.core(conditioning_states, forcings, scaled_target, sigmas)
        return raw * coeffs.c_out[:, None, None, None] + noisy_target * coeffs.c_skip[
            :, None, None, None
        ]

    def forward(
        self,
        conditioning_states: torch.Tensor,
        forcings: torch.Tensor,
        noisy_target: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> torch.Tensor:
        return self.denoise(conditioning_states, forcings, noisy_target, sigmas)
