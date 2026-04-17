from __future__ import annotations

import torch
from torch import nn

from .layers import MLP


def scatter_mean(messages: torch.Tensor, indices: torch.Tensor, dim_size: int) -> torch.Tensor:
    batch_size, _, feature_dim = messages.shape
    aggregated = messages.new_zeros((batch_size, dim_size, feature_dim))
    counts = messages.new_zeros((dim_size,), dtype=messages.dtype)
    ones = torch.ones_like(indices, dtype=messages.dtype)
    counts.index_add_(0, indices, ones)
    counts = counts.clamp_min(1.0)
    for batch in range(batch_size):
        aggregated[batch].index_add_(0, indices, messages[batch])
    aggregated = aggregated / counts.view(1, dim_size, 1)
    return aggregated


class BipartiteGraphBlock(nn.Module):
    def __init__(
        self,
        sender_dim: int,
        receiver_dim: int,
        edge_dim: int,
        hidden_dim: int,
        cond_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.message_mlp = MLP(
            in_dim=sender_dim + receiver_dim + edge_dim + cond_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_hidden_layers=2,
            dropout=dropout,
        )
        self.update_mlp = MLP(
            in_dim=receiver_dim + hidden_dim + cond_dim,
            hidden_dim=hidden_dim,
            out_dim=receiver_dim,
            num_hidden_layers=2,
            dropout=dropout,
        )
        self.receiver_norm = nn.LayerNorm(receiver_dim + hidden_dim + cond_dim)

    def forward(
        self,
        sender_features: torch.Tensor,
        receiver_features: torch.Tensor,
        edge_features: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        sender_values = sender_features[:, senders, :]
        receiver_values = receiver_features[:, receivers, :]
        conditioning_edges = conditioning[:, None, :].expand(-1, senders.shape[0], -1)
        edge_values = edge_features[None, :, :].expand(sender_features.shape[0], -1, -1)
        message_inputs = torch.cat(
            [sender_values, receiver_values, edge_values, conditioning_edges],
            dim=-1,
        )
        messages = self.message_mlp(message_inputs)
        aggregated = scatter_mean(messages, receivers, receiver_features.shape[1])
        conditioning_nodes = conditioning[:, None, :].expand(-1, receiver_features.shape[1], -1)
        update_inputs = torch.cat([receiver_features, aggregated, conditioning_nodes], dim=-1)
        return receiver_features + self.update_mlp(self.receiver_norm(update_inputs))
