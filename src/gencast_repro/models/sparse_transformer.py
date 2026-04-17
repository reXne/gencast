from __future__ import annotations

import math

import torch
from torch import nn

from .layers import AdaLayerNorm, MLP


class NeighborhoodSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim debe ser divisible por num_heads.")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, hidden_dim = x.shape
        neighbor_index = neighbors.clamp_min(0)
        neighbor_values = x[:, neighbor_index, :]

        q = self.q_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        k = self.k_proj(neighbor_values).view(
            batch_size, num_nodes, neighbor_index.shape[1], self.num_heads, self.head_dim
        )
        v = self.v_proj(neighbor_values).view(
            batch_size, num_nodes, neighbor_index.shape[1], self.num_heads, self.head_dim
        )

        scores = torch.einsum("bnhd,bnkhd->bnhk", q, k) / math.sqrt(self.head_dim)
        mask = neighbors.ge(0)[None, :, None, :]
        scores = scores.masked_fill(~mask, -1e9)
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        attended = torch.einsum("bnhk,bnkhd->bnhd", attention, v)
        attended = attended.reshape(batch_size, num_nodes, hidden_dim)
        return self.out_proj(attended)


class SparseTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        cond_dim: int,
        num_heads: int,
        ffw_hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = AdaLayerNorm(dim, cond_dim)
        self.norm2 = AdaLayerNorm(dim, cond_dim)
        self.attention = NeighborhoodSelfAttention(dim, num_heads, dropout=dropout)
        self.ffn = MLP(
            in_dim=dim,
            hidden_dim=ffw_hidden_dim,
            out_dim=dim,
            num_hidden_layers=2,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
        neighbors: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attention(self.norm1(x, conditioning), neighbors)
        x = x + self.ffn(self.norm2(x, conditioning))
        return x


class SparseTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        cond_dim: int,
        num_layers: int,
        num_heads: int,
        ffw_hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                SparseTransformerBlock(
                    dim=dim,
                    cond_dim=cond_dim,
                    num_heads=num_heads,
                    ffw_hidden_dim=ffw_hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
        neighbors: torch.Tensor,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, conditioning, neighbors)
        return x

