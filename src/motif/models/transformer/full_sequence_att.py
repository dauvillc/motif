"""Full-sequence spatio-temporal self-attention over all merged source tokens."""

import torch.nn as nn
from torch import Tensor

from motif.datatypes import SourceEmbedding
from motif.models.flash_attention import SpatiotemporalFlashAttention
from motif.models.motif.rope_attention import SpatiotemporalRoPEAttention
from motif.models.motif.small_layers import RMSNorm


class FullSequenceAttention(nn.Module):
    """Self-attention over the full merged sequence of all source tokens.

    Expects a SourceEmbedding whose embedding and coords are already flat, i.e.
    of shape (B, L_total, dim) and (B, L_total, coords_dim).  Returns a Tensor
    of the same shape (B, L_total, dim).
    """

    def __init__(
        self,
        dim: int,
        inner_dim: int,
        num_heads: int,
        positional_encoding: str = "rpb",
        coords_dim: int | None = None,
        coords_inner_dim: int | None = None,
        dropout: float = 0.0,
    ):
        """
        Args:
            dim: Dimension of the input features.
            inner_dim: Dimension of queries, keys, and values.
            num_heads: Number of attention heads.
            positional_encoding: "rpb" for relative positional bias or "rope" for RoPE.
            coords_dim: Coordinate embedding dimension (required for "rpb").
            coords_inner_dim: Projected coordinate dimension (required for "rpb").
            dropout: Attention dropout rate.
        """
        super().__init__()
        self.inner_dim = inner_dim
        self.dropout_rate = dropout
        self.positional_encoding = positional_encoding

        self.f_qk_proj = nn.Sequential(nn.Linear(dim, inner_dim * 2), RMSNorm(inner_dim * 2))

        if positional_encoding == "rpb":
            assert coords_dim is not None and coords_inner_dim is not None, (
                "coords_dim and coords_inner_dim must be specified for relative positional bias"
            )
            self.c_qk_proj = nn.Sequential(
                nn.Linear(coords_dim, coords_inner_dim * 2), RMSNorm(coords_inner_dim * 2)
            )
            self.attention = SpatiotemporalFlashAttention(inner_dim, coords_inner_dim, num_heads)
        elif positional_encoding == "rope":
            self.attention = SpatiotemporalRoPEAttention(inner_dim, num_heads)
        else:
            raise ValueError(f"Unknown positional_encoding: {positional_encoding}")

        self.v_proj = nn.Linear(dim, inner_dim)
        self.proj_back = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: SourceEmbedding) -> Tensor:
        """
        Args:
            x: SourceEmbedding with:
                - embedding: (B, L_total, dim)
                - coords: (B, L_total, coords_dim)
        Returns:
            Tensor of shape (B, L_total, dim).
        """
        features, coords = x.embedding, x.coords

        f_qk = self.f_qk_proj(features)
        f_q, f_k = f_qk.chunk(2, dim=-1)

        if self.positional_encoding == "rpb":
            c_qk = self.c_qk_proj(coords)
        else:
            c_qk = coords.repeat(1, 1, 2)
        c_q, c_k = c_qk.chunk(2, dim=-1)

        v = self.v_proj(features)

        out = self.attention(
            f_q,
            f_k,
            c_q,
            c_k,
            v,
            dropout=self.dropout_rate if self.training else 0.0,
        )
        return self.proj_back(out)
