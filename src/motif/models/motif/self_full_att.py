"""Self-attention computed over each source independently using full attention."""

import torch
import torch.nn as nn
from einops import rearrange

from motif.datatypes import MultisourceTensor, SourceEmbeddingDict
from motif.models.motif.flash_attention import SpatiotemporalFlashAttention
from motif.models.motif.rope_attention import SpatiotemporalRoPEAttention
from motif.models.motif.small_layers import RMSNorm


class SeparateAttention(nn.Module):
    """Computes self-attention over each source independently.
    Uses spatio-temporal coordinates as positional encoding,
    and flash attention for memory efficiency.
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
            dim (int): Dimension of the input features.
            inner_dim (int): Dimension of the inner features used for computing queries and keys.
            num_heads (int): Number of attention heads.
            positional_encoding (str): Method for positional encoding in attention layers.
                Can be either "rope" for RoPE or "rpb" for relative positional bias.
            coords_dim (int | None): Dimension of the coordinate embeddings.
            coords_inner_dim (int | None): Dimension of the inner coordinate embeddings used
                for computing queries and keys.
            dropout (float, optional): Dropout rate for attention weights. Defaults to 0.0.
        """
        super().__init__()
        self.dim = dim
        self.inner_dim = inner_dim
        self.dropout_rate = dropout
        self.positional_encoding = positional_encoding

        # Projections to queries and keys for the features
        self.f_qk_proj = nn.Sequential(nn.Linear(dim, inner_dim * 2), RMSNorm(inner_dim * 2))

        # The coords will only be projected to keys and queries if we use relative positional bias.
        if positional_encoding == "rpb":
            assert coords_dim is not None and coords_inner_dim is not None, (
                "coords_dim and coords_inner_dim must be specified for relative positional bias"
            )
            self.c_qk_proj = nn.Sequential(
                nn.Linear(coords_dim, coords_inner_dim * 2), RMSNorm(coords_inner_dim * 2)
            )
            self.attention = SpatiotemporalFlashAttention(
                self.inner_dim, coords_inner_dim, num_heads
            )
        elif positional_encoding == "rope":
            self.attention = SpatiotemporalRoPEAttention(self.inner_dim, num_heads)

        # Projection to values
        self.v_proj = nn.Linear(dim, self.inner_dim)
        self.proj_back = nn.Sequential(nn.Linear(self.inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: SourceEmbeddingDict) -> MultisourceTensor:
        """
        Args:
            inputs: dict {src: SourceEmbedding}. A SourceEmbedding includes notably:
                - embedding: the embedded data of shape (B, h, w, dim)
                - coords: the embedded coordinates of shape (B, h, w, coords_dim)

        Returns:
            Dictionary {(src): x_s} of re-weighted data tensors of shape (B, ..., dim).
        """
        outputs = {}
        for src, x_src in x.items():
            features, coords = x_src.embedding, x_src.coords
            spatial_dims = features.shape[1:-1]
            if len(spatial_dims) < 2:
                outputs[src] = x_src
                continue

            # 1. Project Features
            # f_q, f_k, f_v shapes: (b, h, w, D_feat)
            f_qkv = torch.cat([self.f_qk_proj(features), self.v_proj(features)], dim=-1)
            f_qkv = rearrange(f_qkv, "b h w (D k) -> b h w D k", k=3)
            f_q, f_k, f_v = f_qkv.unbind(dim=-1)

            # - We only project the coords to qk if using relative positional bias.
            if self.positional_encoding == "rpb":
                c_qk = self.c_qk_proj(coords)
            else:
                # If using RoPE, keep the coords as (lat/lon/time).
                # Duplicate the last dim to then into queries and keys.
                c_qk = coords.repeat(1, 1, 1, 2)
            c_qk = rearrange(c_qk, "b h w (D k) -> b h w D k", k=2)
            c_q, c_k = c_qk.unbind(dim=-1)  # c_q, c_k shapes: (b, h, w, D_coord)

            # 6. Compute the attention. The module takes care of splitting the heads,
            # and using the coordinates as bias.
            out = self.attention(
                f_q,
                f_k,
                c_q,
                c_k,
                f_v,
                attn_mask=None,
                dropout=self.dropout_rate if self.training else 0.0,
            )
            outputs[src] = self.proj_back(out)

        return outputs
