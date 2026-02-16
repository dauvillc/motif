"""Self-attention computed over each source independently using swin attention."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from motif.datatypes import MultisourceTensor, SourceEmbeddingDict
from motif.models.motif.flash_attention import SpatiotemporalFlashAttention
from motif.models.motif.rope_attention import SpatiotemporalRoPEAttention
from motif.models.motif.small_layers import RMSNorm


class SeparateWindowedAttention(nn.Module):
    """Computes self-attention over each source independently, using Swin attention
    logic for each source.
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
        window_size: int = 8,
        shifted: bool = False,
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
            window_size (int): Size of the window for attention.
            shifted (bool, optional): Whether to use shifted windows as in Swin attention.
            dropout (float, optional): Dropout rate for attention weights. Defaults to 0.0.
        """
        super().__init__()
        self.dim = dim
        self.inner_dim = inner_dim
        self.window_size = window_size
        self.shifted = shifted
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

        # Relative positional embeddings
        self.pos_embeddings = nn.Parameter(torch.randn(window_size * 2 - 1, window_size * 2 - 1))
        self.indices = torch.tensor(
            np.array([[x, y] for x in range(window_size) for y in range(window_size)])
        )
        self.relative_indices = self.indices[None, :, :] - self.indices[:, None, :]
        self.relative_indices += self.window_size - 1

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
            h, w = spatial_dims
            device = features.device

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

            # 4. Window Partitioning (Standard Swin logic)
            pad_h = (self.window_size - h % self.window_size) % self.window_size
            pad_w = (self.window_size - w % self.window_size) % self.window_size

            def window_transform(t):
                """From (B, h, w, D) to (B, Wh, Ww, w^2, D) where Wh is the number of windows
                along the vertical dim, Ww is the number of windows along the horizontal dim,
                and w^2 is the number of elements in each window."""
                t = F.pad(t, (0, 0, 0, pad_w, 0, pad_h))
                if self.shifted:
                    t = torch.roll(t, (-self.window_size // 2, -self.window_size // 2), dims=(1, 2))
                return rearrange(
                    t,
                    "b (Wh w1) (Ww w2) D -> b Wh Ww (w1 w2) D",
                    w1=self.window_size,
                    w2=self.window_size,
                )

            # From (B, h, w, D) to (B, Wh, Ww, w^2, D)
            f_q_win = window_transform(f_q)
            f_k_win = window_transform(f_k)
            c_q_win = window_transform(c_q)
            c_k_win = window_transform(c_k)
            f_v_win = window_transform(f_v)

            # 5. Bias mask: we build a float mask that includes both the relative
            # positional bias and the swin shift mask (using -inf float values).

            # Positional bias
            pos_bias = self.pos_embeddings[
                self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]
            ]  # (w^2, w^2)
            attn_bias = pos_bias.to(device)
            # Make it broadcastable with the attention weights
            attn_bias = rearrange(attn_bias, "w1 w2 -> 1 1 1 w1 w2")

            # Shift mask (as usual in Swin attention)
            if self.shifted:
                H_pad, W_pad = h + pad_h, w + pad_w
                Wh, Ww = H_pad // self.window_size, W_pad // self.window_size
                mask = torch.zeros(
                    (Wh, Ww, self.window_size**2, self.window_size**2), device=device
                )

                s = self.window_size
                row_mask = torch.zeros((s**2, s**2), device=device)
                row_mask[-s * (s // 2) :, : -s * (s // 2)] = float("-inf")
                row_mask[: -s * (s // 2), -s * (s // 2) :] = float("-inf")
                col_mask = rearrange(row_mask, "(r w1) (c w2) -> (w1 r) (w2 c)", w1=s, w2=s)

                mask[:, -1, :, :] += row_mask
                mask[-1, :, :, :] += col_mask

                attn_bias = attn_bias + mask

            # 6. Compute the attention. The module takes care of splitting the heads,
            # and using the coordinates as bias.
            out = self.attention(
                f_q_win,
                f_k_win,
                c_q_win,
                c_k_win,
                f_v_win,
                attn_mask=attn_bias,
                dropout=self.dropout_rate if self.training else 0.0,
            )

            # 7. Reshape Back
            out = rearrange(
                out,
                "b Wh Ww (w1 w2) D -> b (Wh w1) (Ww w2) D",
                w1=self.window_size,
                w2=self.window_size,
            )
            if self.shifted:
                out = torch.roll(out, (self.window_size // 2, self.window_size // 2), dims=(1, 2))

            # Remove padding that was added for the window partitioning
            out = out[:, :h, :w, :]
            outputs[src] = self.proj_back(out)

        return outputs
