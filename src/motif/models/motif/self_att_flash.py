"""Self-attention computed over each source independently using swin attention."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from motif.datatypes import MultisourceTensor, SourceEmbeddingDict
from motif.models.motif.flash_attention import SpatiotemporalFlashAttention
from motif.models.motif.small_layers import RMSNorm


class SeparateWindowedAttention(nn.Module):
    """Computes self-attention over each source independently, using Swin attention logic for each source.
    Uses spatio-temporal coordinates as relative positional encoding in the attention weights,
    and flash attention for memory efficiency.
    """

    def __init__(
        self,
        dim: int,
        inner_dim: int,
        coords_dim: int,
        coords_inner_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        window_size: int = 8,
        shifted: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.inner_dim = inner_dim
        self.coords_dim = coords_dim
        self.coords_inner_dim = coords_inner_dim
        self.window_size = window_size
        self.shifted = shifted
        self.dropout_rate = dropout

        # Projections
        self.f_qk_proj = nn.Sequential(
            nn.Linear(dim, self.inner_dim * 2), RMSNorm(self.inner_dim * 2)
        )
        self.c_qk_proj = nn.Sequential(
            nn.Linear(coords_dim, self.coords_inner_dim * 2),
            RMSNorm(self.coords_inner_dim * 2),
        )
        self.v_proj = nn.Linear(dim, self.inner_dim)
        self.proj_back = nn.Sequential(nn.Linear(self.inner_dim, dim), nn.Dropout(dropout))

        # Attention computation module
        self.attention = SpatiotemporalFlashAttention(
            self.inner_dim, self.coords_inner_dim, num_heads
        )

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

            # 1. Project Features and Coords
            # f_q, f_k, f_v shapes: (b, h, w, D_feat)
            f_qkv = torch.cat([self.f_qk_proj(features), self.v_proj(features)], dim=-1)
            f_qkv = rearrange(f_qkv, "b h w (D k) -> b h w D k", k=3)
            f_q, f_k, f_v = f_qkv.unbind(dim=-1)

            # c_q, c_k shapes: (b, h, w, D_coord)
            c_qk = self.c_qk_proj(coords)
            c_qk = rearrange(c_qk, "b h w (D k) -> b h w D k", k=2)
            c_q, c_k = c_qk.unbind(dim=-1)

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
