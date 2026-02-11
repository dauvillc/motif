"""Self-attention computed over each source independently using swin attention."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from motif.datatypes import MultisourceTensor, SourceEmbeddingDict
from motif.models.motif.small_layers import RMSNorm


class SeparateWindowedAttention(nn.Module):
    """Attention block that computes the attention over each source independently,
    using a spatial window over the tokens as in the Swin Transformer.

    Uses spatio-temporal coordinates as relative positional encoding in the attention weights.
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
        """
        Args:
            dim (int): Dimension of the input features.
            inner_dim (int): Dimension of the inner features used for computing queries and keys.
            coords_dim (int): Dimension of the coordinate embeddings.
            coords_inner_dim (int): Dimension of the inner coordinate embeddings used
                for computing queries and keys.
            num_heads (int): Number of heads in the attention block.
            dropout (float): Dropout rate.
            window_size (int): Number of tokens included in each window.
            shifted (bool): Whether to shift the windows by half the window size.
        """
        super().__init__()
        self.dim = dim
        self.inner_dim = inner_dim
        self.coords_dim = coords_dim
        self.coords_inner_dim = coords_inner_dim
        self.num_heads = num_heads
        if inner_dim % num_heads != 0:
            raise ValueError(
                f"inner_dim must be divisible by num_heads, got {inner_dim} and {num_heads}"
            )
        self.head_dim = inner_dim // num_heads
        self.window_size = window_size
        self.shifted = shifted

        # Projection to queries and keys for both features and coordinates.
        # For the queries and keys, we apply an RMSNorm to stabilize the training.
        self.f_qk_proj = nn.Sequential(
            nn.Linear(dim, self.inner_dim * 2), RMSNorm(self.inner_dim * 2)
        )
        self.c_qk_proj = nn.Sequential(
            nn.Linear(coords_dim, self.coords_inner_dim * 2),
            RMSNorm(self.coords_inner_dim * 2),
        )
        # Projection to values and back projection.
        self.v_proj = nn.Linear(dim, self.inner_dim)
        self.proj_back = nn.Sequential(nn.Linear(self.inner_dim, dim), nn.Dropout(dropout))

        # Parameter to weight the contribution of the coordinate-based attention scores
        # for each head.
        self.alpha = nn.Parameter(torch.ones((1, self.num_heads, 1, 1, 1, 1)))

        # Dropout layer for the attention map
        self.dropout = nn.Dropout(dropout)

        # Relative positional embeddings. We use one embedding per relative position in the window,
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
            # If the source is 0D or 1D, do nothing
            if len(spatial_dims) < 2:
                outputs[src] = x_src
                continue
            h, w = spatial_dims

            # Project the features and coordinates to queries, keys and values
            f_qkv = torch.cat([self.f_qk_proj(features), self.v_proj(features)], dim=-1)
            f_qkv = rearrange(f_qkv, "b h w (d k) -> b h w d k", k=3)
            c_qk = self.c_qk_proj(coords)
            c_qk = rearrange(c_qk, "b h w (d k) -> b h w d k", k=2)

            def pad_roll_reshape(qkv: Tensor, window_size: int):
                """Factorizes the treatment for both features and coords"""
                # Pad the vectors so that h and w are multiples of the window size
                pad_h = (self.window_size - h % self.window_size) % self.window_size
                pad_w = (self.window_size - w % self.window_size) % self.window_size
                qkv = F.pad(qkv, (0, 0, 0, 0, 0, pad_w, 0, pad_h))
                # Roll the windows if needed
                if self.shifted:
                    qkv = torch.roll(
                        qkv, (-self.window_size // 2, -self.window_size // 2), dims=(1, 2)
                    )
                # Reshape to windows and separate the heads
                qkv = rearrange(
                    qkv,
                    "b (Wh w1) (Ww w2) (e H) k -> b H Wh Ww (w1 w2) e k",
                    w1=self.window_size,
                    w2=self.window_size,
                    H=self.num_heads,
                )
                return qkv

            f_qkv = pad_roll_reshape(f_qkv, self.window_size)
            c_qk = pad_roll_reshape(c_qk, self.window_size)

            # Compute the queries, keys and values
            f_q, f_k, v = f_qkv.chunk(3, dim=6)
            f_q, f_k, v = f_q.squeeze(6), f_k.squeeze(6), v.squeeze(6)  # (b H Wh Ww w**2 d)
            c_q, c_k = c_qk.chunk(2, dim=6)
            c_q, c_k = c_q.squeeze(6), c_k.squeeze(6)  # (b H Wh Ww w**2 d_c)

            # Compute the attention weights
            dots = (f_q @ f_k.transpose(4, 5)) / self.head_dim**0.5 + self.alpha * (
                c_q @ c_k.transpose(4, 5)
            ) / self.coords_inner_dim**0.5
            # (b H Wh Ww w**2 w**2)

            # Add the positional embeddings
            dots += self.pos_embeddings[
                self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]
            ]

            # For shifted windows, compute the attention mask
            if self.shifted:
                row_mask = torch.zeros((self.window_size**2, self.window_size**2)).cuda()
                row_mask[
                    -self.window_size * (self.window_size // 2) :,
                    0 : -self.window_size * (self.window_size // 2),
                ] = float("-inf")
                row_mask[
                    0 : -self.window_size * (self.window_size // 2),
                    -self.window_size * (self.window_size // 2) :,
                ] = float("-inf")
                column_mask = rearrange(
                    row_mask,
                    "(r w1) (c w2) -> (w1 r) (w2 c)",
                    w1=self.window_size,
                    w2=self.window_size,
                )
                dots[:, :, -1, :] += row_mask
                dots[:, :, :, -1] += column_mask
            # Deduce the attention weights
            att_weights = F.softmax(dots, dim=-1)
            att_weights = self.dropout(att_weights)

            # Dot product to get the updated values
            y = att_weights @ v  # (b H Wh Ww w**2 d)
            # Reshape back to the original spatial layout of the tokens
            y = rearrange(
                y,
                "b H Wh Ww (w1 w2) e -> b (Wh w1) (Ww w2) (H e)",
                w1=self.window_size,
                w2=self.window_size,
                H=self.num_heads,
            )
            # Remove the padding
            y = y[:, :h, :w, :]
            outputs[src] = self.proj_back(y)

        return outputs
