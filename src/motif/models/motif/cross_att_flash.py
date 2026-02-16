from math import prod

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from motif.datatypes import MultisourceTensor, SourceEmbeddingDict
from motif.models.motif.flash_attention import SpatiotemporalFlashAttention
from motif.models.motif.rope_attention import SpatiotemporalRoPEAttention
from motif.models.motif.small_layers import RMSNorm


class MultisourcesWindowedCrossAttention(nn.Module):
    """Computes attention across the sources using a windowed system.
    - For a given source $S$ and spatial window $u$, the features are projected to queries,
       keys and values.
    - Within each window $u$, the feature queries and keys are averaged over the spatial dims
        to get $Qf_u^S, Kf_u^S$.
    - The keys and queries are concatenated into cross-window, cross-source sequences
        $(Qf, Kf)$.
    - The attention weights A are computed following:
        $$A = \\text{softmax}\\left(\\frac{Qf Kf^T}{\\sqrt{Df}}\\right)$$
    - Within each window $u$, the features are concatenated along the channel dim
        to get the values $V_u^S$.
    - All windows' values are concatenated into a single sequence of vectors V.
    - The re-weighted values are computed as V' = A @ V.
    - V' is split back into windows, projected back to the original dimension,
        and summed back to the original features.

    The coordinates are used as positional encoding, in two possible ways:
    - Either as Relative Positional Bias (RPB), in which case they are projected
        to queries and keys and used to compute a bias matrix that is added to the attention
        scores before the softmax.
    - Or using RoPE, in which case the coordinates are used to compute rotation angles
        that are applied to the feature queries and keys before computing the attention scores.

    For 2D sources, the windows are square patches of size window_size x window_size.
    """

    def __init__(
        self,
        dim: int,
        inner_dim: int,
        window_size: int,
        num_heads: int,
        positional_encoding: str = "rpb",
        coords_dim: int | None = None,
        coords_inner_dim: int | None = None,
        inner_dim_v: int | None = None,
        mask_self_attention: bool = True,
        dropout: float = 0.0,
    ):
        """
        Args:
            dim (int): Dimension of the input features.
            inner_dim (int): Dimension of the inner features used for computing queries and keys.
            window_size (int): Size of the window for attention.
            num_heads (int): Number of attention heads.
            positional_encoding (str): Method for positional encoding in attention layers.
                Can be either "rope" for RoPE or "rpb" for relative positional bias.
            coords_dim (int | None): Dimension of the coordinate embeddings.
            coords_inner_dim (int | None): Dimension of the inner coordinate embeddings used
                for computing queries and keys.
            inner_dim_v (int, optional): Dimension of the inner features used for computing values.
                If None, defaults to inner_dim.
            mask_self_attention (bool, optional): Whether to mask out attention weights
                between elements of the same source.
            dropout (float, optional): Dropout rate for attention weights. Defaults to 0.0.
        """
        super().__init__()
        self.dim = dim
        self.inner_dim = inner_dim
        self.window_size = window_size
        if inner_dim_v is None:
            inner_dim_v = inner_dim
        self.inner_v_dim = inner_dim_v
        self.dropout = dropout
        self.mask_self_attention = mask_self_attention

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

        # Projection (compression) to values
        self.v_proj = nn.Linear(dim, inner_dim_v, bias=False)
        # Projection back to the original dimension
        self.v_back_proj = nn.Linear(inner_dim_v, dim, bias=False)

    def forward(self, inputs: SourceEmbeddingDict) -> MultisourceTensor:
        """
        Args:
            inputs: dict {src: SourceEmbedding}
                containing the input features and coordinates for each source.

        Returns:
            Dictionary {src: x_s} of re-weighted data tensors of shape (B, ..., dim).
        """
        f_keys, f_queries, values = {}, {}, {}
        c_keys, c_queries = {}, {}
        windowed_shapes, n_windows = {}, []
        for src, x_src in inputs.items():
            features, coords = x_src.embedding, x_src.coords
            _, *spatial_dims, _ = features.shape

            # First step: reshape each source into windows
            if len(spatial_dims) == 2:
                # Pad the spatial dimensions to be divisible by the window size
                pad_h = (self.window_size - spatial_dims[0] % self.window_size) % self.window_size
                pad_w = (self.window_size - spatial_dims[1] % self.window_size) % self.window_size
                features = F.pad(features, (0, 0, 0, pad_w, 0, pad_h))
                coords = F.pad(coords, (0, 0, 0, pad_w, 0, pad_h))
                # Reshape to windows.
                features = rearrange(
                    features,
                    "b (Wh w1) (Ww w2) d -> b Wh Ww (w1 w2) d",
                    w1=self.window_size,
                    w2=self.window_size,
                )
                coords = rearrange(
                    coords,
                    "b (Wh w1) (Ww w2) d -> b Wh Ww (w1 w2) d",
                    w1=self.window_size,
                    w2=self.window_size,
                )
            else:
                raise NotImplementedError("Only 2D sources are supported.")

            # Average the vectors within each window
            f_avg = features.mean(dim=-2)  # (b, Wh, Ww, Dv)
            c_avg = coords.mean(dim=-2)  # (b, Wh, Ww, Dc)
            # Store the shape of the windowed source and the number of windows
            windowed_shapes[src] = features.shape[1:3]
            n_windows.append(prod(windowed_shapes[src]))

            # Project the features to queries and keys
            f_qk: Tensor = self.f_qk_proj(f_avg)  # (b, Wh, Ww, 2 * Dqk)
            # If using RPB, project the coords to queries and keys as well.
            if self.positional_encoding == "rpb":
                c_qk: Tensor = self.c_qk_proj(c_avg)  # (b, Wh, Ww, 2 * Dqk_c)
            else:
                # If using RoPE, the coords dim remains 3 (lat,lon,time),
                # but we still duplicate the last dim to then split into queries and keys.
                c_qk = c_avg.repeat(1, 1, 1, 2)  # (b, Wh, Ww, 2 * Dc)

            # Regroup the windows into a single sequence
            f_qk = rearrange(f_qk, "b Wh Ww Df -> b (Wh Ww) Df")
            c_qk = rearrange(c_qk, "b Wh Ww Dc -> b (Wh Ww) Dc")

            # Split into queries and keys
            f_queries[src], f_keys[src] = f_qk.chunk(2, dim=-1)
            c_queries[src], c_keys[src] = c_qk.chunk(2, dim=-1)

            # Project to values and stack the vectors of each window along the
            # feature dimension to form a single vector per window.
            v = self.v_proj(features)  # (b, Wh, Ww, w1*w2*Dv')
            values[src] = rearrange(v, "b Wh Ww n Df -> b (Wh Ww) (n Df)")

        # Concatenate all sequences across sources
        f_q = torch.cat(list(f_queries.values()), dim=-2)  # (B, H, N, Df)
        f_k = torch.cat(list(f_keys.values()), dim=-2)  # (B, H, N, Df)
        c_q = torch.cat(list(c_queries.values()), dim=-2)  # (B, H, N, Dc)
        c_k = torch.cat(list(c_keys.values()), dim=-2)  # (B, H, N, Dc)
        values = torch.cat(list(values.values()), dim=-2)  # (B, H, N, Df * n)

        if self.mask_self_attention:
            # For each source, the attention weights contain a block centered on the diagonal that
            # correspond to key/query pairs from the same source. In order to prioritize attention
            # between different sources only in this layer, we'll mask out those blocks.
            # Note: torch SDPA format: True means attend, False means mask out.
            blocks = [torch.full((n, n), True) for n in n_windows]
            attn_mask = ~torch.block_diag(*blocks)  # (N, N)
            # Add a batch dim to make the mask broadcastable with the attention scores
            attn_mask = rearrange(attn_mask, "n1 n2 -> 1 n1 n2")
            attn_mask = attn_mask.to(values.device)
        else:
            attn_mask = None

        # Compute the attention scores. The layer takes care of splitting heads and using the
        # coordinates as bias.
        v_out = self.attention.forward(
            f_q,
            f_k,
            c_q,
            c_k,
            values,
            attn_mask=attn_mask,
            dropout=self.dropout if self.training else 0.0,
        )

        # Project back to the original dimension
        v_out = rearrange(
            v_out,
            "b N (n Df) -> b N n Df",
            n=self.window_size**2,
        )
        features_out = self.v_back_proj(v_out)  # (B, N, n, Df)
        # Split back into the sources
        out_split = torch.split(features_out, n_windows, dim=1)

        # Re-insert the updated embeddings back into the windows
        outputs = {}
        for i, (source, (Wh, Ww)) in enumerate(windowed_shapes.items()):
            out = rearrange(
                out_split[i],
                "b (Wh Ww) (w1 w2) Df -> b (Wh w1) (Ww w2) Df",
                Wh=Wh,
                Ww=Ww,
                w1=self.window_size,
                w2=self.window_size,
            )
            # Remove the padding if it was added
            h, w = inputs[source].embedding.shape[1:3]
            out = out[:, :h, :w, :]
            outputs[source] = out
        return outputs
