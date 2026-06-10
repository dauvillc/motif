"""Implements anchor-point-based cross-source attention."""

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from motif.datatypes import MultisourceTensor, SourceEmbeddingDict
from motif.models.flash_attention import SpatiotemporalFlashAttention
from motif.models.motif.rope_attention import SpatiotemporalRoPEAttention
from motif.models.motif.small_layers import RMSNorm


class MultisourcesAnchoredCrossAttention(nn.Module):
    """Computes cross-source attention using an anchor points system.

    For each source a regular subgrid of anchor tokens is selected. All anchors
    from all sources are concatenated into a single sequence on which attention is
    computed. The resulting updates are scatter-added back to the anchor positions
    of the original feature maps.

    Positional encoding follows the same two-mode convention as the rest of the
    backbone:
    - "rpb": learned projections for both feature and coordinate Q/K pairs, fed to
      SpatiotemporalFlashAttention.
    - "rope": raw (lat, lon, time) coordinates fed to SpatiotemporalRoPEAttention.
    """

    def __init__(
        self,
        dim: int,
        inner_dim: int,
        anchor_points_spacing: int,
        num_heads: int = 8,
        positional_encoding: str = "rpb",
        coords_dim: int | None = None,
        coords_inner_dim: int | None = None,
        mask_self_attention: bool = True,
        dropout: float = 0.0,
    ):
        """
        Args:
            dim: Dimension of the input feature embeddings.
            inner_dim: Dimension used for Q/K/V projections.
            anchor_points_spacing: Step between anchor grid points along each spatial
                axis. E.g. 4 selects every 4th row and column.
            num_heads: Number of attention heads.
            positional_encoding: "rpb" (relative positional bias) or "rope".
            coords_dim: Dimension of the coordinate embeddings. Required for "rpb".
            coords_inner_dim: Inner dimension for coordinate Q/K projections. Required
                for "rpb".
            mask_self_attention: If True, attention between anchors of the same source
                is masked out so the layer focuses on cross-source interactions.
            dropout: Dropout rate applied to attention weights during training.
        """
        super().__init__()
        self.dim = dim
        self.inner_dim = inner_dim
        self.anchor_points_spacing = anchor_points_spacing
        self.mask_self_attention = mask_self_attention
        self.dropout = dropout
        self.positional_encoding = positional_encoding

        self.f_qk_proj = nn.Sequential(nn.Linear(dim, inner_dim * 2), RMSNorm(inner_dim * 2))

        if positional_encoding == "rpb":
            assert coords_dim is not None and coords_inner_dim is not None, (
                "coords_dim and coords_inner_dim must be specified for positional_encoding='rpb'"
            )
            self.c_qk_proj = nn.Sequential(
                nn.Linear(coords_dim, coords_inner_dim * 2), RMSNorm(coords_inner_dim * 2)
            )
            self.attention = SpatiotemporalFlashAttention(inner_dim, coords_inner_dim, num_heads)
        elif positional_encoding == "rope":
            self.attention = SpatiotemporalRoPEAttention(inner_dim, num_heads)
        else:
            raise ValueError(f"Unknown positional_encoding: {positional_encoding}")

        self.v_proj = nn.Linear(dim, inner_dim, bias=False)
        self.v_back_proj = nn.Linear(inner_dim, dim, bias=False)

    def _anchor_indices(self, size: int) -> Tensor:
        """Return centered anchor indices along one spatial axis."""
        indices = torch.arange(0, size, self.anchor_points_spacing)
        indices = indices + ((size - 1) % self.anchor_points_spacing) // 2
        return indices

    def forward(self, inputs: SourceEmbeddingDict) -> MultisourceTensor:
        """
        Args:
            inputs: {src: SourceEmbedding} with embedding (B, ..., dim) and
                coords (B, ..., coords_dim).

        Returns:
            {src: Tensor} of shape (B, ..., dim) — original embeddings with anchor
            positions updated by the cross-source attention output.
        """
        f_queries, f_keys = {}, {}
        c_queries, c_keys = {}, {}
        values: dict = {}
        anchor_indices: dict = {}
        n_anchors_list: list[int] = []

        for src, x_src in inputs.items():
            embedding, coords = x_src.embedding, x_src.coords
            spatial_dims = embedding.shape[1:-1]

            if len(spatial_dims) == 0:
                # 0D source: single token is the anchor
                anc_emb = embedding.unsqueeze(1)  # (B, 1, dim)
                anc_coords = coords.unsqueeze(1)   # (B, 1, coords_dim)
                n_anchors = 1

            elif len(spatial_dims) == 2:
                h, w = spatial_dims
                rows = self._anchor_indices(h).to(embedding.device)
                cols = self._anchor_indices(w).to(embedding.device)
                anc_emb = embedding[:, rows[:, None], cols]          # (B, n_r, n_c, dim)
                anc_emb = rearrange(anc_emb, "b r c d -> b (r c) d")
                anc_coords = coords[:, rows[:, None], cols]          # (B, n_r, n_c, coords_dim)
                anc_coords = rearrange(anc_coords, "b r c d -> b (r c) d")
                anchor_indices[src] = (rows, cols)
                n_anchors = len(rows) * len(cols)

            else:
                raise NotImplementedError(
                    f"Only 0D and 2D sources are supported, got spatial_dims={spatial_dims}"
                )

            n_anchors_list.append(n_anchors)

            # Feature Q/K projections
            f_qk: Tensor = self.f_qk_proj(anc_emb)  # (B, N, 2*inner_dim)
            f_q, f_k = f_qk.chunk(2, dim=-1)
            f_queries[src], f_keys[src] = f_q, f_k

            # Coordinate Q/K projections
            if self.positional_encoding == "rpb":
                c_qk: Tensor = self.c_qk_proj(anc_coords)  # (B, N, 2*coords_inner_dim)
            else:
                # Duplicate raw coords so we can split into q/k
                c_qk = anc_coords.repeat(1, 1, 2)
            c_q, c_k = c_qk.chunk(2, dim=-1)
            c_queries[src], c_keys[src] = c_q, c_k

            # Value projection
            values[src] = self.v_proj(anc_emb)  # (B, N, inner_dim)

        # Concatenate all sources into one sequence
        f_q_cat = torch.cat(list(f_queries.values()), dim=1)
        f_k_cat = torch.cat(list(f_keys.values()), dim=1)
        c_q_cat = torch.cat(list(c_queries.values()), dim=1)
        c_k_cat = torch.cat(list(c_keys.values()), dim=1)
        v_cat = torch.cat(list(values.values()), dim=1)

        if self.mask_self_attention:
            # Block-diagonal True regions mark same-source pairs; we want to mask those out.
            blocks = [torch.full((n, n), True) for n in n_anchors_list]
            attn_mask = ~torch.block_diag(*blocks)  # (N_total, N_total)
            attn_mask = rearrange(attn_mask, "n1 n2 -> 1 n1 n2").to(v_cat.device)
        else:
            attn_mask = None

        v_out = self.attention.forward(
            f_q_cat,
            f_k_cat,
            c_q_cat,
            c_k_cat,
            v_cat,
            attn_mask=attn_mask,
            dropout=self.dropout if self.training else 0.0,
        )  # (B, N_total, inner_dim)

        v_out = self.v_back_proj(v_out)  # (B, N_total, dim)

        # Split back to sources and scatter-add to the full feature maps
        v_out_split = torch.split(v_out, n_anchors_list, dim=1)
        outputs: MultisourceTensor = {}
        for i, (src, x_src) in enumerate(inputs.items()):
            spatial_dims = x_src.embedding.shape[1:-1]
            out = x_src.embedding.clone()

            if len(spatial_dims) == 0:
                out += v_out_split[i].squeeze(1)

            elif len(spatial_dims) == 2:
                rows, cols = anchor_indices[src]
                n_rows = len(rows)
                delta = rearrange(v_out_split[i], "b (r c) d -> b r c d", r=n_rows)
                out[:, rows[:, None], cols] += delta

            outputs[src] = out

        return outputs
