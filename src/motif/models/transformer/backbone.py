"""Implements a full-sequence transformer backbone for multi-source geospatial data."""

from functools import partial
from math import prod
from typing import Callable, cast

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from motif.datatypes import MultisourceTensor, SourceEmbedding, SourceEmbeddingDict, SourceIndex
from motif.models.acn import SequenceAdaptiveConditionalNormalization
from motif.models.transformer.full_sequence_att import FullSequenceAttention


class SequenceFeedForward(nn.Module):
    """Feed-forward network applied to a flat merged SourceEmbedding.

    Equivalent to the per-source FeedForward but takes a SourceEmbedding and
    returns a Tensor, matching the interface expected by
    SequenceAdaptiveConditionalNormalization.
    """

    def __init__(self, dim: int, inner_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        inner_dim = int(dim * inner_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: SourceEmbedding) -> Tensor:
        return self.net(x.embedding)


class TransformerBackbone(nn.Module):
    """Full-sequence transformer backbone for multi-source geospatial data.

    At backbone entry all source tokens are flattened and concatenated into a
    single sequence (B, L_total, dim).  Each block applies:
      1. Full spatio-temporal self-attention over the entire merged sequence;
      2. Position-wise feed-forward network.
    Both layers are wrapped in SequenceAdaptiveConditionalNormalization (DiT-style).
    At backbone exit the merged sequence is split and reshaped back to the
    original per-source spatial shapes.

    Input/output contract is identical to MultisourceGeneralBackbone.
    """

    def __init__(
        self,
        n_blocks: int,
        dim: int,
        att_inner_dim: int,
        positional_encoding: str = "rpb",
        coords_dim: int | None = None,
        coords_inner_dim: int | None = None,
        cond_dim: int | None = None,
        num_heads: int = 8,
        mlp_inner_ratio: float = 2.0,
        dropout: float = 0.0,
        use_checkpointing: bool = False,
    ):
        """
        Args:
            n_blocks: Number of transformer blocks.
            dim: Embedding dimension.
            att_inner_dim: Inner dimension for attention queries/keys/values.
            positional_encoding: "rpb" for relative positional bias or "rope" for RoPE.
            coords_dim: Coordinate embedding dimension (required for "rpb").
            coords_inner_dim: Projected coordinate dimension (required for "rpb").
            cond_dim: Conditioning vector dimension. Defaults to dim.
            num_heads: Number of attention heads.
            mlp_inner_ratio: Expansion ratio for the feed-forward hidden layer.
            dropout: Dropout rate applied in attention and feed-forward layers.
            use_checkpointing: Use gradient checkpointing to save memory.
        """
        super().__init__()
        cond_dim = cond_dim if cond_dim is not None else dim
        self.dim, self.cond_dim = dim, cond_dim

        if positional_encoding == "rpb":
            assert coords_dim is not None and coords_inner_dim is not None, (
                "coords_dim and coords_inner_dim must be specified for relative positional bias"
            )
        elif positional_encoding != "rope":
            raise ValueError(f"Unknown positional_encoding: {positional_encoding}")

        if use_checkpointing:
            self.ckpt = cast(
                Callable[..., SourceEmbedding],
                partial(checkpoint, use_reentrant=False),
            )
        else:
            self.ckpt = cast(Callable[..., SourceEmbedding], lambda fn, x: fn(x))

        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            block = nn.ModuleList(
                [
                    SequenceAdaptiveConditionalNormalization(
                        FullSequenceAttention(
                            dim,
                            att_inner_dim,
                            num_heads=num_heads,
                            positional_encoding=positional_encoding,
                            coords_dim=coords_dim,
                            coords_inner_dim=coords_inner_dim,
                            dropout=dropout,
                        ),
                        dim,
                        cond_dim,
                    ),
                    SequenceAdaptiveConditionalNormalization(
                        SequenceFeedForward(dim, inner_ratio=mlp_inner_ratio, dropout=dropout),
                        dim,
                        cond_dim,
                    ),
                ]
            )
            self.blocks.append(block)

    def _merge(
        self, x: SourceEmbeddingDict
    ) -> tuple[SourceEmbedding, list[int], list[tuple[SourceIndex, list[int]]]]:
        """Flatten and concatenate all source SourceEmbeddings into one sequence.

        Returns:
            merged: SourceEmbedding with tensors of shape (B, L_total, *).
            split_sizes: Number of tokens per source, used to split at exit.
            meta: List of (SourceIndex, spatial_shape) for reconstruction.
        """
        embeddings, coords_list, conds = [], [], []
        split_sizes: list[int] = []
        meta: list[tuple[SourceIndex, list[int]]] = []
        has_cond = any(v.conditioning is not None for v in x.values())

        for src, data in x.items():
            spatial_shape = list(data.embedding.shape[1:-1])
            L = prod(spatial_shape)
            embeddings.append(rearrange(data.embedding, "B ... D -> B (...) D"))
            coords_list.append(rearrange(data.coords, "B ... D -> B (...) D"))
            if has_cond:
                assert data.conditioning is not None, (
                    f"Source {src} is missing conditioning while others have it"
                )
                conds.append(rearrange(data.conditioning, "B ... D -> B (...) D"))
            split_sizes.append(L)
            meta.append((src, spatial_shape))

        merged_emb = torch.cat(embeddings, dim=1)
        merged_coords = torch.cat(coords_list, dim=1)
        merged_cond = torch.cat(conds, dim=1) if has_cond else None

        return SourceEmbedding(
            embedding=merged_emb, coords=merged_coords, conditioning=merged_cond
        ), split_sizes, meta

    def _split(
        self,
        merged: SourceEmbedding,
        split_sizes: list[int],
        meta: list[tuple[SourceIndex, list[int]]],
    ) -> MultisourceTensor:
        """Split the merged sequence back into per-source spatial tensors."""
        chunks: tuple[Tensor, ...] = torch.split(merged.embedding, split_sizes, dim=1)
        outputs: MultisourceTensor = {}
        for (src, spatial_shape), chunk in zip(meta, chunks):
            # chunk: (B, L_s, dim) → (B, *spatial_shape, dim)
            outputs[src] = chunk.view(chunk.shape[0], *spatial_shape, chunk.shape[-1])
        return outputs

    def forward(self, x: SourceEmbeddingDict) -> MultisourceTensor:
        """
        Args:
            x: Dict {(source_name, index): SourceEmbedding} with embeddings of
                shape (B, h, w, dim).
        Returns:
            Dict {(source_name, index): Tensor} of shape (B, h, w, dim).
        """
        merged, split_sizes, meta = self._merge(x)

        for block in self.blocks:
            for layer in cast(nn.ModuleList, block):
                merged = self.ckpt(layer, merged)

        return self._split(merged, split_sizes, meta)
