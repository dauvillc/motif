"""Implements a multi-source backbone using anchor-point cross-attention."""

from functools import partial
from typing import Callable, cast

import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from motif.datatypes import MultisourceTensor, SourceEmbedding, SourceEmbeddingDict
from motif.models.motif.cross_att_anchor import MultisourcesAnchoredCrossAttention
from motif.models.motif.patch_merging import (
    MultiSourcePatchMerging,
    MultiSourcePatchSplitting,
)
from motif.models.motif.self_att_flash import SeparateWindowedAttention
from motif.models.motif.small_layers import FeedForward


class MultisourceAnchorBackbone(nn.Module):
    """Multi-source backbone that uses anchor-point cross-attention.

    Each block consists of three layers:
    1. Anchor-based cross-source attention;
    2. Individual windowed self-attention (IWSA);
    3. Feed-forward network.
    Each layer is wrapped in an adaptive conditional normalization module, as in DiT.
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
        anchor_points_spacing: int = 4,
        iwsa_window_size: int = 8,
        mlp_inner_ratio: int = 2,
        dropout: float = 0.0,
        downsample_input: bool = False,
        use_checkpointing: bool = False,
    ):
        """
        Args:
            n_blocks: Number of blocks in the backbone.
            dim: Embedding dimension.
            att_inner_dim: Inner dimension for attention Q/K/V projections.
            positional_encoding: "rpb" (relative positional bias) or "rope".
            coords_dim: Dimension of coordinate embeddings. Required for "rpb".
            coords_inner_dim: Inner dimension for coordinate Q/K projections.
                Required for "rpb".
            cond_dim: Dimension of the conditioning vector. Defaults to dim.
            num_heads: Number of attention heads.
            anchor_points_spacing: Spacing between anchor grid points for
                cross-source attention.
            iwsa_window_size: Window size for the individual windowed self-attention.
            mlp_inner_ratio: Expansion ratio for the feed-forward hidden layer.
            dropout: Dropout rate.
            downsample_input: If True, inputs are downsampled 2× before the blocks
                and upsampled back at the end.
            use_checkpointing: If True, use gradient checkpointing to trade compute
                for memory.
        """
        super().__init__()
        cond_dim = cond_dim if cond_dim is not None else dim
        self.dim, self.cond_dim = dim, cond_dim
        if positional_encoding == "rpb":
            assert coords_dim is not None and coords_inner_dim is not None, (
                "coords_dim and coords_inner_dim must be specified for relative positional bias"
            )
            self.coords_dim = coords_dim
            self.coords_inner_dim = coords_inner_dim
        elif positional_encoding == "rope":
            self.coords_dim = 0
            self.coords_inner_dim = 0
        else:
            raise ValueError(f"Unknown positional_encoding: {positional_encoding}")

        if use_checkpointing:
            self.ckpt = cast(
                Callable[..., SourceEmbeddingDict],
                partial(checkpoint, use_reentrant=False),
            )
        else:
            self.ckpt = cast(Callable[..., SourceEmbeddingDict], lambda fn, x: fn(x))

        if downsample_input:
            self.input_downsampling = MultiSourcePatchMerging(dim, cond_dim)
            dim *= 2
            cond_dim *= 2
            self.input_upsampling = MultiSourcePatchSplitting(dim)

        self.blocks = nn.ModuleList()
        for block_idx in range(n_blocks):
            shifted = bool(block_idx % 2)
            block = nn.ModuleList(
                [
                    AdapativeConditionalNormalization(
                        MultisourcesAnchoredCrossAttention(
                            dim,
                            att_inner_dim,
                            anchor_points_spacing=anchor_points_spacing,
                            num_heads=num_heads,
                            positional_encoding=positional_encoding,
                            coords_dim=coords_dim,
                            coords_inner_dim=coords_inner_dim,
                            dropout=dropout,
                        ),
                        dim,
                        cond_dim,
                    ),
                    AdapativeConditionalNormalization(
                        SeparateWindowedAttention(
                            dim,
                            att_inner_dim,
                            num_heads=num_heads,
                            positional_encoding=positional_encoding,
                            coords_dim=coords_dim,
                            coords_inner_dim=coords_inner_dim,
                            window_size=iwsa_window_size,
                            shifted=shifted,
                            dropout=dropout,
                        ),
                        dim,
                        cond_dim,
                    ),
                    AdapativeConditionalNormalization(
                        FeedForward(
                            dim,
                            inner_ratio=mlp_inner_ratio,
                            dropout=dropout,
                        ),
                        dim,
                        cond_dim,
                    ),
                ]
            )
            self.blocks.append(block)

    def forward(self, x: SourceEmbeddingDict) -> MultisourceTensor:
        """
        Args:
            x: {src: SourceEmbedding} with embedding (B, ..., dim) and optional
                conditioning (B, ..., cond_dim).

        Returns:
            {src: Tensor} of shape (B, ..., dim).
        """
        shapes_before_ds = {}
        if hasattr(self, "input_downsampling"):
            shapes_before_ds = {src: x[src].embedding.shape for src in x.keys()}
            x = self.ckpt(self.input_downsampling, x)

        for block in self.blocks:
            for layer in cast(nn.ModuleList, block):
                x = self.ckpt(layer, x)

        if hasattr(self, "input_upsampling"):
            x = self.ckpt(self.input_upsampling, x)
            for src in x.keys():
                if src in shapes_before_ds:
                    original_shape = shapes_before_ds[src]
                    x[src].embedding = x[src].embedding[
                        (slice(None),) + tuple(slice(0, s) for s in original_shape[1:])
                    ]

        return {src: x[src].embedding for src in x.keys()}


class AdapativeConditionalNormalization(nn.Module):
    """Wraps a module with adaptive conditional normalization (DiT-style) and a
    residual connection. If no conditioning is present the layer reduces to a
    plain pre-norm residual block."""

    def __init__(self, module: nn.Module, dim: int, cond_dim: int):
        super().__init__()
        self.module = module
        self.cond_dim = cond_dim
        self.norm = nn.LayerNorm(dim)
        self.cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(self.cond_dim, dim * 3))
        nn.init.zeros_(cast(Tensor, self.cond_proj[1].weight))
        nn.init.zeros_(cast(Tensor, self.cond_proj[1].bias))

    def forward(self, data: SourceEmbeddingDict, *args, **kwargs) -> SourceEmbeddingDict:
        skips, gates = {}, {}
        modulated_x: SourceEmbeddingDict = {}

        for src, source_data in data.items():
            skip = source_data.embedding
            x = self.norm(skip)
            skips[src] = skip

            cond = source_data.conditioning
            if cond is not None:
                shift, scale, gate = self.cond_proj(cond).chunk(3, dim=-1)
                x = x * (scale + 1) + shift
                gates[src] = gate

            modulated_x[src] = SourceEmbedding(
                embedding=x, coords=source_data.coords, conditioning=source_data.conditioning
            )

        module_output: MultisourceTensor = self.module(modulated_x, *args, **kwargs)

        outputs: SourceEmbeddingDict = {}
        for src, source_output in module_output.items():
            if src in gates:
                source_output = source_output * gates[src] + skips[src]
            else:
                source_output = source_output + skips[src]
            outputs[src] = SourceEmbedding(
                embedding=source_output,
                coords=data[src].coords,
                conditioning=data[src].conditioning,
            )
        return outputs
