"""Implements a multi-source MoTiF backbone."""

from functools import partial
from typing import Callable, cast

import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from motif.datatypes import MultisourceTensor, SourceEmbedding, SourceEmbeddingDict
from motif.models.motif.cross_att_flash import MultisourcesWindowedCrossAttention
from motif.models.motif.patch_merging import (
    MultiSourcePatchMerging,
    MultiSourcePatchSplitting,
)
from motif.models.motif.self_att_flash import SeparateWindowedAttention
from motif.models.motif.small_layers import FeedForward


class MultisourceGeneralBackbone(nn.Module):
    """General backbone for multi-source geospatial data.
    Each block consists of three layers:
    1. Cross-source attention;
    2. Individual windowed attention;
    3. Feed-forward network.
    Each layer is wrapped in an adaptive conditional normalization module, as in DiT.
    """

    def __init__(
        self,
        n_blocks: int,
        dim: int,
        att_inner_dim: int,
        positional_encoding: str = "rope",
        coords_dim: int | None = None,
        coords_inner_dim: int | None = None,
        cond_dim: int | None = None,
        num_heads: int = 8,
        cross_att_window_size: int = 4,
        iwsa_window_size: int = 8,
        mlp_inner_ratio: int = 2,
        dropout: float = 0.0,
        downsample_input: bool = False,
        use_checkpointing: bool = False,
    ):
        """
        Args:
            n_blocks (int): number of blocks in the backbone.
            dim (int): Embedding dimension.
            att_inner_dim (int): Inner dimension for attention.
            positional_encoding (str): Method for positional encoding in attention layers.
                Can be either "rope" for RoPE or "rpb" for relative positional bias.
            coords_dim (int, optional): Dimension of the coordinate embeddings. Must be specified
                if positional_encoding is "rpb". Ignored if positional_encoding is "rope".
            coords_inner_dim (int, optional): Inner dimension for coordinate embeddings. Must
                be specified if positional_encoding is "rpb".
                Ignored if positional_encoding is "rope".
            cond_dim (int, optional): Dimension of the conditioning vector for each source.
                if None, defaults to dim.
            downsample_input (bool, optional): Whether to downsample the inputs by a factor 2
                before feeding them to the backbone.
            use_checkpointing (bool, optional): Whether to use gradient checkpointing
                to save memory at the cost of extra computation.
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
            # For RoPE, the coordinate embeddings are not learned but computed on the fly
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

        # Optional downsampling and upsampling layers
        if downsample_input:
            self.input_downsampling = MultiSourcePatchMerging(dim, cond_dim)
            dim *= 2
            cond_dim *= 2
            self.input_upsampling = MultiSourcePatchSplitting(dim)

        # Build the successive blocks
        self.blocks = nn.ModuleList()
        for block_idx in range(n_blocks):
            shifted = bool(block_idx % 2)  # Shifting in both attention modules
            block = nn.ModuleList(
                [
                    AdapativeConditionalNormalization(
                        MultisourcesWindowedCrossAttention(
                            dim,
                            att_inner_dim,
                            window_size=cross_att_window_size,
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
            x: Map {(src, index): data_src} where data_src is an object containing:
                - embedding: tensor of shape (B, ..., dim)
                - conditioning: tensor of shape (B, ..., cond_dim) (optional)
        Returns:
            Dictionary {(source_name, index): x_s} of predicted data tensors
                of shape (B, ..., dim).
        """
        # Optional downsampling
        shapes_before_ds = {}
        if hasattr(self, "input_downsampling"):
            # Save the shapes before downsampling for later upsampling
            shapes_before_ds = {src: x[src].embedding.shape for src in x.keys()}
            x = self.ckpt(self.input_downsampling, x)

        for block in self.blocks:
            for layer in cast(nn.ModuleList, block):
                x = self.ckpt(layer, x)

        # Optional upsampling
        if hasattr(self, "input_upsampling"):
            x = self.ckpt(self.input_upsampling, x)
            # Crop the values to the original shapes
            for src in x.keys():
                if src in shapes_before_ds:
                    original_shape = shapes_before_ds[src]
                    x[src].embedding = x[src].embedding[
                        (slice(None),) + tuple(slice(0, s) for s in original_shape[1:])
                    ]

        # Return only the predicted values, like the original backbone
        return {src: x[src].embedding for src in x.keys()}


class AdapativeConditionalNormalization(nn.Module):
    """Wraps a torch module to apply adaptive conditional normalization, as in DiT.
    The module expects the data to include a key "conditioning", of same shape
    as the values. If that key is absent, no conditioning is applied and the inputs
    are just passed through LayerNorms, and residual connections are applied to
    the wrapped module's output.
    """

    def __init__(self, module: nn.Module, dim: int, cond_dim: int):
        """
        Args:
            module: The module to wrap.
            dim: Dimension of the values vector for each source.
            cond_dim: Dimension of the conditioning vector for each source.
        """
        super().__init__()
        self.module = module
        self.cond_dim = cond_dim

        # Normalization and conditioning projection
        self.norm = nn.LayerNorm(dim)
        self.cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(self.cond_dim, dim * 3))
        # Initialize the weights of the conditional normalization to zero (no effect)
        nn.init.zeros_(cast(Tensor, self.cond_proj[1].weight))
        nn.init.zeros_(cast(Tensor, self.cond_proj[1].bias))

    def forward(self, data: SourceEmbeddingDict, *args, **kwargs) -> SourceEmbeddingDict:
        """Args:
            data: Dictionary {(src, index): data_src} where data_src is an object containing:
                - embedding: tensor of shape (B, ..., dim)
                - conditioning: tensor of shape (B, ..., cond_dim) (optional)
        Returns:
            Dictionary {(source_name, index): data_src} in which the embeddings
                have been updated but the conditioning is unchanged.
        """
        skips, gates = {}, {}
        modulated_x: SourceEmbeddingDict = {}

        for src, source_data in data.items():
            skip = source_data.embedding
            x = self.norm(skip)
            skips[src] = skip

            # Apply conditioning if available
            cond = source_data.conditioning
            if cond is not None:
                # Apply conditioning to values
                shift, scale, gate = self.cond_proj(cond).chunk(3, dim=-1)
                x = x * (scale + 1) + shift
                gates[src] = gate

            # Save the module's inputs for that source
            modulated_x[src] = SourceEmbedding(
                embedding=x, coords=source_data.coords, conditioning=source_data.conditioning
            )

        # Apply the wrapped module with the updated inputs
        module_output: MultisourceTensor = self.module(modulated_x, *args, **kwargs)

        # Apply gates and skip connections
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
