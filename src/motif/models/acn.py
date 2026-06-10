"""Shared normalization wrapper layers used across backbone variants."""

from typing import cast

import torch.nn as nn
from torch import Tensor

from motif.datatypes import MultisourceTensor, SourceEmbedding, SourceEmbeddingDict


class AdaptiveConditionalNormalization(nn.Module):
    """Wraps a torch module to apply adaptive conditional normalization, as in DiT.
    The module expects the data to include a key "conditioning", of same shape
    as the values. If that key is absent, no conditioning is applied and the inputs
    are just passed through LayerNorms, and residual connections are applied to
    the wrapped module's output.

    Operates on a SourceEmbeddingDict (one entry per source, arbitrary spatial shape).
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

        self.norm = nn.LayerNorm(dim)
        self.cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(self.cond_dim, dim * 3))
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


class SequenceAdaptiveConditionalNormalization(nn.Module):
    """Wraps a torch module to apply adaptive conditional normalization on a single
    merged sequence SourceEmbedding of shape (B, L, dim).

    The wrapped module receives a SourceEmbedding and returns a Tensor (B, L, dim).
    """

    def __init__(self, module: nn.Module, dim: int, cond_dim: int):
        """
        Args:
            module: Module taking SourceEmbedding → Tensor (B, L, dim).
            dim: Feature dimension.
            cond_dim: Conditioning dimension.
        """
        super().__init__()
        self.module = module
        self.cond_dim = cond_dim

        self.norm = nn.LayerNorm(dim)
        self.cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(self.cond_dim, dim * 3))
        nn.init.zeros_(cast(Tensor, self.cond_proj[1].weight))
        nn.init.zeros_(cast(Tensor, self.cond_proj[1].bias))

    def forward(self, data: SourceEmbedding) -> SourceEmbedding:
        """Args:
            data: SourceEmbedding with:
                - embedding: (B, L, dim)
                - coords: (B, L, coords_dim)
                - conditioning: (B, L, cond_dim) or None
        Returns:
            SourceEmbedding with updated embedding; coords and conditioning unchanged.
        """
        skip = data.embedding
        x = self.norm(skip)

        gate: Tensor | None = None
        if data.conditioning is not None:
            shift, scale, gate = self.cond_proj(data.conditioning).chunk(3, dim=-1)
            x = x * (scale + 1) + shift

        module_out: Tensor = self.module(
            SourceEmbedding(embedding=x, coords=data.coords, conditioning=data.conditioning)
        )

        if gate is not None:
            out = module_out * gate + skip
        else:
            out = module_out + skip

        return SourceEmbedding(embedding=out, coords=data.coords, conditioning=data.conditioning)
