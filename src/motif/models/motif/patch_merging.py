"""Implements the PatchMerging and PatchSplitting modules."""

import torch
from torch import nn
from torch.nn import functional as F

from motif.datatypes import SourceEmbedding, SourceEmbeddingDict


class PatchMerging(nn.Module):
    """Implements the patch merging of Swin Transformer."""

    def __init__(self, input_dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * input_dim)
        self.reduction = nn.Linear(4 * input_dim, 2 * input_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, H, W, C).
        Returns:
            torch.Tensor: Output tensor of shape (B, H//2, W//2, C*2).
        """
        # Pad the input tensor to have spatial dims divisible by 2
        B, H, W, C = x.shape
        if H % 2 != 0 or W % 2 != 0:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2), mode="constant", value=0)

        # Stack the corners of each 4x4 patch
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H//2, W//2, 4*C)

        # Reduce the dimensionality and normalize
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchSplitting(nn.Module):
    """Inverse operation of the patch merging."""

    def __init__(self, input_dim):
        super().__init__()
        self.expansion = nn.Linear(input_dim, 2 * input_dim, bias=False)
        self.norm = nn.LayerNorm(2 * input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, H//2, W//2, C).
        Returns:
            torch.Tensor: Output tensor of shape (B, H, W, C//2).
        """
        # Expand the dimensionality
        x = self.expansion(x)
        x = self.norm(x)

        # Reshape and split the channels
        B, H_half, W_half, C4 = x.shape
        C = C4 // 4
        x0 = x[:, :, :, 0 * C : 1 * C]
        x1 = x[:, :, :, 1 * C : 2 * C]
        x2 = x[:, :, :, 2 * C : 3 * C]
        x3 = x[:, :, :, 3 * C : 4 * C]

        # Rearrange to reconstruct the original spatial dimensions
        H, W = H_half * 2, W_half * 2
        x = torch.zeros((B, H, W, C), device=x.device, dtype=x.dtype)
        x[:, 0::2, 0::2, :] = x0
        x[:, 1::2, 0::2, :] = x1
        x[:, 0::2, 1::2, :] = x2
        x[:, 1::2, 1::2, :] = x3

        return x


class MultiSourcePatchMerging(nn.Module):
    """Implements the patch merging of Swin Transformer for multi-source inputs.
    Downsamples the embedding and the conditioning of each source by a factor 2,
    while doubling the feature dimension.
    """

    def __init__(self, dim: int, cond_dim: int | None = None):
        super().__init__()
        self.merging = PatchMerging(dim)
        if cond_dim is not None:
            self.conditioning_merging = PatchMerging(cond_dim)

    def forward(self, inputs: SourceEmbeddingDict) -> SourceEmbeddingDict:
        """
        Args:
            x: Map {(src, index): data_src} where data_src is an object containing:
                - embedding: tensor of shape (B, ..., dim)
                - conditioning: tensor of shape (B, ..., cond_dim) (optional)

        Returns:
            dict: {(source_name, index): merged_data_src}.
        """
        # Apply patch merging to each source independently
        outputs = {}
        for source, src_data in inputs.items():
            embedding = src_data.embedding
            conditioning = src_data.conditioning

            cond = None
            if conditioning is not None:
                cond = self.conditioning_merging(conditioning)

            outputs[source] = SourceEmbedding(embedding=self.merging(embedding), conditioning=cond)

        return outputs


class MultiSourcePatchSplitting(nn.Module):
    """Inverse operation of the patch merging for multi-source inputs.
    Upsamples the embedding and the conditioning of each source by a factor 2,
    while halving the feature dimension.
    """

    def __init__(self, dim: int, cond_dim: int | None = None):
        """Args:
        dim: Dimension of the embedding to be split.
        cond_dim: Dimension of the conditioning to be split.
        """
        super().__init__()
        self.embedding_splitting = PatchSplitting(dim)
        self.cond_dim = cond_dim
        if cond_dim is not None:
            self.conditioning_splitting = PatchSplitting(cond_dim)

    def forward(self, inputs: SourceEmbeddingDict) -> SourceEmbeddingDict:
        """
        Args:
            x: Map {(src, index): data_src} where data_src is an object containing:
                - embedding: tensor of shape (B, ..., dim)
                - conditioning: tensor of shape (B, ..., cond_dim) (optional)
        Returns:
            dict: {(source_name, index): split_data_src}.
        """
        # Apply patch splitting to each source independently
        outputs = {}
        for source, src_data in inputs.items():
            embedding = src_data.embedding
            conditioning = src_data.conditioning

            cond = None
            if conditioning is not None and self.cond_dim is not None:
                cond = self.conditioning_splitting(conditioning)

            outputs[source] = SourceEmbedding(
                embedding=self.embedding_splitting(embedding), conditioning=cond
            )

        return outputs
