import torch
import torch.nn as nn

from motif.datatypes import MultisourceTensor, SourceEmbeddingDict


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dropout: float = 0.0,
        act_layer: nn.Module = nn.GELU(),
        inner_ratio: float = 4.0,
        **kwargs,
    ):
        """Args:
        dim (int): Dimension of the values.
        dropout (float): Dropout rate.
        act_layer (nn.Module): Activation layer to use.
        inner_ratio (int): Ratio for the inner dimension compared to the input dimension.
        """
        super().__init__()
        inner_dim = int(dim * inner_ratio)

        # Network for values
        self.values_net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            act_layer,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: SourceEmbeddingDict, **kwargs) -> MultisourceTensor:
        """
        Args:
            x (SourceEmbeddingDict): dict {src: SourceEmbedding}.
        Returns:
            Dictionary {src: x_s} of updated features of shape (B, ..., dim).
        """
        outputs = {}
        for src, data in x.items():
            features = data.embedding
            outputs[src] = self.values_net(features)
        return outputs


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# Taken from torchtune
class RMSNorm(nn.Module):
    """
    Implements Root Mean Square Normalization introduced in
    https://arxiv.org/abs/1910.07467.

    Reference implementation (used for correctness verification)
    can be found here:
    https://github.com/facebookresearch/llama/blob/main/llama/model.py

    Args:
        dim (int): embedding size
        eps (float): small value to avoid division by zero. Default: 1e-6
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor to normalize

        Returns:
            torch.Tensor: The normalized and scaled tensor having the same shape as ``x``.
        """
        # computation is in fp32
        x_fp32 = x.float()
        x_normed = (x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(
            x
        )
        return x_normed * self.scale
