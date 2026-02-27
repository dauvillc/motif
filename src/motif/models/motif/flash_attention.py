"""Implements the SpatiotemporalFlashAttention module."""

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention


class SpatiotemporalFlashAttention(nn.Module):
    """Computes attention scores based on inputs that are in two parts:
    - "Features": embedded vectors (e.g. embedded vit patches);
    - "Coordinates": embedded coordinate vectors which are used as
            relative positional encoding.
    More precisely, given feature queries and keys $(Qf, Kf)$ and coordinate queries and keys
    $(Qc, Kc)$, computes

    $$A = Softmax(Qf @ Kf^T / sqrt(Df) + \alpha * Qc @ Kc^T / sqrt(Dc))$$

    where $\alpha$ is a learned parameter which only varies across heads, and $Df$ and $Dc$ are
    the feature and coordinate embedding dimensions, respectively.

    Uses torch.scaled_dot_product_attention for efficient attention computation.
    """

    def __init__(self, features_dim: int, coords_dim: int, num_heads: int):
        """
        Args:
            features_dim (int): Dimension of the feature embeddings.
            coords_dim (int): Dimension of the coordinate embeddings.
            num_heads (int): Number of attention heads.
        """
        super().__init__()
        self.features_dim = features_dim
        self.coords_dim = coords_dim
        self.num_heads = num_heads
        head_dim = features_dim // num_heads
        if head_dim * num_heads != features_dim:
            raise ValueError(
                f"features_dim must be divisible by num_heads, got {features_dim} and {num_heads}"
            )
        coords_head_dim = coords_dim // num_heads
        if coords_head_dim * num_heads != coords_dim:
            raise ValueError(
                f"coords_dim must be divisible by num_heads, got {coords_dim} and {num_heads}"
            )

        # Pre-scaling trick:
        # We're going to concatenate the features and the coordinates
        # along the channel dimension, so that the total attention scores are effectively
        # the sums of the feature-based and coordinate-based scores.
        # However, by just doing that, torch's SDPA would normalize the total dot product by
        # sqrt(D_feat + D_coord), which would make the coordinate-based scores have a smaller
        # impact just because they are encoded with less dimensions. Instead, we want the
        # importance of the coordinates in the attention scores to be learned. Therefore:
        # - We pre-scale the feature and coord-based queries so that the effective
        #   normalizations after SDPA are sqrt(D_feat) and sqrt(D_coord) respectively;
        # - We multiply the coordinate part by a learned alpha parameter, which only
        #   varies across heads.

        D_feat = head_dim
        D_coord = coords_head_dim
        D_total = D_feat + D_coord

        self.scale_feat = (D_total / D_feat) ** 0.5
        self.scale_coord = (D_total / D_coord) ** 0.5
        self.alpha = nn.Parameter(torch.ones(num_heads))

    def forward(
        self,
        qf: Tensor,
        kf: Tensor,
        qc: Tensor,
        kc: Tensor,
        v: Tensor,
        attn_mask: Tensor | None = None,
        dropout: float = 0.0,
    ) -> Tensor:
        """
        Args:
            qf (Tensor): Queries for the features, of shape (B, ..., L, D_feat) where
                L is the query sequence length, D_feat is the feature dimension,
                and ... is an arbitrary number of dimensions.
            kf (Tensor): Keys for the features, of shape (B, ..., S, D_feat)
                where S is the key/value sequence length.
            qc (Tensor): Queries for the coordinates, of shape (B, ..., L, D_coord)
            kc (Tensor): Keys for the coordinates, of shape (B, ..., S, D_coord)
            v (Tensor): Values, of shape (B, ..., S, D_values). Note that D_values may be
                different from D_feat.
            attn_mask (Tensor, optional): Attention mask, either boolean or float, which will
                be summed to the dot product before calling softmax.
                (see torch.nn.functional.scaled_dot_product_attention).
                Expected of shape (B, ..., L, S).
            dropout (float, optional): Dropout rate. Defaults to 0.0.

        Returns:
            Tensor: The output of the attention layer, of shape (B, ..., D_values).
        """

        # Reshape to split the attention heads
        qf = rearrange(qf, "B ... L (H D) -> B ... H L D", H=self.num_heads)
        kf = rearrange(kf, "B ... S (H D) -> B ... H S D", H=self.num_heads)
        qc = rearrange(qc, "B ... L (H D) -> B ... H L D", H=self.num_heads)
        kc = rearrange(kc, "B ... S (H D) -> B ... H S D", H=self.num_heads)
        v = rearrange(v, "B ... S (H D) -> B ... H S D", H=self.num_heads)
        if attn_mask is not None:
            attn_mask = rearrange(attn_mask, "B ... L S -> B ... 1 L S")

        # Expand alpha to be broadcastable with the queries
        # -> view as (1, ..., H, 1, 1)
        alpha = self.alpha.view((1,) * (len(qf.shape) - 3) + (-1,) + (1, 1))

        # Pre-scale the queries
        qf = qf * self.scale_feat
        qc = qc * (alpha * self.scale_coord)

        # Concatenate the features and coordiantes along the channel dimension
        q_cat = torch.cat([qf, qc], dim=-1)  # (B, ..., H, L, D_feat + D_coord)
        k_cat = torch.cat([kf, kc], dim=-1)  # (B, ..., H, S, D_feat + D_coord)

        # Compute the attention scores using torch's SDPA
        dot_prod = scaled_dot_product_attention(
            q_cat, k_cat, v, attn_mask=attn_mask, dropout_p=dropout
        )

        # Regroup the heads and return
        out = rearrange(dot_prod, "B ... H L D -> B ... L (H D)")
        return out
