"""Implements the SpatiotemporalRoPEAttention module."""

from typing import cast

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention


class SpatiotemporalRoPEAttention(nn.Module):
    """Computes spatiotemporal attention with rotary positional encoding (RoPE).

    The feature queries/keys are rotated before attention with three coordinate-driven axes:
    - latitude axis (axial 2D RoPE),
    - longitude axis (axial 2D RoPE),
    - time axis (separate 1D RoPE).

    The attention logits are then computed with

    $$A = Softmax(RoPE(Q_f, q_c) @ RoPE(K_f, k_c)^T / \\sqrt{D_h})$$

    where $q_c$ and $k_c$ store raw coordinates `(lat, lon, time)`.
    """

    def __init__(
        self,
        features_dim: int,
        num_heads: int,
        rope_base: float = 10000.0,
        lat_scale: float = 1.0,
        lon_scale: float = 1.0,
        time_scale: float = 1.0,
        rope_dims: tuple[int, int, int] | None = None,
    ):
        """
        Args:
            features_dim (int): Dimension of the feature embeddings.
            num_heads (int): Number of attention heads.
            rope_base (float): Base frequency for RoPE.
            lat_scale (float): Multiplicative scaling for latitude positions.
            lon_scale (float): Multiplicative scaling for longitude positions.
            time_scale (float): Multiplicative scaling for time positions.
            rope_dims (tuple[int, int, int] | None): Per-head rotary dimensions
                `(lat_dim, lon_dim, time_dim)`. If None, dimensions are inferred.

        Raises:
            ValueError: If dimensions are incompatible with multi-head attention or RoPE.
        """
        super().__init__()
        self.features_dim = features_dim
        self.num_heads = num_heads
        head_dim = features_dim // num_heads
        if head_dim * num_heads != features_dim:
            raise ValueError(
                f"features_dim must be divisible by num_heads, got {features_dim} and {num_heads}"
            )
        self.head_dim = head_dim

        if rope_dims is None:
            lat_dim, lon_dim, time_dim = self._infer_rope_dims(head_dim)
        else:
            if len(rope_dims) != 3:
                raise ValueError(f"rope_dims must contain 3 values, got {len(rope_dims)}")
            lat_dim, lon_dim, time_dim = rope_dims

        for axis_name, axis_dim in zip(
            ("lat_dim", "lon_dim", "time_dim"),
            (lat_dim, lon_dim, time_dim),
            strict=True,
        ):
            if axis_dim <= 0 or axis_dim % 2 != 0:
                raise ValueError(
                    f"{axis_name} must be a strictly positive even integer, got {axis_dim}"
                )

        total_rope_dim = lat_dim + lon_dim + time_dim
        if total_rope_dim > head_dim:
            raise ValueError(
                f"Sum of rotary dimensions must be <= head_dim, got {total_rope_dim} > {head_dim}"
            )

        self.lat_dim = lat_dim
        self.lon_dim = lon_dim
        self.time_dim = time_dim
        self.pass_dim = head_dim - total_rope_dim

        self.lat_scale = lat_scale
        self.lon_scale = lon_scale
        self.time_scale = time_scale

        self.register_buffer(
            "inv_freq_lat", self._build_inv_freq(self.lat_dim, rope_base), persistent=False
        )
        self.register_buffer(
            "inv_freq_lon", self._build_inv_freq(self.lon_dim, rope_base), persistent=False
        )
        self.register_buffer(
            "inv_freq_time", self._build_inv_freq(self.time_dim, rope_base), persistent=False
        )

    @staticmethod
    def _infer_rope_dims(head_dim: int) -> tuple[int, int, int]:
        """Infer per-axis rotary dimensions from the per-head feature dimension."""
        total_pairs = head_dim // 2
        lat_pairs = total_pairs // 3
        lon_pairs = total_pairs // 3
        time_pairs = total_pairs - lat_pairs - lon_pairs
        if min(lat_pairs, lon_pairs, time_pairs) <= 0:
            raise ValueError(
                "head_dim is too small to allocate RoPE for lat/lon/time. "
                "Use head_dim >= 6 or provide rope_dims explicitly."
            )
        return 2 * lat_pairs, 2 * lon_pairs, 2 * time_pairs

    @staticmethod
    def _build_inv_freq(rotary_dim: int, rope_base: float) -> Tensor:
        """Build inverse frequencies for rotary channels."""
        exponent = torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim
        return 1.0 / (rope_base**exponent)

    @staticmethod
    def _apply_rope_axis(
        x_axis: Tensor,
        position: Tensor,
        inv_freq: Tensor,
        position_scale: float,
    ) -> Tensor:
        """Apply RoPE rotation on one axis.

        Args:
            x_axis (Tensor): Input feature slice. Shape [B, ..., H, L, D_axis].
            position (Tensor): Raw coordinates for one axis. Shape [B, ..., L].
            inv_freq (Tensor): Inverse frequencies. Shape [D_axis // 2].
            position_scale (float): Multiplicative scaling of the coordinates.

        Returns:
            Tensor: Rotated feature slice. Shape [B, ..., H, L, D_axis].
        """
        if x_axis.shape[-1] == 0:
            return x_axis

        # Reshape the positions to add a head dim and a features dim.
        position = position.to(dtype=torch.float32) * position_scale  # [B, ..., L]
        angles = position.unsqueeze(-2).unsqueeze(-1)  # [B, ..., 1, L, 1]

        # Multiply the angles by the inverse frequencies to get the rotation phases.
        inv_freq = inv_freq.to(device=x_axis.device)
        freq_shape = (1,) * (angles.ndim - 1) + (inv_freq.shape[0],)
        phases = angles * inv_freq.view(freq_shape)  # [B, ..., 1, L, D_axis//2]

        cos_phase = phases.cos().to(dtype=x_axis.dtype)
        sin_phase = phases.sin().to(dtype=x_axis.dtype)

        # Apply the rotation to the input features.
        x_even = x_axis[..., 0::2]  # [B, ..., H, L, D_axis//2]
        x_odd = x_axis[..., 1::2]  # [B, ..., H, L, D_axis//2]
        rotated_even = x_even * cos_phase - x_odd * sin_phase
        rotated_odd = x_even * sin_phase + x_odd * cos_phase
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)

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
            qc (Tensor): Queries for the coordinates, of shape (B, ..., L, 3),
                where the channels are in order (lat, lon, time).
            kc (Tensor): Keys for the coordinates, of shape (B, ..., S, 3)
            v (Tensor): Values, of shape (B, ..., S, D_values). Note that D_values may be
                different from D_feat.
            attn_mask (Tensor, optional): Attention mask, either boolean or float, which will
                be summed to the dot product before calling softmax.
                (see torch.nn.functional.scaled_dot_product_attention).
                Expected of shape (B, ..., L, S).
            dropout (float, optional): Dropout rate. Defaults to 0.0.

        Returns:
            Tensor: The output of the attention layer, of shape (B, ..., L, D_values).

        Raises:
            ValueError: If tensor shapes are incompatible.
        """

        if qc.shape[-1] != 3 or kc.shape[-1] != 3:
            raise ValueError(
                "qc and kc must contain raw coordinates with last dim = 3 (lat, lon, time)."
            )

        values_dim = v.shape[-1]
        if values_dim % self.num_heads != 0:
            raise ValueError(
                "The values dimension must be divisible by num_heads, got "
                f"{values_dim} and {self.num_heads}."
            )

        # Split attention heads.
        # qf: [B, ..., L, D_feat] -> [B, ..., H, L, Dh]
        # kf: [B, ..., S, D_feat] -> [B, ..., H, S, Dh]
        qf = rearrange(qf, "B ... L (H D) -> B ... H L D", H=self.num_heads)
        kf = rearrange(kf, "B ... S (H D) -> B ... H S D", H=self.num_heads)
        v = rearrange(v, "B ... S (H D) -> B ... H S D", H=self.num_heads)

        # Build axis-wise feature slices.
        q_lat = qf[..., : self.lat_dim]
        q_lon = qf[..., self.lat_dim : self.lat_dim + self.lon_dim]
        q_time = qf[..., self.lat_dim + self.lon_dim : self.lat_dim + self.lon_dim + self.time_dim]
        q_pass = qf[..., self.lat_dim + self.lon_dim + self.time_dim :]

        k_lat = kf[..., : self.lat_dim]
        k_lon = kf[..., self.lat_dim : self.lat_dim + self.lon_dim]
        k_time = kf[..., self.lat_dim + self.lon_dim : self.lat_dim + self.lon_dim + self.time_dim]
        k_pass = kf[..., self.lat_dim + self.lon_dim + self.time_dim :]

        # Apply axial 2D RoPE (lat/lon) + temporal RoPE.
        inv_freq_lat = cast(Tensor, self.inv_freq_lat)
        inv_freq_lon = cast(Tensor, self.inv_freq_lon)
        inv_freq_time = cast(Tensor, self.inv_freq_time)

        q_lat = self._apply_rope_axis(q_lat, qc[..., 0], inv_freq_lat, self.lat_scale)
        q_lon = self._apply_rope_axis(q_lon, qc[..., 1], inv_freq_lon, self.lon_scale)
        q_time = self._apply_rope_axis(q_time, qc[..., 2], inv_freq_time, self.time_scale)

        k_lat = self._apply_rope_axis(k_lat, kc[..., 0], inv_freq_lat, self.lat_scale)
        k_lon = self._apply_rope_axis(k_lon, kc[..., 1], inv_freq_lon, self.lon_scale)
        k_time = self._apply_rope_axis(k_time, kc[..., 2], inv_freq_time, self.time_scale)

        q_rope = torch.cat((q_lat, q_lon, q_time, q_pass), dim=-1)  # [B, ..., H, L, Dh]
        k_rope = torch.cat((k_lat, k_lon, k_time, k_pass), dim=-1)  # [B, ..., H, S, Dh]

        # Add a head dim to the attention mask
        if attn_mask is not None:
            attn_mask = rearrange(attn_mask, "B ... L S -> B ... 1 L S")

        # SDPA on RoPE-conditioned q/k.
        # out_heads: [B, ..., H, L, Dv_per_head]
        out_heads = scaled_dot_product_attention(
            q_rope,
            k_rope,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout,
        )

        # Merge heads back.
        # out: [B, ..., L, Dv]
        out = rearrange(out_heads, "B ... H L D -> B ... L (H D)")
        return out
