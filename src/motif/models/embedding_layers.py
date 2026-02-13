"""Implements embedding layers."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from motif.datatypes import SourceData, SourceEmbedding


class LinearEmbedding(nn.Module):
    """Embeds a vector using a linear layer."""

    def __init__(self, input_dim: int, output_dim: int, n_layers: int = 1, norm: bool = False):
        super(LinearEmbedding, self).__init__()
        if n_layers > 1:
            layers = []
            for _ in range(n_layers - 1):
                layers.append(nn.Linear(output_dim, output_dim))
                layers.append(nn.GELU())
            self.embedding = nn.Sequential(self.embedding, *layers)
        else:
            self.embedding = nn.Linear(input_dim, output_dim)
        self.act = nn.GELU()
        self.use_norm = norm
        if norm:
            self.ln = nn.LayerNorm(output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)
        x = self.act(x)
        if self.use_norm:
            x = self.ln(x)
        return x


class ConvPatchEmbedding2d(nn.Module):
    """A module that embeds an image into a sequence of patches using
    a 2D convolutional layer.
    """

    def __init__(
        self, channels: int, patch_size: int, emb_dim: int, mlp_layers: int = 0, norm: bool = True
    ):
        """
        Args:
            channels: The number of channels in the image.
            patch_size: The size of the patches.
            emb_dim: The dimension of the embedding space.
            mlp_layers: The number of linear layers to apply after the convolution.
            norm: Whether to apply layer normalization after the embedding.
        """
        super().__init__()
        self.patch_size = patch_size

        self.embedding = nn.Sequential(
            nn.Conv2d(channels, emb_dim, kernel_size=self.patch_size, stride=self.patch_size),
            nn.GELU(),
        )
        if mlp_layers > 0:
            mlp = []
            for _ in range(mlp_layers):
                mlp.append(nn.Linear(emb_dim, emb_dim))
                mlp.append(nn.GELU())
            self.mlp = nn.Sequential(*mlp)
        if norm:
            self.norm = nn.LayerNorm(emb_dim)

    def forward(self, image: Tensor) -> Tensor:
        """
        Args:
            image: A tensor of shape (B, C, H, W) containing the image.
        Returns:
            embedded_image: Tensor of shape (B, h, w, emb_dim).
        """
        # Compute padding dynamically
        H, W = image.shape[2:]
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        image = F.pad(image, (0, pad_w, 0, pad_h))

        embedded_image = self.embedding(image)  # (B, emb_dim, h, w)
        embedded_image = embedded_image.permute(0, 2, 3, 1)  # (B, h, w, emb_dim)

        if hasattr(self, "mlp"):
            embedded_image = self.mlp(embedded_image)

        if hasattr(self, "norm"):
            embedded_image = self.norm(embedded_image)
        return embedded_image


class SinusoidalEmbedding(nn.Module):
    """
    Sinusoidal embedding using Fourier features.
    Based on the standard transformer position embedding.
    """

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("Embedding dimension must be even.")
        self.dim = dim
        self.max_period = max_period

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (..., C) or (..., C, 1).
        Returns:
            Tensor of shape (..., C, fourier_dim) containing sinusoidal embeddings.
        """
        if x.size(-1) > 1:
            x = x.unsqueeze(-1)  # (..., C, 1)

        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=x.device)  # (half,)

        args = x.float() * freqs[None]  # (..., C, half)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (..., C, dim)

        return embedding


class SpatioTemporalFourierCoordinateEmbedding(nn.Module):
    """
    Embeds spatio-temporal coordinates (latitude, longitude, time) using Fourier features
    and a projection MLP.
    """

    def __init__(self, fourier_dim: int):
        """
        Args:
            fourier_dim: Dimension of the Fourier features for both spatial
                and temporal embeddings.
        """
        super().__init__()
        self.embedder = SinusoidalEmbedding(fourier_dim)

    def forward(self, coords: Tensor, times: Tensor) -> Tensor:
        """
        Args:
            coords: Tensor of shape (B, 2, H, W) containing lat/lon at each pixel.
            times: Tensor of shape (B,).
        Returns:
            Tensor: Tensor of shape (B, 3 * F, H, W) where F is the fourier_dim.
        """
        B, _, H, W = coords.shape
        lat_lon = coords.permute(0, 2, 3, 1).reshape(B * H * W, 2)  # (B*H*W, 2)
        times = times.view(B, 1).expand(B, H * W).reshape(B * H * W, 1)  # (B*H*W, 1)
        full_coords = torch.cat([lat_lon, times], dim=1)  # (B*H*W, 3)

        fourier_coords: Tensor = self.embedder(full_coords)  # (B*H*W, 3, fourier_dim)
        fourier_coords = fourier_coords.view(B, H, W, 3 * fourier_coords.size(-1))
        fourier_coords = fourier_coords.permute(0, 3, 1, 2)  # (B, 3*fourier_dim, H, W)

        return fourier_coords


class AvgPoolCoordinateEmbedding(nn.Module):
    """
    Embeds spatio-temporal coordinates (latitude, longitude, time) using average pooling.
    """

    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.pool = nn.AvgPool2d(kernel_size=patch_size, stride=patch_size)

    def forward(self, coords: Tensor, times: Tensor) -> Tensor:
        """
        Args:
            coords: Tensor of shape (B, 2, H, W) containing lat/lon at each pixel.
            times: Tensor of shape (B,).
        Returns:
            Tensor: Tensor of shape (B, h, w, 3) containing embedded coordinates.
        """

        # Compute padding dynamically
        H, W = coords.shape[2:]
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        coords = F.pad(coords, (0, pad_w, 0, pad_h))

        embedded_coords = self.pool(coords)  # (B, 2, h, w)
        B, _, h, w = embedded_coords.shape
        times = times.view(B, 1, 1, 1).expand(B, 1, h, w)  # (B, 1, h, w)
        embedded_coords = torch.cat([embedded_coords, times], dim=1)  # (B, 3, h, w)
        embedded_coords = embedded_coords.permute(0, 2, 3, 1)  # (B, h, w, 3)
        return embedded_coords


class SourcetypeEmbedding2d(nn.Module):
    """A class that embeds the values and coordinates of a 2D source, including optional
    characteristic variables.

    This module handles both coordinate embeddings (latitude, longitude,
    and time) and values embeddings (channels, masks, diffusion timestep),
    and returns a set of embedded vectors.

    Additionally, a conditioning tensor is computed that embeds the conditioning
    that isn't the values or the spatio-temporal coordinates. This includes:
    - The characteristic variables.
    - The diffusion timestep.
    If those elements aren't given, the conditioning tensor is set to None.
    """

    def __init__(
        self,
        channels: int,
        patch_size: int,
        dim: int,
        cond_dim: int,
        dff_step_fourier_dim: int = 256,
        n_charac_vars: int = 0,
        use_diffusion_t: bool = True,
        pred_mean_channels: int = 0,
        conditioning_mlp_layers: int = 0,
        coords_encoding_method: str = "fourier",
        coords_fourier_dim: int = 64,
        coords_dim: int | None = None,
    ):
        """
        Args:
            channels: Number of channels for the source data, excluding
                land-sea and availability masks.
            patch_size: Size of the patches to be used for convolution.
            dim: Dimension of the embedding space.
            cond_dim: Dimension of the conditioning embedding space.
            dff_step_fourier_dim: Dimension of the Fourier features for
                the diffusion timestep embedding.
            n_charac_vars: Number of optional characteristic variables.
            use_diffusion_t: Whether to include a diffusion timestep embedding.
            pred_mean_channels: Number of channels for the predicted mean. If zero,
                the predicted mean is not used.
            conditioning_mlp_layers: Number of linear layers to apply
                after the conditioning concatenation.
            coords_encoding_method: Method to encode the spatio-temporal coordinates.
                Options:
                    - "fourier": Use Fourier features as in the original DiT.
                    - "avg_pool": Use average pooling to embed the coordinates.
            coords_fourier_dim (int, optional): Dimension of the Fourier features
                if coords_encoding_method is "fourier".
            coords_dim (int, optional): Dimension of the coordinate embedding if
                coords_encoding_method is "fourier".
        """
        super().__init__()

        # Values embedding layers
        self.values_patch_size = patch_size
        self.use_diffusion_t = use_diffusion_t
        self.use_predicted_mean = pred_mean_channels > 0

        # landmask + availability mask + predicted mean
        ch = channels + 2 + pred_mean_channels
        self.values_embedding = ConvPatchEmbedding2d(ch, patch_size, dim, norm=True)

        # Coords embedding layers
        self.coords_patch_size = patch_size
        self.coords_encoding_method = coords_encoding_method
        if coords_encoding_method == "fourier":
            if coords_dim is None:
                raise ValueError("coords_dim must be specified if using fourier encoding.")
            self.coords_fourier_embedder = SpatioTemporalFourierCoordinateEmbedding(
                coords_fourier_dim
            )
            self.coords_embedding = ConvPatchEmbedding2d(
                coords_fourier_dim * 3, patch_size, coords_dim, norm=True
            )
        else:
            self.coords_embedding = AvgPoolCoordinateEmbedding(patch_size)

        # Conditioning embedding layers
        # - spatial conditioning
        ch_spatial_cond = 1  # Land-sea mask
        self.spatial_cond_embedding = ConvPatchEmbedding2d(
            ch_spatial_cond,
            patch_size,
            cond_dim,
            norm=False,
            mlp_layers=conditioning_mlp_layers,
        )
        # - 0D / 1D embeddings
        # Availability flag + characteristic variables
        ch_cond = 1 + n_charac_vars
        self.cond_embedding = LinearEmbedding(
            ch_cond, cond_dim, n_layers=conditioning_mlp_layers, norm=False
        )
        # If requested, including a sinusoidal embedding of the diffusion timestep.
        if self.use_diffusion_t:
            self.time_embedder = SinusoidalEmbedding(dff_step_fourier_dim)
            # 2. MLP to project it to the conditioning dimension
            self.diff_step_mlp = nn.Sequential(
                nn.Linear(dff_step_fourier_dim, cond_dim),
                nn.SiLU(),  # Swish (SiLU) is standard for time embeddings
                nn.Linear(cond_dim, cond_dim),
            )
        # final layer normalization for the conditioning
        self.cond_norm = nn.LayerNorm(cond_dim)

    def forward(self, data: SourceData) -> SourceEmbedding:
        """
        Args:
            data (dict): Must contain:
                - "coords": (B, 2, H, W) lat/lon tensor.
                - "dt": (B,) time delta tensor.
                - "values": (B, C, H, W) source pixel data.
                - "avail_mask": (B, H, W) data availability mask.
                - "landmask": (B, H, W) land-sea mask.
                - "avail": (B,) tensor, valued 1 if the sample is available, -1 if missing,
                    and 0 if masked (and to be reconstructed).
                - Optionally "characs": (B, n_characs_vars) characteristic variables.
                - Optionally "diffusion_t": (B,) diffusion timestep.
                - Optionally "pred_mean": (B, C_out, H, W), predicted mean.
        Returns:
            A SourceEmbedding object containing:
                - embedding: (B, h, w, dim) tensor
                - coords: (B, h, w, coords_dim) tensor. The coords_dim is:
                    - 3 if using the AvgPoolCoordinateEmbedding (lat, lon, time).
                    - dim if using fourier features.
                - conditioning: (B, h, w, cond_dim) tensor or None.
        """

        # Embed values
        with torch.backends.cudnn.flags(enabled=self.values_patch_size <= 4):
            values_tensors = []
            values = data.values
            avail_mask = data.avail_mask.unsqueeze(1)
            landmask = data.landmask.unsqueeze(1)
            values_tensors = [values, avail_mask, landmask]
            if self.use_predicted_mean and data.pred_mean is not None:
                values_tensors.append(data.pred_mean)
            values = torch.cat(values_tensors, dim=1)  # (B, C, H, W)
            embedded_values = self.values_embedding(values)

        # Embed coords
        coords = data.coords  # (B, 2, H, W)
        dt = data.dt  # (B,)
        if self.coords_encoding_method == "fourier":
            fourier_coords = self.coords_fourier_embedder(coords, dt)  # (B, 3*F, H, W)
            with torch.backends.cudnn.flags(enabled=self.values_patch_size <= 4):
                embedded_coords = self.coords_embedding(fourier_coords)  # (B, h, w, dim)
        else:
            embedded_coords = self.coords_embedding(coords, dt)  # (B, h, w, 3)

        # Conditioning tensor
        available_conditioning = []
        b = values.size(0)
        # Spatial conditionings (embedded together via patch embedding)
        # - Land-sea mask
        spatial_cond = self.spatial_cond_embedding(landmask)  # (B, h, w, cond_dim)
        available_conditioning.append(spatial_cond)
        # Non-spatial conditionings: we'll concatenate the ones that are available
        # and embed them with a single linear embedding.
        conds = []
        # - Availability flag (always used)
        conds.append(data.avail.float().view(-1, 1))  # (B, 1)
        # - Characteristic variables
        if data.characs is not None:
            conds.append(data.characs)  # (B, n_characs_vars)
        # Concatenate the conditioning tensors
        conds = torch.cat(conds, dim=1)  # (B, ch_cond)
        conds = self.cond_embedding(conds)  # (B, cond_dim)
        # Reshape to sum to the spatial dimensions
        conds = conds.view(b, 1, 1, -1)
        available_conditioning.append(conds)
        # - Diffusion timestep (if used)
        if self.use_diffusion_t:
            if data.diffusion_t is None:
                raise ValueError("diffusion_t must be provided if use_diffusion_t is True.")
            diffusion_t = data.diffusion_t  # (B,)
            # Embed the diffusion timestep with a sinusoidal embedding + MLP
            diffusion_t_emb = self.time_embedder(diffusion_t)  # (B, time_emb_dim)
            diffusion_t_emb = self.diff_step_mlp(diffusion_t_emb)  # (B, cond_dim)
            # Reshape to sum to the spatial dimensions
            diffusion_t_emb = diffusion_t_emb.view(b, 1, 1, -1)
            available_conditioning.append(diffusion_t_emb)
        # Finally, sum the available conditioning tensors
        # and apply layer normalization.
        if len(available_conditioning) > 0:
            conditioning = sum(available_conditioning)
            conditioning = self.cond_norm(conditioning)
        else:
            conditioning = None

        return SourceEmbedding(
            embedding=embedded_values, coords=embedded_coords, conditioning=conditioning
        )
