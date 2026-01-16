"""Implements embedding layers."""

import math

import torch
import torch.nn as nn


class LinearEmbedding(nn.Module):
    """Embeds a vector using a linear layer."""

    def __init__(self, input_dim, output_dim, n_layers=1, norm=False):
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

    def forward(self, x):
        x = self.embedding(x)
        x = self.act(x)
        if self.use_norm:
            x = self.ln(x)
        return x


class CornerAndCenterEmbedding2d(nn.Module):
    """Isolates the corner and center pixels of a 2D input tensor, and embeds them
    using a linear layer.
    """

    def __init__(self, channels, emb_dim, norm=True):
        """
        Args:
            channels (int): Number of channels in the input tensor.
            emb_dim (int): Dimension of the embedding space.
            norm (bool): Whether to apply layer normalization after the embedding.
        """
        super().__init__()
        self.channels = channels
        self.emb_dim = emb_dim
        self.embedding = nn.Linear(5 * channels, emb_dim)
        self.act = nn.GELU()
        if norm:
            self.ln = nn.LayerNorm(emb_dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): A tensor of shape (B, C, H, W).
        Returns:
            embedded: torch.Tensor of shape (B, emb_dim).
        """
        B, C, H, W = x.shape
        if C != self.channels:
            raise ValueError(f"Expected input with {self.channels} channels, got {C}.")
        if H < 3 or W < 3:
            raise ValueError("Input height and width must be at least 3.")
        corners_and_center = torch.cat(
            [
                x[:, :, 0, 0],  # top-left
                x[:, :, 0, -1],  # top-right
                x[:, :, -1, 0],  # bottom-left
                x[:, :, -1, -1],  # bottom-right
                x[:, :, H // 2, W // 2],  # center
            ],
            dim=1,
        )  # (B, 5*C)
        embedded = self.embedding(corners_and_center)  # (B, emb_dim)
        embedded = self.act(embedded)
        if hasattr(self, "ln"):
            embedded = self.ln(embedded)
        return embedded


class ConvPatchEmbedding2d(nn.Module):
    """A module that embeds an image into a sequence of patches using
    a 2D convolutional layer.
    """

    def __init__(self, channels, patch_size, emb_dim, mlp_layers=0, norm=True):
        """
        Args:
            channels (int): The number of channels in the image.
            patch_size (int): The size of the patches.
            emb_dim (int): The dimension of the embedding space.
            mlp_layers (int): The number of linear layers to apply after the convolution.
            norm (bool): Whether to apply layer normalization after the embedding.
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

    def forward(self, image):
        """
        Args:
            image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
        Returns:
            embedded_image: torch.Tensor of shape (B, h, w, emb_dim).
        """
        # Compute padding dynamically
        H, W = image.shape[2:]
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        pad = nn.ZeroPad2d((0, pad_w, 0, pad_h))

        image = pad(image)
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

    def __init__(self, dim, max_period=10000):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("Embedding dimension must be even.")
        self.dim = dim
        self.max_period = max_period

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Tensor of shape (..., C) or (..., C, 1) containing the input values.
        Returns:
            torch.Tensor: Tensor of shape (..., C, fourier_dim) containing sinusoidal embeddings.
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

    def __init__(self, fourier_dim):
        """
        Args:
            fourier_dim (int): Dimension of the Fourier features for both spatial
                and temporal embeddings.
        """
        super().__init__()
        self.embedder = SinusoidalEmbedding(fourier_dim)

    def forward(self, coords, times):
        """
        Args:
            coords (torch.Tensor): Tensor of shape (B, 2, H, W) containing lat/lon at each pixel.
            times (torch.Tensor): Tensor of shape (B,) containing time values.
        Returns:
            torch.Tensor: Tensor of shape (B, 3 * F, H, W) containing the embedded coordinates
                where F is the fourier_dim.
        """
        B, _, H, W = coords.shape
        lat_lon = coords.permute(0, 2, 3, 1).reshape(B * H * W, 2)  # (B*H*W, 2)
        times = times.view(B, 1).expand(B, H * W).reshape(B * H * W, 1)  # (B*H*W, 1)
        full_coords = torch.cat([lat_lon, times], dim=1)  # (B*H*W, 3)
        fourier_coords = self.embedder(full_coords)  # (B*H*W, 3, fourier_dim)
        fourier_coords = fourier_coords.view(B, H, W, 3 * fourier_coords.size(-1))
        fourier_coords = fourier_coords.permute(0, 3, 1, 2)  # (B, 3*fourier_dim, H, W)
        return fourier_coords


class SourcetypeEmbedding2d(nn.Module):
    """A class that embeds the values and coordinates of a 2D source, including optional
    characteristic variables.

    This module handles both coordinate embeddings (latitude, longitude,
    and time) and values embeddings (channels, masks, diffusion timestep),
    then outputs their embedded representations.
    Additionally, a conditioning tensor is computed that embeds the conditioning
    that isn't the values or the spatio-temporal coordinates. This includes:
    - The characteristic variables.
    - The diffusion timestep.
    If those elements aren't given, the conditioning tensor is set to None.
    """

    def __init__(
        self,
        channels,
        patch_size,
        values_dim,
        coords_dim,
        cond_dim,
        dff_step_fourier_dim=256,
        coords_fourier_dim=64,
        n_charac_vars=0,
        use_diffusion_t=True,
        pred_mean_channels=0,
        conditioning_mlp_layers=0,
        coords_corner_and_center_embedding=False,
    ):
        """
        Args:
            channels (int): Number of channels for the source data, excluding
                land-sea and availability masks.
            patch_size (int): Size of the patches to be used for convolution.
            values_dim (int): Dimension of the values embedding space.
            coords_dim (int): Dimension of the coordinate embedding space.
            cond_dim (int): Dimension of the conditioning embedding space.
            dff_step_fourier_dim (int): Dimension of the Fourier features for
                the diffusion timestep embedding.
            coords_fourier_dim (int): Dimension of the Fourier features for
                the spatio-temporal coordinates.
            n_charac_vars (int): Number of optional characteristic variables.
            use_diffusion_t (bool): Whether to include a diffusion timestep embedding.
            pred_mean_channels (int): Number of channels for the predicted mean. If zero,
                the predicted mean is not used.
            conditioning_mlp_layers (int): Number of linear layers to apply
                after the conditioning concatenation.
            coords_corner_and_center_embedding (bool): Whether to use a corner-and-center
                embedding for the coordinates instead of a patch embedding. Serves as a
                baseline for ablation studies.
        """
        super().__init__()

        # Values embedding layers
        self.values_patch_size = patch_size
        self.use_diffusion_t = use_diffusion_t
        self.use_predicted_mean = pred_mean_channels > 0

        # landmask + availability mask + predicted mean
        ch = channels + 2 + pred_mean_channels
        self.values_embedding = ConvPatchEmbedding2d(ch, patch_size, values_dim, norm=True)

        # Coords embedding layers
        self.coords_emb_dim = coords_dim
        # - Fourier embedding
        self.coords_fourier_embedder = SpatioTemporalFourierCoordinateEmbedding(coords_fourier_dim)
        # - ViT patch embedding or corner-and-center embedding
        if coords_corner_and_center_embedding:
            self.coords_embedding = CornerAndCenterEmbedding2d(
                coords_fourier_dim * 3, coords_dim, norm=False
            )
        else:
            self.coords_patch_size = patch_size
            self.coords_embedding = ConvPatchEmbedding2d(
                coords_fourier_dim * 3, patch_size, coords_dim, norm=False
            )
        self.coords_norm = nn.LayerNorm(coords_dim)

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
            self.time_mlp = nn.Sequential(
                nn.Linear(dff_step_fourier_dim, cond_dim),
                nn.SiLU(),  # Swish (SiLU) is standard for time embeddings
                nn.Linear(cond_dim, cond_dim),
            )
        # final layer normalization for the conditioning
        self.cond_norm = nn.LayerNorm(cond_dim)

    def forward(self, data):
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
            embedded_values: (B, h, w, values_dim) tensor of embedded values.
            embedded_coords: (B, h, w, coords_dim) tensor of embedded coordinates.
            conditioning: (B, h, w, cond_dim) tensor containing the conditioning, or
                None if no conditioning is given.
        """

        # Embed values
        with torch.backends.cudnn.flags(enabled=self.values_patch_size <= 4):
            values_tensors = []
            values = data["values"]
            avail_mask = data["avail_mask"].unsqueeze(1)
            landmask = data["landmask"].unsqueeze(1)
            values_tensors = [values, avail_mask, landmask]
            if self.use_predicted_mean:
                values_tensors.append(data["pred_mean"])
            values = torch.cat(values_tensors, dim=1)  # (B, C, H, W)
            embedded_values = self.values_embedding(values)

        # Embed coords
        coords = data["coords"]  # (B, 2, H, W)
        dt = data["dt"]  # (B,)
        fourier_coords = self.coords_fourier_embedder(coords, dt)  # (B, 3*F, H, W)
        with torch.backends.cudnn.flags(enabled=self.values_patch_size <= 4):
            embedded_coords = self.coords_embedding(fourier_coords)  # (B, h, w, coords_dim)
        embedded_coords = self.coords_norm(embedded_coords)

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
        conds.append(data["avail"].float().view(-1, 1))  # (B, 1)
        # - Characteristic variables
        if "characs" in data and data["characs"] is not None:
            conds.append(data["characs"])  # (B, n_characs_vars)
        # Concatenate the conditioning tensors
        conds = torch.cat(conds, dim=1)  # (B, ch_cond)
        conds = self.cond_embedding(conds)  # (B, values_dim)
        # Reshape to sum to the spatial dimensions
        conds = conds.view(b, 1, 1, -1)
        available_conditioning.append(conds)
        # - Diffusion timestep (if used)
        if self.use_diffusion_t:
            diffusion_t = data["diffusion_t"]  # (B,)
            # Embed the diffusion timestep with a sinusoidal embedding + MLP
            diffusion_t_emb = self.time_embedder(diffusion_t)  # (B, time_emb_dim)
            diffusion_t_emb = self.time_mlp(diffusion_t_emb)  # (B, cond_dim)
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

        return embedded_values, embedded_coords, conditioning
