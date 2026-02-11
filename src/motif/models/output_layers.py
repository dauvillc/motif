"""Implements layers to project the output of a ViT in latent space to the output space."""

from torch import Tensor, nn

from motif.datatypes import SourceEmbedding
from motif.models.icnr import ICNR


class SourcetypeProjection2d(nn.Module):
    """Receives the embeddings and conditioning of a source and projects them to
    that source's original space. Meant to be shared across all sources of the same type.
    """

    def __init__(
        self,
        dim: int,
        out_channels: int,
        patch_size: int,
        cond_dim: int,
        use_modulation: bool = False,
    ):
        """
        Args:
            dim (int): Embedding dimension.
            out_channels (int): Number of channels in the output space.
            patch_size (int): Size of the embedding patches.
            cond_dim (int): Dimension of the conditioning embedding.
            use_modulation (bool): If True, applies modulation to the embeddings.
        """
        super().__init__()
        self.patch_size = patch_size

        self.norm = nn.LayerNorm(dim)
        if use_modulation:
            self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * dim))

        # Subpixel convolution to project the latent space to the output space
        self.conv = nn.Conv2d(
            in_channels=dim,
            out_channels=out_channels * patch_size**2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.pixel_shuffle = nn.PixelShuffle(patch_size)
        # Apply the ICNR initialization to the deconvolution, to reduce checkerboard artifacts
        weight = ICNR(
            self.conv.weight, initializer=nn.init.kaiming_normal_, upscale_factor=patch_size
        )
        self.conv.weight.data.copy_(weight)

    def forward(self, source_embedding: SourceEmbedding) -> Tensor:
        """
        Args:
            source_embedding: A SourceEmbedding object containing:
                - embedding: Embedding tensors of shape (B, h, w, Dv).
                - conditioning: Embedded conditioning of shape (B, h, w, Dc) or None.
        Returns:
            torch.Tensor of shape (B, channels, H, W) containing the projected output.
        """
        x = source_embedding.embedding  # (B, h, w, Dv)
        cond = source_embedding.conditioning  # (B, h, w, Dc) or None

        x = self.norm(x)

        if hasattr(self, "modulation") and cond is not None:
            # Apply the modulation to the embeddings
            shift, scale = self.modulation(cond).chunk(2, dim=-1)
            x: Tensor = (1 + scale) * x + shift

        # Transpose x from (B, h, w, Dv) to (B, Dv, h, w)
        x = x.permute(0, 3, 1, 2)
        # Deconvolve the latent space using subpixel convolutions
        x = self.conv(x)
        x = self.pixel_shuffle(x)  # (B, C, H, W)

        return x
