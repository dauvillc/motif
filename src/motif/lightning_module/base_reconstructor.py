"""Implements a Lightning module which receives several sources as input,
tokenizes them, masks a portion of the tokens, and trains a model to predict
the masked tokens."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, cast

import torch
import torch.nn as nn
from torch import Tensor

from motif.data.source import Source
from motif.datatypes import (
    BatchWithSampleIndexes,
    MultisourceTensor,
    Prediction,
    PreprocessedBatch,
    SourceData,
    SourceEmbedding,
    SourceEmbeddingDict,
)

# Local module imports
from motif.lightning_module.base_module import MultisourceAbstractModule
from motif.models.embedding_layers import SourcetypeEmbedding2d
from motif.models.output_layers import SourcetypeProjection2d


class MultisourceAbstractReconstructor(MultisourceAbstractModule, ABC):
    """Given a torch model which receives inputs from multiple sources, this module masks
    some of the sources and trains its backbone to reconstruct the missing data.
    The sources may have different dimensionalities (e.g. 2D, 1D, 0D) and come with
    spatiotemporal coordinates.

    The structure expects its input as a dict D {(source_name, index): map}, where D[(source_name, index)] contains the
    following key-value pairs (all shapes excluding the batch dimension):
    - "id" is a list of strings of length (B,) each uniquely identifying the elements.
    - "avail" is a scalar tensor of shape (B,) containing 1 if the element is available
        and -1 otherwise.
    - "values" is a tensor of shape (B, C, ...) containing the values of the source.
    - "landmask" is a tensor of shape (B, ...) containing the land-sea mask of the source.
    - "coords" is a tensor of shape (B, 2, ...) containing the spatial coordinates of the source.
    - "dt" is a scalar tensor of shape (B,) containing the time delta between the reference time
        and the element's time, normalized by dt_max.
    - "dist_to_center" is a tensor of shape (B, ...) containing the distance
        to the center of the storm at each pixel, in km.

    The structure outputs a dict {(source_name, index): tensor} containing the predicted values
    for each source.
    """

    def __init__(
        self,
        sources: List[Source],
        cfg: Dict[str, Any],
        backbone: nn.Module,
        n_sources_to_mask: int,
        patch_size: int,
        dim: int,
        adamw_kwargs: Dict[str, Any],
        lr_scheduler_kwargs: Dict[str, Any],
        cond_dim: int | None = None,
        coords_encoding_method: str = "fourier",
        coords_dim: int | None = None,
        loss_max_distance_from_center: int | None = None,
        ignore_land_pixels_in_loss: bool = False,
        normalize_coords_across_sources: bool = False,
        mask_only_sources: list | None = None,
        validation_dir: str | None = None,
        metrics: Dict[str, Callable] = {},
        use_modulation_in_output_layers: bool = False,
        conditioning_mlp_layers: int = 0,
        sources_selection_seed: int = 123,
        **kwargs,
    ):
        """
        Args:
            sources (list of Source): Source objects defining the dataset's sources.
            cfg (dict): The whole configuration of the experiment. This will be saved
                within the checkpoints and can be used to rebuild the exact experiment.
            backbone (nn.Module): Backbone model to train.
            n_sources_to_mask (int): Number of sources to mask in each sample.
            patch_size (int): Size of the patches to split the images into.
            dim (int): Embedding dimension.
            adamw_kwargs (dict): Arguments to pass to torch.optim.AdamW (other than params).
            lr_scheduler_kwargs (dict): Arguments to pass to the learning rate scheduler.
            cond_dim (int or None): Dimension of the conditioning embedding.
                If None, defaults to dim.
            coords_encoding_method (str): Method to encode the coordinates, either:
                - "fourier": Use Fourier features to embed the coordinates.
                - "avg_pool": Use average pooling to embed the coordinates.
            coords_dim (int, optional): Dimension of the coordinate embedding if
                using fourier encoding.
            loss_max_distance_from_center (int or None): If specified, only pixels within this
                distance from the center of the storm (in km) will be considered
                in the loss computation.
            ignore_land_pixels_in_loss (bool): If True, the pixels that are on land
                will be ignored in the loss computation.
            normalize_coords_across_sources (bool): If True, the coordinates of each source
                will be normalized across all sources in the batch so that the minimum
                latitude and longitude in each example is 0 and maximum is 1.
                If False, the coordinates will be normalized as sinusoïds.
            mask_only_sources (str or list of str): List of source names to mask. If not None,
                the masked source will chosen among those only.
            validation_dir (optional, str or Path): Directory where to save the validation plots.
                If None, no plots will be saved.
            metrics (dict of str: callable): Metrics to compute during training and validation.
                A metric should have the signature metric(y_pred, y_true, masks, **kwargs)
                and return a dict {source: tensor of shape (batch_size,)}.
            use_modulation_in_output_layers (bool): If True, applies modulation to the output layers.
            conditioning_mlp_layers (int): Number of linear layers to apply in the conditioning
                embedding after the concatenation of the different conditioning variables.
            sources_selection_seed (int, optional): Seed for the random number generator used to select
                the sources to mask.
            **kwargs: Additional arguments to pass to the LightningModule constructor.
        """
        super().__init__(
            sources,
            cfg,
            adamw_kwargs,
            lr_scheduler_kwargs,
            loss_max_distance_from_center=loss_max_distance_from_center,
            ignore_land_pixels_in_loss=ignore_land_pixels_in_loss,
            normalize_coords_across_sources=normalize_coords_across_sources,
            validation_dir=validation_dir,
            metrics=metrics,
            **kwargs,
        )

        self.backbone = backbone
        self.n_sources_to_mask = n_sources_to_mask
        self.patch_size = patch_size
        self.dim = dim
        self.cond_dim = cond_dim if cond_dim is not None else dim
        self.coords_encoding_method = coords_encoding_method
        self.coords_dim = coords_dim

        # RNG that will be used to select the sources to mask
        self.source_select_gen = torch.Generator().manual_seed(sources_selection_seed)
        # We'll pre-generate a tensor of seeds for each sample in the predict_step. This will
        # ensure reproducibility when predicting with different batch sizes.
        self.predict_step_seeds = torch.randint(
            0, 2**32 - 1, (1_000_000,), generator=self.source_select_gen
        )

        if isinstance(mask_only_sources, str):
            mask_only_sources = [mask_only_sources]
        self.mask_only_sources = mask_only_sources

        # Initialize the embedding layers
        self.init_embedding_layers(
            use_modulation_in_output_layers,
            conditioning_mlp_layers=conditioning_mlp_layers,
        )

    def init_embedding_layers(
        self,
        use_modulation_in_output_layers: bool,
        conditioning_mlp_layers: int = 0,
    ):
        """Initializes the weights of the embedding layers."""
        if not hasattr(self, "use_diffusion_t"):
            self.use_diffusion_t = False
        if not hasattr(self, "use_det_model"):
            self.use_det_model = False
        # Embedding and output projection layers
        # An embedding and an output layer for each source type
        # - We need to retrieve the list of each source type from the sources,
        #   as well as the number of characs variables for each source type.
        self.sourcetypes_characs_vars = {}
        self.sourcetype_embeddings = nn.ModuleDict()
        self.sourcetype_output_projs = nn.ModuleDict()
        for source in self.sources.values():
            # Only create the embedding layer for that source type if it doesn't exist yet
            if source.type not in self.sourcetypes_characs_vars:
                self.sourcetypes_characs_vars[source.type] = source.n_charac_variables()
                n_input_channels = source.n_input_variables()
                n_output_channels = source.n_output_variables()
                # Whether to include a predicted mean in the embedding layer
                pred_mean_channels = n_output_channels if self.use_det_model else 0
                # Create the embedding layers for that source type depending on
                # its dimensionality
                if source.dim == 2:
                    self.sourcetype_embeddings[source.type] = SourcetypeEmbedding2d(
                        n_input_channels,
                        self.patch_size,
                        self.dim,
                        self.cond_dim,
                        n_charac_vars=source.n_charac_variables(),
                        coords_encoding_method=self.coords_encoding_method,
                        coords_dim=self.coords_dim,
                        use_diffusion_t=self.use_diffusion_t,
                        pred_mean_channels=pred_mean_channels,
                        conditioning_mlp_layers=conditioning_mlp_layers,
                    )
                    self.sourcetype_output_projs[source.type] = SourcetypeProjection2d(
                        self.dim,
                        n_output_channels,
                        self.patch_size,
                        cond_dim=self.cond_dim,
                        use_modulation=use_modulation_in_output_layers,
                    )
                else:
                    raise NotImplementedError(
                        f"Embedding layers for source type {source.type} with "
                        f"dimensionality {source.dim} are not implemented yet."
                    )

            else:
                # Check that the number of characs variables is the same for all sources
                # of the same type
                if self.sourcetypes_characs_vars[source.type] != source.n_charac_variables():
                    raise ValueError(
                        "Number of characs variables is not "
                        "the same for all sources of type {source.type}"
                    )

    def embed(self, x: PreprocessedBatch) -> SourceEmbeddingDict:
        """Embeds all sources using their corresponding embedding layers.
        Args:
            x: Preprocessed input sources.
        Returns:
            output: Embedded sources.
        """
        output: SourceEmbeddingDict = {}
        for src, data in x.items():
            source_name = src.name
            source_obj = self.sources[source_name]

            # Only keep the current source's input variables from the values.
            input_mask = torch.tensor(
                source_obj.get_input_variables_mask(),
                device=data.values.device,
                dtype=torch.bool,
            )
            values = data.values[:, input_mask]  # (B, C_in, ...)
            if data.pred_mean is not None:
                pred_mean = data.pred_mean[:, input_mask]  # (B, C_in, ...)
            else:
                pred_mean = None

            emb_data = SourceData(
                values=values,
                coords=data.coords,  # The other fields are unchanged
                dt=data.dt,
                avail=data.avail,
                avail_mask=data.avail_mask,
                dist_to_center=data.dist_to_center,
                landmask=data.landmask,
                characs=data.characs,
                diffusion_t=data.diffusion_t,
                pred_mean=pred_mean,
            )

            # Run the embedding layer corresponding to the source type
            source_type = source_obj.type
            src_embedding = self.sourcetype_embeddings[source_type](emb_data)

            output[src] = src_embedding

        return output

    def select_sources_to_mask(
        self, x: PreprocessedBatch, sample_indices: List[int] | None = None
    ) -> MultisourceTensor:
        """Given a multi-sources batch, randomly selects a source to mask in each sample.
        Does not actually perform the masking.
        Args:
            x: Input sources.
            sample_indices: Indices of the samples in the batch.
                If provided, will use these indices to seed the random number generator
                for reproducibility.
        Returns:
            avail_flags: The availability flags for each source,
                as tensors of shape (B,), such that:
                * avail_flags[s][i] == 1 if the source s is available for the sample i,
                * avail_flags[s][i] == 0 if the source s is masked for the sample i,
                * avail_flags[s][i] == -1 if the source s is missing for the sample i.
        """
        n_sources = len(x)
        any_elem = next(iter(x.values())).values
        batch_size = any_elem.shape[0]
        device = any_elem.device

        # Select the sources to mask, which can differ between samples in the batch.
        # Missing sources cannot be masked.
        # Strategy: we'll generate a random noise tensor of shape (B, n_sources)
        # and for each row, mask the sources with the highest noise.
        if sample_indices is None:
            noise = torch.rand((batch_size, n_sources), generator=self.source_select_gen).to(device)
        else:
            # Generate a noise tensor using a seed that depends on the sample index
            seeds = self.predict_step_seeds[sample_indices].to(device)
            noise = torch.stack(
                [
                    torch.rand(
                        (n_sources,),
                        generator=torch.Generator().manual_seed(cast(int, seed.item())),
                    )
                    for seed in seeds
                ],
                dim=0,
            ).to(device)
        for i, (src, data) in enumerate(x.items()):
            # Multiply the noise by the availability mask (-1 for missing sources, 1 otherwise)
            noise[:, i] = noise[:, i] * data.avail.squeeze(-1)
        # If only masking among a subset of sources, set the noise of the other sources to -2
        if self.mask_only_sources is not None:
            for i, src in enumerate(x):
                source_name = src.name
                if source_name not in self.mask_only_sources:
                    noise[:, i] = -2.0  # Lower than the minimum possible noise (-1)
        # Gather the indices of the sources to mask for each sample
        _, sources_to_mask = noise.topk(self.n_sources_to_mask, dim=1)  # (B, n_sources_to_mask)
        # Deduce a matrix M of shape (B, n_sources) such that M[b, i] = 1 if the source i
        # should be masked for the sample b, and 0 otherwise.
        masked_sources_matrix = torch.zeros(
            (batch_size, n_sources), dtype=torch.bool, device=device
        )  # (B, n_sources)
        masked_sources_matrix.scatter_(1, sources_to_mask, True)
        # Deduce the availability flags for each source
        avail_flags = {}
        for i, (src, data) in enumerate(x.items()):
            avail_flag = data.avail.clone()
            avail_flag[masked_sources_matrix[:, i]] = 0
            avail_flags[src] = avail_flag
        return avail_flags

    def forward(self, x: PreprocessedBatch) -> MultisourceTensor:
        """Computes the forward pass of the model.
        Args:
            x: The input sources, masked.
        Returns:
            y: The predicted values for each source.
        """
        # Save the shape of the tokens before they're embedded, so that we can
        # later remove the padding.
        spatial_shapes = {
            src: data.values.shape[2:] for src, data in x.items() if len(data.values.shape) > 2
        }
        # Embed and mask the sources
        embed_x: SourceEmbeddingDict = self.embed(x)

        # Run the transformer backbone
        backbone_out: MultisourceTensor = self.backbone(embed_x)

        # Create a source embedding dict with updated embeddings for the projection layers
        embed_out = {
            src: SourceEmbedding(
                embedding=backbone_out[src],
                conditioning=embed_x[src].conditioning,
                coords=embed_x[src].coords,
            )
            for src in backbone_out
        }

        pred: MultisourceTensor = {}
        for src, y in embed_out.items():
            # Project from latent space to values space using the output layer
            # corresponding to the source type
            source_name = src.name
            src_type = self.sources[source_name].type
            pred[src] = self.sourcetype_output_projs[src_type](y)
        # For 2D sources, remove the padding
        for src, spatial_shape in spatial_shapes.items():
            pred[src] = pred[src][..., : spatial_shape[0], : spatial_shape[1]]

        return pred

    @abstractmethod
    def training_step(self, batch: BatchWithSampleIndexes, batch_idx: int) -> Tensor:
        pass

    @abstractmethod
    def validation_step(self, batch: BatchWithSampleIndexes, batch_idx: int) -> Tensor:
        pass

    @abstractmethod
    def predict_step(self, batch: BatchWithSampleIndexes, batch_idx: int) -> Prediction:
        pass
