"""Implements a Lightning module which receives several sources as input,
tokenizes them, masks a portion of the tokens, and trains a model to predict
the masked tokens."""

from typing import Any, Dict, List, cast

import torch
import torch.nn as nn

from motif.data.source import Source
from motif.datatypes import (
    BatchWithSampleIndexes,
    MultisourceTensor,
    Prediction,
    PreprocessedBatch,
    SourceEmbeddingDict,
)
from motif.lightning_module.base_reconstructor import MultisourceAbstractReconstructor

# Visualization imports
from motif.utils.visualization import display_realizations


class MultisourceDeterministicReconstructor(MultisourceAbstractReconstructor):
    """Given a torch model which receives inputs from multiple sources, this module masks
    some of the sources and trains its backbone to reconstruct the missing data.
    The sources may have different dimensionalities (e.g. 2D, 1D, 0D) and come with
    spatiotemporal coordinates.

    The structure expects its input as a dict D {(source_name, index): map},
    where D[(source_name, index)] contains the following key-value pairs
    (all shapes excluding the batch dimension):
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

    The structure outputs a dict {(source_name, index): tensor} containing
    the predicted values for each source.
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
        loss_max_distance_from_center: int | None = None,
        ignore_land_pixels_in_loss: bool = False,
        normalize_coords_across_sources: bool = False,
        mask_only_sources: list | None = None,
        validation_dir: str | None = None,
        use_modulation_in_output_layers: bool = False,
        metrics: Dict[str, Any] = {},
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
            cond_dim (int or None): If specified, dimension of the conditioning embeddings.
                If None, defaults to values_dim.
            loss_max_distance_from_center (int or None): If specified, only pixels within this
                distance from the center of the storm (in km) will be considered
                in the loss computation.
            ignore_land_pixels_in_loss (bool): If True, the pixels that are on land
                will be ignored in the loss computation.
            normalize_coords_across_sources (bool): If True, the coordinates of each source
                will be normalized across all sources in the batch so that the minimum
                latitude and longitude in each example is 0 and maximum is 1.
                If False, the coordinates will be normalized as sinusoïds.
            mask_only_sources (str or list of str): List of sources to mask. If None, all sources
                may be masked.
            validation_dir (optional, str or Path): Directory where to save the validation plots.
                If None, no plots will be saved.
            metrics (dict of str: callable): Metrics to compute during training and validation.
                A metric should have the signature metric(y_pred, y_true, masks, **kwargs)
                and return a dict {source: tensor of shape (batch_size,)}.
            use_modulation_in_output_layers (bool): If True, applies modulation to the values
                embeddings in the output layers.
            kwargs (dict): Additional arguments to pass to the LightningModule constructor.
        """
        super().__init__(
            sources,
            cfg,
            backbone,
            n_sources_to_mask,
            patch_size,
            dim,
            adamw_kwargs,
            lr_scheduler_kwargs,
            cond_dim=cond_dim,
            loss_max_distance_from_center=loss_max_distance_from_center,
            ignore_land_pixels_in_loss=ignore_land_pixels_in_loss,
            normalize_coords_across_sources=normalize_coords_across_sources,
            mask_only_sources=mask_only_sources,
            validation_dir=validation_dir,
            metrics=metrics,
            use_modulation_in_output_layers=use_modulation_in_output_layers,
            **kwargs,
        )

        # [MASK] token that will replace the embeddings of the masked tokens
        self.mask_token = nn.Parameter(torch.randn(1, self.dim))

    def embed(self, x: PreprocessedBatch) -> SourceEmbeddingDict:
        """Embeds all sources using their corresponding embedding layers.
        Args:
            x: Preprocessed input sources.
        Returns:
            output: Embedded sources.
        """
        # Embeds the values and coordinates using the embedding layers
        y = super().embed(x)
        # We replace the embeddings of the masked sources with a [MASK] token.
        for src, data in y.items():
            src_embedding = data.embedding  # (B, ..., Dv)
            where_masked = x[src].avail == 0  # 0 means masked
            where_masked = where_masked.view(
                (where_masked.shape[0],) + (1,) * (src_embedding.dim() - 1)
            )
            token = self.mask_token.view((1,) * (src_embedding.dim() - 1) + (-1,))
            data.embedding = torch.where(where_masked, token, src_embedding)
        return y

    def mask(
        self, x: PreprocessedBatch, sample_indices: List[int] | None = None
    ) -> PreprocessedBatch:
        """Masks a portion of the sources. A missing source cannot be chosen to be masked.
        Supposes that there are at least as many non-missing sources as the number of sources
        to mask. The number of sources to mask is determined by self.masking ratio.

        Masked sources have their values replaced by the [MASK] token.

        The availability flag is set to 0 where the source is masked.
        Args:
            x : The input sources.
            sample_indices: If specified, a list of integers of length (B,)
                containing the indices of the samples in the dataset. Used to seed the RNG
                for reproducible masking.
        Returns:
            masked_x (dict of (source_name, index) to dict of str to tensor):
                The input sources with a portion of the sources masked.
        """
        # Choose the sources to mask.
        avail_flags = super().select_sources_to_mask(x, sample_indices=sample_indices)
        # avail_flags[s][i] == 0 if the source s should be masked.

        # We just need to update the avail flag of each source. Where the flag is set to 0,
        # the embeddings will be replaced by the [MASK] token in the embedding step.
        masked_x = {}
        for src, data in x.items():
            # Copy the data to avoid modifying the original dict
            masked_data = data.clone()
            avail_flag = avail_flags[src]
            masked_data.avail = avail_flag
            # Set the availability mask to 0 everywhere for noised sources.
            # (!= from the avail flag)
            masked_data.avail_mask[avail_flag == 0] = 0
            masked_x[src] = masked_data
        return masked_x

    def compute_loss(
        self, pred: MultisourceTensor, batch: PreprocessedBatch, masked_batch: PreprocessedBatch
    ) -> torch.Tensor:
        avail_flag = {src: data.avail for src, data in masked_batch.items()}
        targets = {src: batch[src].values for src in batch}

        # Only the keep the output variables from the ground truth
        targets = self.filter_output_variables(targets)
        # Compute the loss masks: a dict {(s,i): M} where M is a binary mask of shape
        # (B, ...) indicating which points should be considered in the loss.
        loss_masks = self.compute_loss_mask(batch, avail_flag)

        # Compute the MSE loss for each source
        losses: MultisourceTensor = {}
        for src in pred:
            # Compute the pointwise loss for each source.
            source_loss = (pred[src] - targets[src]).pow(2)
            # Multiply by the loss mask
            source_loss_mask = loss_masks[src].unsqueeze(1).expand_as(source_loss)
            source_loss = source_loss * source_loss_mask
            # Compute the mean over the number of available points
            mask_sum = source_loss_mask.sum()
            if mask_sum == 0:
                # If all points are masked, we skip the loss computation for this source
                continue
            losses[src] = source_loss.sum() / mask_sum

        # Compute the total loss
        loss = sum(losses.values()) / len(losses)
        loss = cast(torch.Tensor, loss)
        return loss

    def training_step(self, batch: BatchWithSampleIndexes, batch_idx: int) -> torch.Tensor:
        _, raw_batch = batch
        batch_size = raw_batch[list(raw_batch.keys())[0]].values.shape[0]
        preproc_batch = self.preproc_input(raw_batch)
        # Mask the sources
        masked_x = self.mask(preproc_batch)
        # Make predictions
        pred = self.forward(masked_x)
        # Compute the loss
        loss = self.compute_loss(pred, preproc_batch, masked_x)

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, batch: BatchWithSampleIndexes, batch_idx: int) -> torch.Tensor:
        _, raw_batch = batch
        preproc_batch = self.preproc_input(raw_batch)
        # Mask the sources
        masked_batch = self.mask(preproc_batch)
        # Make predictions
        pred = self.forward(masked_batch)
        # Compute the loss
        loss = self.compute_loss(pred, preproc_batch, masked_batch)
        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=True,
        )

        avail_flags = {src: masked_batch[src].avail for src in masked_batch}
        if self.validation_dir is not None and batch_idx % 2 == 0:
            # For every 2 batches, make a prediction and display it.
            if batch_idx % 2 == 0:
                display_realizations(
                    Prediction(pred=pred, avail=avail_flags),
                    raw_batch,
                    self.validation_dir / f"realizations_{batch_idx}",
                    display_fraction=1.0,
                )

        # Evaluate the metrics
        y_true = {src: raw_batch[src].values for src in raw_batch}
        y_true = self.filter_output_variables(y_true)
        masks = self.compute_loss_mask(raw_batch, avail_flags)
        for metric_name, metric in self.metrics.items():
            metric_res = metric(pred, y_true, masks)
            # Compute the average metric over all sources
            avg_res = torch.stack(list(metric_res.values())).mean()
            self.log(
                f"val_{metric_name}",
                avg_res,
                on_epoch=True,
                on_step=False,
                sync_dist=True,
            )
        return loss

    def predict_step(self, batch: BatchWithSampleIndexes, batch_idx: int) -> Prediction:
        sample_indices, input_batch = batch
        preproc_batch = self.preproc_input(input_batch)
        # Mask the sources
        masked_batch = self.mask(preproc_batch, sample_indices=sample_indices)
        # Make predictions
        pred = self.forward(masked_batch)
        # Fetch the availability flags for each source, so that
        # whatever processes the output can know which elements
        # were masked.
        avail_flags = {src: masked_batch[src].avail for src in masked_batch}
        return Prediction(pred=pred, avail=avail_flags)
