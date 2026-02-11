"""Implements a Lightning module which receives several sources as input,
tokenizes them, masks a portion of the tokens, and trains a model to predict
the masked tokens."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List

import lightning.pytorch as pl
import torch
from torch import Tensor

from motif.data.grid_functions import distance_to_overlap
from motif.data.source import Source
from motif.datatypes import (
    Batch,
    BatchWithSampleIndexes,
    MultisourceTensor,
    Prediction,
    PreprocessedBatch,
    SourceData,
)

# Local module imports
from motif.utils.scheduler import CosineAnnealingWarmupRestarts


class MultisourceAbstractModule(pl.LightningModule, ABC):
    """Given a torch model which receives inputs from multiple sources and outputs
    predictions for each source. This classes handles the preprocessing common to its
    subclasses and the filtering of the loss.

    The sources may have different dimensionalities (e.g. 2D, 1D, 0D) and come with
    spatiotemporal coordinates.

    The structure expects its input as a dict D {(source_name, index): map}, where D[(source_name, index)] contains the
    following key-value pairs (all shapes excluding the batch dimension):
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
        adamw_kwargs: Dict[str, Any],
        lr_scheduler_kwargs: Dict[str, Any],
        loss_max_distance_from_center: int | None = None,
        ignore_land_pixels_in_loss: bool = False,
        max_overlap_distance: float | None = None,
        normalize_coords_across_sources: bool = False,
        validation_dir: str | None = None,
        metrics: Dict[str, Callable] = {},
    ):
        """
        Args:
            sources (list of Source): Source objects defining the dataset's sources.
            cfg (dict): The whole configuration of the experiment. This will be saved
                within the checkpoints and can be used to rebuild the exact experiment.
            adamw_kwargs (dict): Arguments to pass to torch.optim.AdamW (other than params).
            lr_scheduler_kwargs (dict): Arguments to pass to the learning rate scheduler.
            loss_max_distance_from_center (int or None): If specified, only pixels within this
                distance from the center of the storm (in km) will be considered
                in the loss computation.
            ignore_land_pixels_in_loss (bool): If True, the pixels that are on land
                will be ignored in the loss computation.
            max_overlap_distance (float): Maximum distance (in km) to consider that two sources
                overlap. Used to compute the overlap mask for the loss.
            normalize_coords_across_sources (bool): If True, the coordinates of each source
                will be normalized across all sources in the batch so that the minimum
                latitude and longitude in each example is 0 and maximum is 1.
                If False, the coordinates will be normalized as sinusoïds.
            validation_dir (optional, str or Path): Directory where to save the validation plots.
                If None, no plots will be saved.
            metrics (dict of str: callable): Metrics to compute during training and validation.
                A metric should have the signature metric(y_pred, y_true, masks, **kwargs)
                and return a dict {source: tensor of shape (batch_size,)}.
            **kwargs: Additional arguments to pass to the LightningModule constructor.
        """
        super().__init__()

        self.sources = {source.name: source for source in sources}
        self.source_names = [source.name for source in sources]
        self.validation_dir = Path(validation_dir) if validation_dir is not None else None
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

        self.loss_max_distance_from_center = loss_max_distance_from_center
        self.ignore_land_pixels_in_loss = ignore_land_pixels_in_loss
        self.max_overlap_distance = max_overlap_distance
        self.adamw_kwargs = adamw_kwargs
        self.metrics = metrics
        self.normalize_coords_across_sources = normalize_coords_across_sources

        # Save the configuration so that it can be loaded from the checkpoints
        self.cfg = cfg
        self.save_hyperparameters(ignore=["backbone", "metrics"])

    def preproc_input(self, x: Batch) -> PreprocessedBatch:
        """Preprocesses the input data before feeding it to the model."""
        preproc_x: PreprocessedBatch = {}
        for i, (src, data) in enumerate(x.items()):
            c, v = data.coords.float(), data.values.float()
            lm, d = data.landmask.float(), data.dist_to_center.float()
            dt = data.dt.float()
            am = data.avail_mask.float()
            # Don't modify the tensors in-place, as we need to keep the NaN values
            # for the loss computation
            dt = torch.nan_to_num(dt, nan=-1.0)
            v = torch.nan_to_num(v, nan=0)
            lm = torch.nan_to_num(lm, nan=0)
            # Where the coords are NaN, set them to 0
            c = torch.nan_to_num(c, nan=0)
            # For the distance tensor, fill the nan values with +inf
            d = torch.nan_to_num(d, nan=float("inf"))
            # Potential characs variables
            ch = None
            if data.characs is not None:
                ch = data.characs.float()
                ch = torch.nan_to_num(ch, nan=0)

            preproc_x[src] = SourceData(
                values=v,
                coords=c,
                dt=dt,
                avail=data.avail,
                dist_to_center=d,
                landmask=lm,
                characs=ch,
                avail_mask=am,
            )

        return preproc_x

    @abstractmethod
    def forward(self, x: PreprocessedBatch) -> MultisourceTensor:
        pass

    def filter_output_variables(self, y_true: MultisourceTensor) -> MultisourceTensor:
        """Given the groundtruth, only keeps the output variables (as defined
        in the Source class).
        Args:
            y_true: The groundtruth, of shape (B, C, ...).
        Returns:
            y_true_out: The filtered groundtruth, of shape (B, C_filtered, ...).
        """
        y_true_out = {}
        for src, true_s in y_true.items():
            # 1. Retrieve the list of output variables for the source and only keep those
            # in the groundtruth.
            source_name = src.name
            output_vars = self.sources[source_name].get_output_variables_mask()
            output_vars_tensor = Tensor(output_vars).to(true_s.device).bool()
            true_s = true_s[:, output_vars_tensor]  # (B, C, ...) to (B, C_filtered, ...)
            y_true_out[src] = true_s
        return y_true_out

    def compute_loss_mask(
        self,
        batch: Batch,
        avail_flag: MultisourceTensor,
    ) -> MultisourceTensor:
        """Computes a binary mask M on the tokens of the sources of shape (B, ...)
        such that M[b, ...] = True if and only if the following conditions are met:
        - Exclude the tokens that were missing in the ground truth.
        - Only consider the tokens that were masked in the input.
        - Optionally exclude the tokens that are too far from the center
          of the storm.
        - Optionally exclude the tokens that are on land.
        Args:
            batch: Input batch, before preprocessing.
            avail_flag: The availability flags for each source.
        Returns:
            masks: The masks, of shape (B, ...),
                valued 1 where the token is available and 0 otherwise.
        """

        masks = {}
        for src, source_data in batch.items():
            # We'll compute a mask M on the tokens of the source of shape (B, C, ...)
            # such that M[b, ...] = True if and only if the following conditions are met:
            # - The source was masked for the sample b (avail flag == 0);
            # - the value at position ... was not missing (true_data["am"] == True);
            loss_mask = source_data.avail_mask >= 1  # (B, ...)
            loss_mask[avail_flag[src] != 0] = False

            # If a maximum distance from the center is specified, exclude the pixels
            # that are too far from the center from the loss computation.
            if self.loss_max_distance_from_center is not None:
                dist_mask = source_data.dist_to_center <= self.loss_max_distance_from_center
                loss_mask = loss_mask & dist_mask

            # Optionally ignore the pixels that are on land
            if self.ignore_land_pixels_in_loss:
                loss_mask[source_data.landmask > 0] = False

            # Overlap masking: exclude from the loss the pixels of the target source
            # where there is no overlap with any other source.
            if self.max_overlap_distance is not None:
                other_coords = [s.coords for k, s in batch.items() if k != src]
                dist_to_overlap = distance_to_overlap(source_data.coords, *other_coords)
                overlap_mask = dist_to_overlap <= self.max_overlap_distance
                loss_mask = loss_mask & overlap_mask

            masks[src] = loss_mask

        return masks

    def configure_optimizers(self) -> Dict[str, Any]:  # type: ignore
        """Configures the optimizer for the model.
        The optimizer is AdamW, with the default parameters.
        The learning rate goes through a linear warmup phase, then follows a cosine
        annealing schedule.
        """
        decay = self.adamw_kwargs.pop("weight_decay", 0.0)
        params = {k: v for k, v in self.named_parameters() if v.requires_grad}

        # Apply weight decay only to the weights that are not in the normalization layers
        decay_params = {k for k, _ in params.items() if "weight" in k and "norm" not in k}
        optimizer = torch.optim.AdamW(
            [
                # Parameters without decay
                {"params": [v for k, v in params.items() if k not in decay_params]},
                # Parameters with decay
                {
                    "params": [v for k, v in params.items() if k in decay_params],
                    "weight_decay": decay,
                },
            ],
            **self.adamw_kwargs,
        )

        scheduler_interval = self.lr_scheduler_kwargs.pop("interval", "epoch")
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            **self.lr_scheduler_kwargs,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": scheduler_interval,
                "frequency": 1,
            },
        }

    @staticmethod
    def erase_source_data(data: SourceData, to_erase: Tensor) -> SourceData:
        """Given the data of a source, erases the data for the samples where to_erase is True.
        Erasing a source means setting its data to what it would be if the source had
        been unavailable when fed to self.preproc_input().
        Args:
            data (dict of str to tensor): The data of a source, as output by self.preproc_input().
            to_erase (Tensor): Boolean tensor of shape (B,) indicating which samples
                should be erased.
        Returns:
            new_data (dict of str to tensor): The erased data of the source, where every tensor
                has the shape (B, ...).
        """  # For every sample where the source is available, modify the entries
        # so that they are as if the source was missing according to self.preproc_input().
        avail = torch.where(to_erase, -1, data.avail)
        dt = torch.where(to_erase, -1.0, data.dt)
        coords = torch.where(
            to_erase.view((-1,) + (1,) * (data.coords.ndim - 1)),
            -torch.ones_like(data.coords),
            data.coords,
        )
        values = torch.where(
            to_erase.view((-1,) + (1,) * (data.values.ndim - 1)),
            torch.zeros_like(data.values),
            data.values,
        )
        characs = None
        if data.characs is not None:
            characs = torch.where(
                to_erase.unsqueeze(1), torch.zeros_like(data.characs), data.characs
            )
        landmask = torch.where(
            to_erase.view((-1,) + (1,) * (data.landmask.ndim - 1)),
            torch.zeros_like(data.landmask),
            data.landmask,
        )
        avail_mask = torch.where(
            to_erase.view((-1,) + (1,) * (data.avail_mask.ndim - 1)),
            torch.full_like(data.avail_mask, -1),
            data.avail_mask,
        )
        dist_to_center = torch.where(
            to_erase.view((-1,) + (1,) * (data.dist_to_center.ndim - 1)),
            torch.full_like(data.dist_to_center, float("inf")),
            data.dist_to_center,
        )
        new_data = SourceData(
            values=values,
            coords=coords,
            dt=dt,
            avail=avail,
            avail_mask=avail_mask,
            dist_to_center=dist_to_center,
            landmask=landmask,
            characs=characs,
        )
        return new_data

    def to_unconditional_batch(
        self,
        batch: PreprocessedBatch,
        which_samples: Tensor | None = None,
    ) -> PreprocessedBatch:
        """Given a batch where some of the sources are masked, creates an unconditional
        copy of the batch where the unmasked sources are erased.
        Erasing a source means setting its data to what it would be if the source had
        been unavailable when fed to self.preproc_input().
        A source is considered masked if its availability flag is 0.
        Args:
            batch: The input batch.
            which_samples: If provided, boolean tensor of shape (B,)
                indicating which samples in the batch should be made unconditional. The others
                will be left unchanged. If None, all samples will be made unconditional.
        Returns:
            new_batch: The unconditional copy
                of the batch, where every tensor has the shape (B, ...).
                The unmasked sources are erased, i.e. set to what they would be if
                the source had been unavailable when fed to self.preproc_input().
        """
        new_batch = {}
        for src, data in batch.items():
            new_data = {}
            to_erase = data.avail == 1  # (B,)
            if which_samples is not None:
                to_erase = to_erase & which_samples
            new_data = self.erase_source_data(data, to_erase)
            new_batch[src] = new_data
        return new_batch

    @abstractmethod
    def training_step(self, batch: BatchWithSampleIndexes, batch_idx: int) -> Tensor:
        pass

    @abstractmethod
    def validation_step(self, batch: BatchWithSampleIndexes, batch_idx: int) -> Tensor:
        pass

    @abstractmethod
    def predict_step(self, batch: BatchWithSampleIndexes, batch_idx: int) -> Prediction:
        pass
