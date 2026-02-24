"""Implements the MultiSourceWriter class"""

import gc
import os
from pathlib import Path
from typing import Sequence, cast

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import xarray as xr
from lightning.pytorch.callbacks import BasePredictionWriter

from motif.data.dataset import MultiSourceDataset
from motif.datatypes import BatchWithSampleIndexes, GenerativePrediction, Prediction
from motif.lightning_module.base_reconstructor import MultisourceAbstractReconstructor


def atomic_to_netcdf(ds: xr.Dataset, path: Path):
    """Atomic write to netcdf file. Absolutely written by an LLM."""
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    ds.to_netcdf(tmp)
    try:
        os.replace(tmp, path)  # atomic move on POSIX
    except FileExistsError:
        # Someone else already wrote the real file
        tmp.unlink(missing_ok=True)


class MultiSourceWriter(BasePredictionWriter):
    """Can be used as callback to a Lightning Trainer to write to disk the predictions of a model
    such that:
    - The targets are of the form {source_name: D} where D is a dictionary with the following keys:
        - dt: The datetime of the observation
        - values: The values to predict as a tensor of shape (bs, C, ...)
            where ... are the spatial dimensions and C is the number of channels.
        - coords: The coordinates of each pixel as a tensor of shape (bs, 2, ...).
            The first channel is the latitude and the second channel is the longitude.
    - The predictions are of the form {source_name: v'} where v' are the predicted values.
        A source may not be included in the predictions.

    The writer automatically detects whether the predictions include multiple realizations
    (flow matching models) or a single prediction (deterministic models) based on the keys
    in the prediction dictionary.

    The predictions are written to disk in the following format:
    root_dir/targets/source_name/index/<sample_index>.nc
    root_dirpredictions/source_name/index/<sample_index>.nc
    root_dirtrue_vf/source_name/index/<sample_index>.nc
    root_dirembeddings/source_name/index/<sample_index>.nc
    Additionally, for each rank a file root_dir/info_<rank>.csv is written
    and contains a DataFrame with the metadata.
    """

    def __init__(self, root_dir: Path, dataset: MultiSourceDataset, mode="w"):
        """
        Args:
            root_dir: The root directory where the predictions will be written.
            dataset: Dataset object.
            mode (str): Writing mode, either 'w' to erase any pre-existing info data,
                or 'a' to append to existing data.
        """
        super().__init__(write_interval="batch")
        self.root_dir = Path(root_dir)
        self.dt_min, self.dt_max = dataset.dt_min_norm, dataset.dt_max_norm
        self.dataset = dataset
        self.mode = mode
        if not root_dir.exists():
            self.root_dir.mkdir(parents=True, exist_ok=True)

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str):
        self.rank = str(trainer.global_rank)
        self.targets_dir = self.root_dir / "targets"
        self.predictions_dir = self.root_dir / "predictions"
        self.embeddings_dir = self.root_dir / "embeddings"
        self.true_vf_dir = self.root_dir / "true_vf"
        self.targets_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.true_vf_dir.mkdir(parents=True, exist_ok=True)
        self.info_file = self.root_dir / f"info_{self.rank}.csv"
        if self.mode == "w" and self.info_file.exists():
            self.info_file.unlink()

    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        prediction: Prediction,
        batch_indices: Sequence[int] | None,
        batch: BatchWithSampleIndexes,
        batch_idx: int,
        dataloader_idx: int,
    ):
        pl_module = cast(MultisourceAbstractReconstructor, pl_module)
        # Retrieve sample indexes and batch data
        sample_indexes, raw_batch = batch
        # Extract the components from the prediction
        pred, avail = prediction.pred, prediction.avail

        # Generative predictions
        if isinstance(prediction, GenerativePrediction):
            has_multiple_realizations = True
            # Leading dims: R (realizations), T (diff steps), B (batch)
            leading_pred_dims = 3

        # Deterministic predictions
        else:
            # Deterministic models typically don't have multiple realizations
            has_multiple_realizations = False
            leading_pred_dims = 1  # batch dim only

        # We'll write to the info file in append mode
        for src, data in raw_batch.items():
            source_name = src.name
            src_index = src.index
            with torch.no_grad():
                # ================= DENORMALIZATION =================
                device = data.values.device
                _, targets = self.dataset.normalize(
                    data.values,
                    source_name,  # Use only source_name for normalization
                    denormalize=True,
                    leading_dims=1,
                    device=device,
                )
                if src in pred:
                    _, pred[src] = self.dataset.normalize(
                        pred[src],
                        source_name,  # Use only source_name for normalization
                        denormalize=True,
                        leading_dims=leading_pred_dims,
                        device=device,
                    )
                # Denormalize predicted means if they exist
                if (
                    isinstance(prediction, GenerativePrediction)
                    and prediction.pred_mean is not None
                    and src in prediction.pred_mean
                ):
                    _, prediction.pred_mean[src] = self.dataset.normalize(
                        prediction.pred_mean[src],
                        source_name,
                        denormalize=True,
                        leading_dims=1,  # batch dim only
                        device=device,
                    )

                # ================= TARGETS =================
                # Create directory structure with source_name/index to organize by both source and index
                target_dir = self.targets_dir / source_name / str(src_index)
                target_dir.mkdir(parents=True, exist_ok=True)
                targets = targets.detach().cpu().float().numpy()
                # Retrieve the lat/lon and dt
                latlon = data.coords.detach().cpu().float().numpy()
                dt_float = data.dt.detach().cpu().float().numpy()
                dt: np.ndarray = dt_float * (self.dt_max - self.dt_min) + self.dt_min

                # The dimensions of the Datasets we'll create depend on the
                # dimensionality of the source.
                if len(targets.shape) == 4:  # 2d sources -> (B, C, H, W)
                    dims = ["sample", "H", "W"]
                elif len(targets.shape) == 2:  # 0d sources -> (B, C)
                    dims = ["sample"]
                else:
                    raise ValueError(f"Unsupported number of dims: {targets.shape}")

                # Fetch the names of the variables in the source
                source_obj = pl_module.sources[source_name]
                input_var_names = [v for v in source_obj.data_vars]
                # Create xarray Dataset for targets. Each channel in the targets
                # is a variable in the Dataset.
                coords = {
                    "sample": sample_indexes,
                    "lat": (dims, latlon[:, 0]),
                    "lon": (dims, latlon[:, 1]),
                    "dt": (("sample"), dt),
                }
                targets_ds = xr.Dataset(
                    {var: (dims, targets[:, i]) for i, var in enumerate(input_var_names)},
                    coords=coords,
                )
                # Write each sample to a separate file. Ignore all samples for which the source
                # is unavailable (i.e., has an availability flag of -1).
                for k, sample_idx in enumerate(sample_indexes):
                    if avail[src][k].item() == -1:
                        continue
                    sample_target_ds = targets_ds.sel(sample=sample_idx)
                    atomic_to_netcdf(sample_target_ds, target_dir / f"{sample_idx}.nc")

                # ================= PREDICTIONS =================
                if src in pred:
                    # If no prediction was made for this source,
                    # skip it.
                    prediction_dir = self.predictions_dir / source_name / str(src_index)
                    prediction_dir.mkdir(parents=True, exist_ok=True)
                    # Only keep variables that are in the source's output_vars
                    predictions = pred[src].detach().cpu().float().numpy()
                    output_var_names = [v for v in source_obj.output_vars]

                    # Create xarray Dataset for predictions. Same principle as for targets.
                    # For generative predictions, create dims and coords for R and T
                    pred_dims = list(dims)
                    pred_coords = coords.copy()
                    if isinstance(prediction, GenerativePrediction):
                        pred_dims = ["realization", "integration_step"] + pred_dims
                        pred_coords["integration_step"] = prediction.time_grid.cpu().float().numpy()
                        pred_coords["realization"] = np.arange(predictions.shape[0])
                    predictions_ds = xr.Dataset(
                        {
                            var: (
                                pred_dims,
                                predictions[(slice(None),) * leading_pred_dims + (i, ...)],
                            )  # Realization, batch, channel
                            for i, var in enumerate(output_var_names)
                        },
                        coords=pred_coords,
                    )

                    # Write predicted means if available
                    if (
                        isinstance(prediction, GenerativePrediction)
                        and prediction.pred_mean is not None
                        and src in prediction.pred_mean
                    ):
                        pred_mean_outputs = prediction.pred_mean[src].detach().cpu().float().numpy()

                        # Create a dataset for the predicted means
                        pred_mean_ds = xr.Dataset(
                            {
                                f"pred_mean_{var}": (dims, pred_mean_outputs[:, i])
                                for i, var in enumerate(output_var_names)
                            },
                            coords=coords,
                        )

                        # Merge the datasets before writing to file
                        predictions_ds = xr.merge([predictions_ds, pred_mean_ds])

                    # Write each sample to a separate file
                    for k, sample_idx in enumerate(sample_indexes):
                        if avail[src][k].item() == -1:
                            continue
                        sample_prediction_ds = predictions_ds.sel(sample=sample_idx)
                        atomic_to_netcdf(sample_prediction_ds, prediction_dir / f"{sample_idx}.nc")

                # ================= INFO FILES =================
                batch_size = latlon.shape[0]
                # Convert the time deltas from fractions of dt_max to absolute durations
                # in hours.
                dt_series = pd.Series(dt, name="dt")

                # Add flags to indicate if the predictions are generative and if they
                # include predicted means.
                includes_intermediate_steps = isinstance(prediction, GenerativePrediction)
                has_multiple_realizations = isinstance(prediction, GenerativePrediction)
                has_pred_means = (
                    isinstance(prediction, GenerativePrediction)
                    and prediction.pred_mean is not None
                    and src in prediction.pred_mean
                )

                # Store the information about that source's data for that batch
                # in the info DataFrame.
                info_df = pd.DataFrame(
                    {
                        "sample_index": sample_indexes,
                        "source_name": [source_name] * batch_size,
                        "source_index": [src_index] * batch_size,  # Add observation index
                        "avail": avail[src].detach().cpu().float().numpy(),
                        "dt": dt_series,
                        "channels": [targets.shape[1]] * batch_size,
                        "spatial_shape": [targets.shape[2:]] * batch_size,
                        "has_multiple_realizations": [has_multiple_realizations] * batch_size,
                        "includes_intermediate_steps": [includes_intermediate_steps] * batch_size,
                        "has_pred_means": [has_pred_means] * batch_size,  # New column
                    },
                )
                # Remove the rows where the source is not available
                info_df = info_df[info_df["avail"] != -1]
                include_header = not self.info_file.exists()
            info_df.to_csv(self.info_file, mode="a", header=include_header, index=False)

            # The lightning Trainer object keeps the predictions in memory for
            # write_on_epoch_end. Since we'll never call that method, we can clear the predictions
            # to limit the memory usage to that of a single batch.
            trainer.predict_loop._predictions = [
                [] for _ in range(trainer.predict_loop.num_dataloaders)
            ]
            gc.collect()
