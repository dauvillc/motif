"""
Implements visual evaluation comparison for multi-source predictions.

This evaluation class creates visualization figures showing targets and predictions
for each sample. The figures are organized as follows:
- Top row: Available sources (avail=1) and target source s(avail=0)
- Subsequent rows: One row per model showing predictions for each source

Usage in Hydra config:
evaluation_classes:
  visual:
    _target_: 'motif.eval.visualization_new.VisualEvaluationComparison'
    eval_fraction: 1.0  # Fraction of samples to visualize
    max_realizations_to_display: 6  # Maximum realizations to show
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Generator, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import Normalize
from tqdm import tqdm

from motif.data.grid_functions import crop_nan_border_numpy
from motif.datatypes import SourceIndex
from motif.eval.abstract_evaluation_metric import AbstractMultisourceEvaluationMetric
from motif.eval.utils import format_tdelta


class VisualEvaluationComparison(AbstractMultisourceEvaluationMetric):
    """Evaluation class that creates visualization figures for targets and predictions."""

    def __init__(
        self,
        model_data: dict[str, dict],
        parent_results_dir: str | Path,
        eval_fraction: float = 1.0,
        max_realizations_to_display: int = 3,
        cmap: str = "viridis",
        num_workers: int = 10,
        **kwargs,
    ):
        """
        Args:
            model_data (dict): Dictionary mapping model_ids to model specifications
            parent_results_dir (str or Path): Parent directory for all results
            eval_fraction (float): Fraction of samples to visualize (0.0 to 1.0)
            max_realizations_to_display (int): Maximum number of realizations to display
            cmap (str): Colormap to use for visualization.
            num_workers (int): Number of workers for parallel processing.
            **kwargs: Additional keyword arguments for the base class.
        """
        super().__init__(
            id_name="visual",
            full_name="Visual Evaluation Comparison",
            model_data=model_data,
            parent_results_dir=parent_results_dir,
            **kwargs,
        )
        self.eval_fraction = eval_fraction
        self.max_realizations_to_display = max_realizations_to_display
        self.cmap = cmap
        self.num_workers = num_workers

    def evaluate(self, **kwargs):
        """
        Creates visualization figures for all samples across all models.

        Args:
            **kwargs: Additional keyword arguments
        """
        n_samples = int(self.eval_fraction * self.n_samples)
        # Decide on a random subset of samples to evaluate
        if self.eval_fraction < 1.0:
            rng = np.random.default_rng(seed=42)
            selected_indices = set(rng.choice(self.n_samples, size=n_samples, replace=False))

            def samples_iterator() -> Generator[
                tuple[
                    pd.DataFrame,
                    dict[SourceIndex, xr.Dataset],
                    dict[str, dict[SourceIndex, xr.Dataset]],
                ],
                None,
                None,
            ]:
                return (
                    (df, targets, preds)
                    for i, (df, targets, preds) in enumerate(self.samples_iterator())
                    if i in selected_indices
                )

        else:
            samples_iterator = self.samples_iterator

        # Collect all samples into a list for parallel processing
        samples = list(samples_iterator())

        if self.num_workers > 1:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit all tasks
                futures = [executor.submit(self._process_sample, *sample) for sample in samples]
                # Process results with progress bar
                for future in tqdm(
                    as_completed(futures), desc="Evaluating samples", total=len(futures)
                ):
                    future.result()  # This will raise any exceptions that occurred
        else:
            # Sequential processing
            for sample in tqdm(samples, desc="Evaluating samples", total=n_samples):
                self._process_sample(*sample)

    def _process_sample(
        self,
        sample_df: pd.DataFrame,
        targets: dict[SourceIndex, xr.Dataset],
        preds: dict[str, dict[SourceIndex, xr.Dataset]],
    ):
        """Process a single sample (helper method for parallel execution).

        Args:
            sample_df (pandas.DataFrame): DataFrame with sample metadata
            targets (dict): Dict mapping source indices to target xarray datasets
            preds (dict): Dict mapping model_ids to dicts of source indices to prediction datasets
        """
        sample_index = sample_df["sample_index"].iloc[0]
        # Retrieve the number of channels from the first target source. We'll assume
        # all sources have the same number of channels.
        channels = cast(list[str], list(next(iter(targets.values())).data_vars))

        for channel_idx in range(len(channels)):
            # Crop the padded borders
            cropped_data = self.crop_padded_borders(targets, preds, sample_df, channel_idx)

            # Plot the data
            self.plot_sample(
                sample_index,
                cropped_data,
                sample_df,
                channels[channel_idx],
            )

    def crop_padded_borders(
        self,
        targets: dict[SourceIndex, xr.Dataset],
        preds: dict[str, dict[SourceIndex, xr.Dataset]],
        sample_df: pd.DataFrame,
        channel_idx: int,
    ) -> tuple[
        dict[SourceIndex, np.ndarray],
        dict[SourceIndex, np.ndarray],
        dict[SourceIndex, np.ndarray],
        dict[SourceIndex, np.ndarray],
        dict[str, dict[SourceIndex, np.ndarray]],
    ]:
        """Crops the padded borders in the sample data.

        Args:
            targets (dict): Dict mapping source indices to target xarray datasets
            preds (dict): Dict mapping model_ids to dicts of source indices to prediction datasets
            sample_df (pandas.DataFrame): DataFrame with sample metadata, indexed
                        by (source_name, source_index).
            channel_idx (int): Index of the channel to process.
        Returns:
            available_sources (dict): Dict (src_name, src_index) -> available source data
            target_sources (dict): Dict (src_name, src_index) -> target source data
            lats (dict): Dict (src_name, src_index) -> latitude coordinates
            lons (dict): Dict (src_name, src_index) -> longitude coordinates
            predictions (dict): Dict model_id -> (src_name, src_index) -> prediction data
        """

        # Because of the batching process in the prediction pipeline, the data generally
        # includes large padded borders, that we want to crop. In the targets / available sources,
        # and coordinates, these are padded with NaNs. In the predictions,
        # they may contain anything.
        # We will crop them using crop_nan_border_numpy, using the targets as a reference. To do
        # so, we'll here retrieve the targets / available sources, coordinates and predictions
        # for all models, and then crop them all at once.
        available_sources = {}  # (src_name, src_index) -> available / source data
        target_sources = {}  # (src_name, src_index) -> target data
        lats, lons = {}, {}  # (src_name, src_index) -> coordinates data
        predictions = {model_id: {} for model_id in self.model_data}
        for src, target_data in targets.items():
            # Get the availability flag (-1: missing, 0: target, 1: available)
            avail = sample_df.loc[(src.name, src.index), "avail"]
            if avail == -1:
                continue
            # Get the coordinates data: latitude and longitude (will be used for the ticklabels)
            src_lat = target_data["lat"].values
            src_lon = target_data["lon"].values
            # We'll use the target as reference for cropping
            channel = list(target_data.data_vars)[channel_idx]
            target_arr = target_data[channel].values
            # Gather all models' predictions for that source if there are any.
            preds_list = []
            for model_id in self.model_data:
                if src in preds[model_id]:
                    preds_list.append(preds[model_id][src][channel].values)
            # Crop all borders all at once using the target as reference
            out = crop_nan_border_numpy(target_arr, [target_arr, src_lat, src_lon] + preds_list)
            lats[src] = out[1]
            lons[src] = out[2]
            # Store the channel data either as target or available source
            if avail == 0:
                target_sources[src] = out[0]
            else:
                available_sources[src] = out[0]
            # Store the cropped predictions
            for k, model_id in enumerate(self.model_data):
                if src in preds[model_id]:
                    predictions[model_id][src] = out[k + 3]

        return available_sources, target_sources, lats, lons, predictions

    def plot_sample(
        self,
        sample_index: int,
        cropped_data: tuple[
            dict[SourceIndex, np.ndarray],
            dict[SourceIndex, np.ndarray],
            dict[SourceIndex, np.ndarray],
            dict[SourceIndex, np.ndarray],
            dict[str, dict[SourceIndex, np.ndarray]],
        ],
        sample_df: pd.DataFrame,
        channel: str,
    ):
        """Plots the targets and predictions for a single sample.

        Args:
            sample_index (int): Index of the sample
            cropped_data (tuple): Tuple containing available sources, target sources,
                latitudes, longitudes, and predictions.
            sample_df (pandas.DataFrame): DataFrame with sample metadata
            channel (str): Name of the channel, which will be used in the plot titles.
        """
        available_sources, target_sources, lats, lons, predictions = cropped_data

        # Retrieve the timedelta of each available source, and sort them in ascending order.
        avail_dts = {
            src: cast(pd.Timedelta, sample_df.loc[(src.name, src.index), "dt"])
            for src in available_sources.keys()
        }
        available_sources = dict(
            sorted(available_sources.items(), key=lambda item: avail_dts[item[0]])
        )

        # Create a figure with subplots: one row per model + 1 for the targets / available sources,
        # and one column per source (up to max_realizations_to_display).
        num_sources, num_models = len(sample_df), len(self.model_data)
        n_cols = max(num_sources, self.max_realizations_to_display)
        fig, axes = plt.subplots(
            nrows=num_models + 1,
            ncols=n_cols,
            figsize=(3 * n_cols, 3 * (num_models + 1)),
            squeeze=False,
        )

        # ------- FIRST ROW: Targets, then available sources -------
        col_cnt = 0
        # Map {(src_name, src_index) --> mpl Normalize}
        norms = {}
        # First, targets
        for src in target_sources.keys():
            display_name = self._display_src_name(src.name)
            channel_data = target_sources[src]
            # Extract the min and max values to create a colormap
            vmin, vmax = np.nanmin(channel_data), np.nanmax(channel_data)
            norm = Normalize(vmin=vmin, vmax=vmax)
            norms[src] = norm

            ax = axes[0, col_cnt]
            ax.imshow(channel_data, aspect="auto", cmap=self.cmap, norm=norm)
            dt = cast(pd.Timedelta, sample_df.loc[(src.name, src.index), "dt"])
            dt_str = format_tdelta(dt)
            ax.set_title(f"Target: {display_name} $\\delta t=${dt_str}")
            self._set_coords_as_ticks(ax, lats[src], lons[src])
            col_cnt += 1

        # Available sources
        for src in available_sources.keys():
            display_name = self._display_src_name(src.name)
            channel_data = available_sources[src]
            ax = axes[0, col_cnt]
            ax.imshow(channel_data, aspect="auto", cmap=self.cmap)
            dt_str = format_tdelta(avail_dts[src])
            ax.set_title(f"{display_name} $\\delta t=${dt_str}")
            self._set_coords_as_ticks(ax, lats[src], lons[src])
            col_cnt += 1

        # Hide the axes that are not used
        for j in range(col_cnt, n_cols):
            axes[0, j].axis("off")

        # ------- SUBSEQUENT ROWS: Model predictions / realizations -----
        for k, model_id in enumerate(self.model_data):
            col_cnt = 0
            for src in target_sources.keys():
                src_name = src.name
                display_name = self._display_src_name(src_name)
                # If the prediction contains one more dim than the target, we
                # assume the first dim is the realization index. In this case,
                # we'll plot one realization per column, up to the maximum number
                # of columns. If there are the same number of dims, there's only
                # a single prediction to plot.
                pred_data = predictions[model_id][src]
                if len(pred_data.shape) == len(target_sources[src].shape) + 1:
                    # Multiple realizations, plot only the first ones
                    for realization_idx in range(
                        min(self.max_realizations_to_display, pred_data.shape[0])
                    ):
                        ax = axes[k + 1, col_cnt]
                        pred_realization = pred_data[realization_idx, ...]
                        ax.imshow(
                            pred_realization,
                            aspect="auto",
                            cmap=self.cmap,
                            norm=norms[src],
                        )
                        ax.set_title(f"{display_name} - {model_id}")
                        self._set_coords_as_ticks(ax, lats[src], lons[src])
                        col_cnt += 1
                else:
                    # Single prediction, plot it directly
                    ax = axes[k + 1, col_cnt]
                    ax.imshow(
                        pred_data,
                        aspect="auto",
                        cmap=self.cmap,
                        norm=norms[src],
                    )
                    ax.set_title(f"{display_name} - {model_id}")
                    self._set_coords_as_ticks(ax, lats[src], lons[src])
                    col_cnt += 1
            # Hide the axes that are not used
            for j in range(col_cnt, n_cols):
                axes[k + 1, j].axis("off")

        plt.suptitle(channel, fontsize=16)
        plt.tight_layout()
        # Save the figure
        fig_path = self.metric_results_dir / f"sample_{sample_index}_{channel}.svg"
        fig.savefig(fig_path, format="svg")

        plt.close(fig)

    @staticmethod
    def _set_coords_as_ticks(ax: Any, lats: np.ndarray, lons: np.ndarray, every_nth_pixel=30):
        """Sets the latitude and longitude coordinates as ticks on the axes."""
        ax.set_xticks(range(0, len(lons[0]), every_nth_pixel))
        ax.set_xticklabels([f"{lon:.2f}" for lon in lons[0, ::every_nth_pixel]], rotation=45)
        ax.set_yticks(range(0, len(lats[:, 0]), every_nth_pixel))
        ax.set_yticklabels([f"{lat:.2f}" for lat in lats[::every_nth_pixel, 0]])
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
