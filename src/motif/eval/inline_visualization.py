"""
Implements inline visual evaluation with a two-row, grouped-column layout.

Each sample produces one figure split into two horizontal rows:

  Row 1 – Observations
    [ Groundtruth observation ] | [ Input sources ]

  Row 2 – Predictions
    [ Model A ] | [ Model B ] | ...

- "Input sources": one sub-column per available source (avail=1), sorted by dt.
- "Groundtruth observation": the single target source (avail=0).
- One group per model: at most max_realizations_to_display sub-columns for the target.

The figure width is determined by the wider of the two rows, keeping each row
compact enough to fit on an A4 page.  Lat/lon tick labels are shown on input
sources and the groundtruth (observation row only); prediction sub-plots have
their tick labels hidden.  Group headers are rendered in a thin label row above
each image row via nested GridSpec/GridSpecFromSubplotSpec so that the separation
between groups is precise and independent of subplot spacing parameters.

Usage in Hydra config::

    evaluation_classes:
      inline_visual:
        _target_: motif.eval.inline_visualization.InlineVisualEval
        eval_fraction: 1.0
        max_realizations_to_display: 3
"""

import textwrap
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Generator, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from tqdm import tqdm

from motif.datatypes import SourceIndex
from motif.eval.abstract_evaluation_metric import AbstractMultisourceEvaluationMetric
from motif.eval.plot_style import apply_paper_style
from motif.eval.utils import format_tdelta


class InlineVisualEval(AbstractMultisourceEvaluationMetric):
    """Evaluation class that creates two-row, grouped visualization figures.

    Row 1 (observations): "Groundtruth observation" and "Input sources" groups,
    with lat/lon tick labels.  Row 2 (predictions): one group per model.
    Each group is visually separated by an inter-group gap and carries a bold header
    label above the images.  Figure width is the wider of the two rows.
    """

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
            model_data: Dictionary mapping model_ids to model specifications.
            parent_results_dir: Parent directory for all results.
            eval_fraction: Fraction of samples to visualize (0.0 to 1.0).
            max_realizations_to_display: Maximum number of realizations to show per model.
            cmap: Matplotlib colormap name.
            num_workers: Number of parallel workers.
            **kwargs: Additional keyword arguments forwarded to the base class.
        """
        super().__init__(
            id_name="inline_visual",
            full_name="Inline Visual Evaluation",
            model_data=model_data,
            parent_results_dir=parent_results_dir,
            **kwargs,
        )
        self.eval_fraction = eval_fraction
        self.max_realizations_to_display = max_realizations_to_display
        self.cmap = cmap
        self.num_workers = num_workers

    def evaluate(self, **kwargs):
        """Creates inline visualization figures for all (or a random subset of) samples."""
        n_samples = int(self.eval_fraction * self.n_samples)
        if self.eval_fraction < 1.0:
            rng = np.random.default_rng(seed=42)
            selected_indices = set(rng.choice(self.n_samples, size=n_samples, replace=False))

            def samples_iterator() -> Generator[
                tuple[
                    pd.DataFrame,
                    dict[str, dict[SourceIndex, xr.Dataset]],
                    dict[str, dict[SourceIndex, xr.Dataset]],
                ],
                None,
                None,
            ]:
                return (
                    (df, true_obs, preds)
                    for i, (df, true_obs, preds) in enumerate(self.samples_iterator())
                    if i in selected_indices
                )

        else:
            samples_iterator = self.samples_iterator

        samples = list(samples_iterator())

        if self.num_workers > 1:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [executor.submit(self._process_sample, *sample) for sample in samples]
                for future in tqdm(
                    as_completed(futures), desc="Evaluating samples", total=len(futures)
                ):
                    future.result()
        else:
            for sample in tqdm(samples, desc="Evaluating samples", total=n_samples):
                self._process_sample(*sample)

    def _process_sample(
        self,
        sample_df: pd.DataFrame,
        true_obs: dict[str, dict[SourceIndex, xr.Dataset]],
        preds: dict[str, dict[SourceIndex, xr.Dataset]],
    ):
        """Plot_sample for each channel in this sample."""
        sample_index = sample_df["sample_index"].iloc[0]
        # There is exactly one target source (avail=0) per sample; use it to determine channels.
        # We can assume the target is the same across all models, so we can look for it in any
        # model's true observations.
        any_model = next(iter(true_obs.keys()))
        any_true_obs = true_obs[any_model]
        target_src = next(
            src
            for src in any_true_obs
            if sample_df.loc[(any_model, src.name, src.index), "avail"] == 0
        )
        channels = cast(list[str], list(any_true_obs[target_src].data_vars))

        # Create one figure per channel in the target source.
        for channel in channels:
            cropped_data = self._extract_numpy_arrays(true_obs, preds, sample_df, channel)
            self.plot_sample(sample_index, cropped_data, sample_df, channel)

    def _extract_numpy_arrays(
        self,
        true_obs: dict[str, dict[SourceIndex, xr.Dataset]],
        preds: dict[str, dict[SourceIndex, xr.Dataset]],
        sample_df: pd.DataFrame,
        channel: str,
    ) -> tuple[
        dict[str, dict[SourceIndex, np.ndarray]],
        dict[SourceIndex, np.ndarray],
        dict[SourceIndex, np.ndarray],
        dict[SourceIndex, np.ndarray],
        dict[str, dict[SourceIndex, np.ndarray]],
    ]:
        """Extracts numpy arrays from the sample data for plotting.

        Returns:
            available_sources: avail==1 data arrays keyed by SourceIndex.
            target_sources:    avail==0 data arrays keyed by SourceIndex.
            lats:              latitude grids keyed by SourceIndex.
            lons:              longitude grids keyed by SourceIndex.
            predictions:       model_id -> SourceIndex -> prediction array.
        """
        target_sources: dict[SourceIndex, np.ndarray] = {}
        lats: dict[SourceIndex, np.ndarray] = {}
        lons: dict[SourceIndex, np.ndarray] = {}
        available_sources: dict[str, dict[SourceIndex, np.ndarray]] = {
            model_id: {} for model_id in self.model_data
        }
        predictions: dict[str, dict[SourceIndex, np.ndarray]] = {
            model_id: {} for model_id in self.model_data
        }

        for model_id, model_true_obs in true_obs.items():
            for src, true_obs_data in model_true_obs.items():
                avail = sample_df.loc[(model_id, src.name, src.index), "avail"]
                if avail == -1:
                    continue
                src_lat = true_obs_data["lat"].values
                src_lon = true_obs_data["lon"].values
                # For target sources, we use the indicated channel.
                # For input sources, if that same channel exists we use it, otherwise
                # we default to the first channel.
                channels = list(true_obs_data.data_vars)
                if channel in channels:
                    true_obs_arr = true_obs_data[channel].values
                else:
                    true_obs_arr = true_obs_data[channels[0]].values

                lats[src] = src_lat
                lons[src] = src_lon

                if avail == 0:
                    target_sources[src] = true_obs_arr
                else:
                    available_sources[model_id][src] = true_obs_arr

                if src in preds[model_id]:
                    predictions[model_id][src] = preds[model_id][src][channel].values

        return available_sources, target_sources, lats, lons, predictions

    def plot_sample(
        self,
        sample_index: int,
        data: tuple[
            dict[str, dict[SourceIndex, np.ndarray]],
            dict[SourceIndex, np.ndarray],
            dict[SourceIndex, np.ndarray],
            dict[SourceIndex, np.ndarray],
            dict[str, dict[SourceIndex, np.ndarray]],
        ],
        sample_df: pd.DataFrame,
        channel: str,
    ):
        """Render one figure with observations on the top row and predictions below.

        Layout::

            Row 1 – Observations
            +--------------------+--+-----------------+
            | Groundtruth        |  | Input sources   |  ← colorbar
            +--------------------+  +-----------------+
            | tgt                |  | avS0 | avS1 |…  |
            +--------------------+  +------+------+---+

            Row 2 – Predictions
            +------------------+--+------------------+--+-...
            | <model_id>       |  | <model_id>       |  | ...
            +------------------+  +------------------+  +
            | r0 | r1 | r2 | … |  | r0 | r1 | r2 | … |  |
            +----+----+----+---+  +----+----+----+---+  +

        Args:
            sample_index: Integer index of the sample (used in the output filename).
            data: Extracted data arrays for the sample, returned by _extract_numpy_arrays.
            sample_df: Per-sample metadata DataFrame.
            channel: Channel name for the figure title and colorbar label.
        """
        all_avail_sources, target_sources, lats, lons, predictions = data

        apply_paper_style()

        # The available sources might differ between the models. We'll take the union
        # of all available sources across models and remove duplicated (source_name, dt) entries.
        available_sources: dict[tuple[SourceIndex, pd.Timedelta], np.ndarray] = {}
        for model_id, model_avail in all_avail_sources.items():
            for src, arr in model_avail.items():
                dt = cast(pd.Timedelta, sample_df.loc[(model_id, src.name, src.index), "dt"])
                key = (src, dt)
                if key not in available_sources:
                    available_sources[key] = arr
        # Sort available sources by dt
        available_sources = dict(sorted(available_sources.items(), key=lambda item: item[0][1]))

        # There is exactly one target source per sample
        target_src = next(iter(target_sources))
        model_ids = list(self.model_data.keys())
        n_avail = len(available_sources)

        # Number of realization columns per model: inspect shape of first prediction
        n_real_per_model: dict[str, int] = {}
        for model_id in model_ids:
            n_real = 1
            if target_src in predictions[model_id]:
                pred = predictions[model_id][target_src]
                tgt_ndim = len(target_sources[target_src].shape)
                if len(pred.shape) == tgt_ndim + 1:
                    n_real = min(self.max_realizations_to_display, pred.shape[0])
            n_real_per_model[model_id] = n_real

        # Normalization shared across all subplots (target, inputs, predictions)
        all_arrays = (
            [target_sources[target_src]]
            + list(available_sources.values())
            + [
                pred_data[r, ...]
                if len(pred_data.shape) == len(target_sources[target_src].shape) + 1
                else pred_data
                for model_id in model_ids
                if target_src in predictions[model_id]
                for r, pred_data in [
                    (r, predictions[model_id][target_src])
                    for r in range(n_real_per_model[model_id])
                ]
            ]
        )
        global_min = float(np.nanmin([np.nanmin(a) for a in all_arrays]))
        global_max = float(np.nanmax([np.nanmax(a) for a in all_arrays]))
        norm = Normalize(vmin=global_min, vmax=global_max)

        has_input_group = n_avail > 0
        gt_group_idx = 0
        model_group_start = 1 + (1 if has_input_group else 0)

        # ---- Group definitions (observation row) ----
        obs_group_labels: list[str] = ["Groundtruth observation"]
        obs_group_col_counts: list[int] = [1]
        if has_input_group:
            obs_group_labels.append("Input sources")
            obs_group_col_counts.append(n_avail)
        n_obs_groups = len(obs_group_labels)

        # ---- Group definitions (prediction row) ----
        pred_group_labels = [textwrap.fill(m, width=20) for m in model_ids]
        pred_group_col_counts = [n_real_per_model[m] for m in model_ids]
        n_pred_groups = len(pred_group_labels)

        # ---- Figure sizing ----
        # Each image sub-column occupies cell_size inches; inter-group gaps add
        # ~SPACER * cell_size of whitespace each; extra width is reserved for colorbar.
        cell_width = 1.5  # inches per image column
        cell_height = 1.8  # inches per image row (independent of column width)
        SPACER = 0.12  # spacer fraction of cell_width between groups
        LABEL_H_RATIO = 0.08  # label row height relative to image row
        obs_effective = sum(obs_group_col_counts) + SPACER * (n_obs_groups - 1)
        pred_effective = sum(pred_group_col_counts) + SPACER * (n_pred_groups - 1)
        fig_width = cell_width * max(obs_effective, pred_effective) + 0.4
        fig_height = cell_height * 2 * (1.0 + LABEL_H_RATIO) + 0.2  # two image rows

        fig = plt.figure(figsize=(fig_width, fig_height))

        # Outer GridSpec: two rows (obs, pred), each containing its own label+image sub-grid.
        outer_gs = GridSpec(nrows=2, ncols=1, figure=fig, hspace=0.65)

        obs_gs = GridSpecFromSubplotSpec(
            nrows=2,
            ncols=n_obs_groups,
            subplot_spec=outer_gs[0],
            height_ratios=[LABEL_H_RATIO, 1.0],
            hspace=0.65,
            wspace=0.15,
            width_ratios=obs_group_col_counts,
        )
        pred_gs = GridSpecFromSubplotSpec(
            nrows=2,
            ncols=n_pred_groups,
            subplot_spec=outer_gs[1],
            height_ratios=[LABEL_H_RATIO, 1.0],
            hspace=0.65,
            wspace=0.15,
            width_ratios=pred_group_col_counts,
        )

        # Allocate all axes up front so we can reference them by (group, col) later
        image_axes: dict[tuple[int, int], plt.Axes] = {}  # type: ignore[type-arg]

        # --- Observation-row axes ---
        for g_local in range(n_obs_groups):
            g_global = g_local
            label_ax = fig.add_subplot(obs_gs[0, g_local])
            label_ax.axis("off")
            label_ax.set_facecolor("#e8e8e8")
            label_ax.patch.set_visible(True)
            label_ax.text(
                0.5,
                0.5,
                obs_group_labels[g_local],
                ha="center",
                va="center",
                transform=label_ax.transAxes,
                fontsize=10,
                fontweight="bold",
            )

            col_count = obs_group_col_counts[g_local]
            is_multi_input_group = has_input_group and g_local == 1 and col_count > 1
            inner_wspace = 0.20 if is_multi_input_group else 0.05
            inner_gs = GridSpecFromSubplotSpec(
                nrows=1,
                ncols=col_count,
                subplot_spec=obs_gs[1, g_local],
                wspace=inner_wspace,
            )
            for c in range(col_count):
                image_axes[(g_global, c)] = fig.add_subplot(inner_gs[0, c])

        # --- Prediction-row axes ---
        for g_local, model_id in enumerate(model_ids):
            g_global = model_group_start + g_local
            label_ax = fig.add_subplot(pred_gs[0, g_local])
            label_ax.axis("off")
            label_ax.set_facecolor("#e8e8e8")
            label_ax.patch.set_visible(True)
            label_ax.text(
                0.5,
                0.5,
                pred_group_labels[g_local],
                ha="center",
                va="center",
                transform=label_ax.transAxes,
                fontsize=10,
                fontweight="bold",
            )

            col_count = pred_group_col_counts[g_local]
            inner_gs = GridSpecFromSubplotSpec(
                nrows=1,
                ncols=col_count,
                subplot_spec=pred_gs[1, g_local],
                wspace=0.05,
            )
            for c in range(col_count):
                image_axes[(g_global, c)] = fig.add_subplot(inner_gs[0, c])

        # ---- Plot: Input sources ----
        if has_input_group:
            for c, (src, dt) in enumerate(available_sources.keys()):
                ax = image_axes[(1, c)]
                ax.imshow(available_sources[(src, dt)], aspect="equal", cmap=self.cmap, norm=norm)
                dt_str = format_tdelta(dt)
                display = self._display_src_name(src.name)
                ax.set_title(f"{display}\n$\\delta t$ = {dt_str}", fontsize=7)
                self._set_coords_as_ticks(ax, lats[src], lons[src], write_labels=False)

        # ---- Plot: Groundtruth observation (single target source) ----
        gt_ax = image_axes[(gt_group_idx, 0)]
        gt_ax.imshow(target_sources[target_src], aspect="equal", cmap=self.cmap, norm=norm)
        # Retrieve the target's dt from any model's info df, as it's the same across models anyway.
        any_model = next(iter(model_ids))
        dt = cast(pd.Timedelta, sample_df.loc[(any_model, target_src.name, target_src.index), "dt"])
        gt_ax.set_title(
            f"{self._display_src_name(target_src.name)}\n$\\delta t$ = {format_tdelta(dt)}",
            fontsize=7,
        )
        self._set_coords_as_ticks(gt_ax, lats[target_src], lons[target_src], write_labels=True)

        # ---- Plot: Model predictions ----
        for k, model_id in enumerate(model_ids):
            g = model_group_start + k
            n_real = n_real_per_model[model_id]

            if target_src not in predictions[model_id]:
                for r in range(n_real):
                    image_axes[(g, r)].axis("off")
                continue

            pred_data = predictions[model_id][target_src]
            tgt_ndim = len(target_sources[target_src].shape)
            is_multi_real = len(pred_data.shape) == tgt_ndim + 1
            display = self._display_src_name(target_src.name)

            for r in range(n_real):
                ax = image_axes[(g, r)]
                realization = pred_data[r, ...] if is_multi_real else pred_data
                ax.imshow(realization, aspect="equal", cmap=self.cmap, norm=norm)
                title = display if n_real == 1 else f"{display} ({r + 1})"
                ax.set_title(title, fontsize=7)
                ax.set_xticks([])
                ax.set_yticks([])
                if not is_multi_real:
                    break  # deterministic: one column used, loop ends

        # ---- Colorbar (one, for the single target source) ----
        # The rect parameter reserves right margin for the colorbar.
        cbar_w = 0.018
        cbar_pad = 0.004
        right_margin = cbar_pad + cbar_w

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            # matplotlib warns that tight layout might not work, but it actually does.
            fig.tight_layout(rect=(0, 0, 1.0 - right_margin, 1.0))

        pos = gt_ax.get_position()
        cbar_left = 1.0 - right_margin + cbar_pad
        cax = fig.add_axes((cbar_left, pos.y0, cbar_w, pos.height))
        mappable = ScalarMappable(norm=norm, cmap=self.cmap)
        mappable.set_array([])
        cbar = fig.colorbar(mappable, cax=cax, label="Temperature (K)")
        cbar.ax.tick_params(labelsize=7)

        # ---- Save ----
        fig_stem = self.metric_results_dir / f"sample_{sample_index}_{channel}"
        fig.savefig(str(fig_stem) + ".svg", format="svg")
        fig.savefig(str(fig_stem) + ".pdf")
        plt.close(fig)

    @staticmethod
    def _set_coords_as_ticks(
        ax: Any, lats: np.ndarray, lons: np.ndarray, n_ticks: int = 5, write_labels: bool = True
    ):
        """Set lat/lon coordinates as axis tick labels."""
        x_indices = np.linspace(0, len(lons[0]) - 1, n_ticks, dtype=int)
        y_indices = np.linspace(0, len(lats[:, 0]) - 1, n_ticks, dtype=int)
        ax.set_xticks(x_indices)
        ax.set_xticklabels([f"{lons[0, i]:.2f}" for i in x_indices], rotation=45, fontsize=6)
        ax.set_yticks(y_indices)
        ax.set_yticklabels([f"{lats[i, 0]:.2f}" for i in y_indices], fontsize=6)
        if write_labels:
            ax.set_xlabel("Longitude", fontsize=7)
            ax.set_ylabel("Latitude", fontsize=7)
