"""
Implements inline visual evaluation with a single-row, grouped-column layout.

Each sample produces one figure where all data appear in a single horizontal row,
organized in clearly separated groups:

  [ Input sources ] | [ Groundtruth observation ] | [ Model A ] | [ Model B ] | ...

- "Input sources": one sub-column per available source (avail=1), sorted by dt.
- "Groundtruth observation": one sub-column per target source (avail=0).
- One group per model: at most max_realizations_to_display sub-columns per target.

Lat/lon tick labels are shown only for input sources and the groundtruth; prediction
sub-plots have their tick labels hidden.  Group headers are rendered in a thin label
row above the image row via nested GridSpec/GridSpecFromSubplotSpec so that the
separation between groups is precise and independent of subplot spacing parameters.

Usage in Hydra config::

    evaluation_classes:
      inline_visual:
        _target_: motif.eval.inline_visualization.InlineVisualEval
        eval_fraction: 1.0
        max_realizations_to_display: 3
"""

import textwrap
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

from motif.data.grid_functions import crop_nan_border_numpy
from motif.datatypes import SourceIndex
from motif.eval.abstract_evaluation_metric import AbstractMultisourceEvaluationMetric
from motif.eval.plot_style import GRID_CELL_SIZE, apply_paper_style
from motif.eval.utils import format_tdelta


class InlineVisualEval(AbstractMultisourceEvaluationMetric):
    """Evaluation class that creates single-row, grouped visualization figures.

    All subplots for one sample appear in a single horizontal row.  Subplot groups
    ("Input sources", "Groundtruth observation", one per model) are visually separated
    by a wide inter-group gap and each carries a bold header label above the images.
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
        targets: dict[SourceIndex, xr.Dataset],
        preds: dict[str, dict[SourceIndex, xr.Dataset]],
    ):
        """Crop borders and call plot_sample for each channel in this sample."""
        sample_index = sample_df["sample_index"].iloc[0]
        channels = cast(list[str], list(next(iter(targets.values())).data_vars))

        for channel_idx in range(len(channels)):
            cropped_data = self.crop_padded_borders(targets, preds, sample_df, channel_idx)
            self.plot_sample(sample_index, cropped_data, sample_df, channels[channel_idx])

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
        """Crop NaN-padded borders from targets, coordinates, and predictions.

        Returns:
            available_sources: avail==1 data arrays keyed by SourceIndex.
            target_sources:    avail==0 data arrays keyed by SourceIndex.
            lats:              latitude grids keyed by SourceIndex.
            lons:              longitude grids keyed by SourceIndex.
            predictions:       model_id -> SourceIndex -> prediction array.
        """
        available_sources: dict[SourceIndex, np.ndarray] = {}
        target_sources: dict[SourceIndex, np.ndarray] = {}
        lats: dict[SourceIndex, np.ndarray] = {}
        lons: dict[SourceIndex, np.ndarray] = {}
        predictions: dict[str, dict[SourceIndex, np.ndarray]] = {
            model_id: {} for model_id in self.model_data
        }

        for src, target_data in targets.items():
            avail = sample_df.loc[(src.name, src.index), "avail"]
            if avail == -1:
                continue
            src_lat = target_data["lat"].values
            src_lon = target_data["lon"].values
            channel = list(target_data.data_vars)[channel_idx]
            target_arr = target_data[channel].values

            preds_list = []
            for model_id in self.model_data:
                if src in preds[model_id]:
                    preds_list.append(preds[model_id][src][channel].values)

            out = crop_nan_border_numpy(target_arr, [target_arr, src_lat, src_lon] + preds_list)
            lats[src] = out[1]
            lons[src] = out[2]

            if avail == 0:
                target_sources[src] = out[0]
            else:
                available_sources[src] = out[0]

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
        """Render one figure with all subplots for this sample in a single row.

        Layout::

            +-----------------+--+--------------------+--+------------------+--+-...
            | Input sources   |  | Groundtruth        |  | <model_id>       |  | ...
            +-----------------+  +--------------------+  +------------------+  +
            | avS0 | avS1 |…  |  | tgt0 | tgt1 | …   |  | r0 | r1 | r2 | … |  |
            +------+------+---+  +------+------+----  +  +----+----+----+---+  +

        Args:
            sample_index: Integer index of the sample (used in the output filename).
            cropped_data: Tuple produced by crop_padded_borders.
            sample_df: Per-sample metadata DataFrame.
            channel: Channel name for the figure title and colorbar label.
        """
        available_sources, target_sources, lats, lons, predictions = cropped_data

        apply_paper_style()

        # Sort available sources by ascending dt
        avail_dts = {
            src: cast(pd.Timedelta, sample_df.loc[(src.name, src.index), "dt"])
            for src in available_sources.keys()
        }
        available_sources = dict(
            sorted(available_sources.items(), key=lambda item: avail_dts[item[0]])
        )

        target_srcs = list(target_sources.keys())
        model_ids = list(self.model_data.keys())
        n_avail = len(available_sources)
        n_tgt = len(target_sources)

        # Per-target normalization (shared by predictions of the same target)
        norms: dict[SourceIndex, Normalize] = {
            src: Normalize(vmin=np.nanmin(data), vmax=np.nanmax(data))
            for src, data in target_sources.items()
        }

        # Number of realization columns per model: inspect shape of first prediction
        n_real_per_model: dict[str, int] = {}
        for model_id in model_ids:
            n_real = 1
            if target_srcs:
                first_tgt = target_srcs[0]
                if first_tgt in predictions[model_id]:
                    pred = predictions[model_id][first_tgt]
                    tgt_ndim = len(target_sources[first_tgt].shape)
                    if len(pred.shape) == tgt_ndim + 1:
                        n_real = min(self.max_realizations_to_display, pred.shape[0])
            n_real_per_model[model_id] = n_real

        # Build group list (skip Input group when there are no available sources)
        group_labels: list[str] = []
        group_col_counts: list[int] = []

        has_input_group = n_avail > 0
        if has_input_group:
            group_labels.append("Input sources")
            group_col_counts.append(n_avail)

        group_labels.append("Groundtruth observation")
        group_col_counts.append(n_tgt)

        for model_id in model_ids:
            group_labels.append(textwrap.fill(model_id, width=20))
            group_col_counts.append(n_tgt * n_real_per_model[model_id])

        n_groups = len(group_labels)

        # ---- Figure sizing ----
        # Each image sub-column occupies GRID_CELL_SIZE inches; inter-group gaps add
        # ~0.18 * GRID_CELL_SIZE of whitespace each; extra width is reserved for colorbars.
        SPACER = 0.18  # spacer fraction of GRID_CELL_SIZE between groups
        total_image_cols = sum(group_col_counts)
        effective_cols = total_image_cols + SPACER * (n_groups - 1)
        LABEL_H_RATIO = 0.12  # label row height relative to image row
        fig_width = GRID_CELL_SIZE * effective_cols + 0.7  # 0.7 in reserved for colorbars
        fig_height = GRID_CELL_SIZE * (1.0 + LABEL_H_RATIO) + 0.45  # 0.45 in for suptitle

        fig = plt.figure(figsize=(fig_width, fig_height))

        # Outer GridSpec: 2 rows (label + image), one column per group.
        # wspace=0.55 creates the visible separation between groups.
        outer_gs = GridSpec(
            nrows=2,
            ncols=n_groups,
            figure=fig,
            height_ratios=[LABEL_H_RATIO, 1.0],
            width_ratios=group_col_counts,
            hspace=0.30,
            wspace=0.25,
        )

        # Allocate all axes up front so we can reference them by (group, col) later
        image_axes: dict[tuple[int, int], plt.Axes] = {}  # type: ignore[type-arg]

        for g in range(n_groups):
            # --- Label axis (top row) ---
            label_ax = fig.add_subplot(outer_gs[0, g])
            # axis("off") hides axis lines/ticks AND sets patch invisible; re-enable patch
            label_ax.axis("off")
            label_ax.set_facecolor("#e8e8e8")
            label_ax.patch.set_visible(True)
            label_ax.text(
                0.5,
                0.5,
                group_labels[g],
                ha="center",
                va="center",
                transform=label_ax.transAxes,
                fontsize=8,
                fontweight="bold",
            )

            # --- Image axes (bottom row) via inner GridSpec ---
            inner_gs = GridSpecFromSubplotSpec(
                nrows=1,
                ncols=group_col_counts[g],
                subplot_spec=outer_gs[1, g],
                wspace=0.05,
            )
            for c in range(group_col_counts[g]):
                image_axes[(g, c)] = fig.add_subplot(inner_gs[0, c])

        # Group index shortcuts
        gt_group_idx = 1 if has_input_group else 0
        model_group_start = gt_group_idx + 1

        # ---- Plot: Input sources ----
        if has_input_group:
            for c, src in enumerate(available_sources.keys()):
                ax = image_axes[(0, c)]
                ax.imshow(available_sources[src], aspect="auto", cmap=self.cmap)
                dt_str = format_tdelta(avail_dts[src])
                display = self._display_src_name(src.name)
                ax.set_title(f"{display}\n$\\delta t$ = {dt_str}", fontsize=7)
                self._set_coords_as_ticks(ax, lats[src], lons[src])

        # ---- Plot: Groundtruth observation ----
        gt_axes_per_src: dict[SourceIndex, plt.Axes] = {}  # type: ignore[type-arg]
        for c, src in enumerate(target_srcs):
            ax = image_axes[(gt_group_idx, c)]
            ax.imshow(target_sources[src], aspect="auto", cmap=self.cmap, norm=norms[src])
            dt = cast(pd.Timedelta, sample_df.loc[(src.name, src.index), "dt"])
            dt_str = format_tdelta(dt)
            display = self._display_src_name(src.name)
            ax.set_title(f"{display}\n$\\delta t$ = {dt_str}", fontsize=7)
            self._set_coords_as_ticks(ax, lats[src], lons[src])
            ax.set_ylabel("")  # lat label is shown on input sources only
            gt_axes_per_src[src] = ax

        # ---- Plot: Model predictions ----
        for k, model_id in enumerate(model_ids):
            g = model_group_start + k
            n_real = n_real_per_model[model_id]

            for t, src in enumerate(target_srcs):
                if src not in predictions[model_id]:
                    for r in range(n_real):
                        image_axes[(g, t * n_real + r)].axis("off")
                    continue

                pred_data = predictions[model_id][src]
                tgt_ndim = len(target_sources[src].shape)
                is_multi_real = len(pred_data.shape) == tgt_ndim + 1
                display = self._display_src_name(src.name)

                for r in range(n_real):
                    ax = image_axes[(g, t * n_real + r)]
                    realization = pred_data[r, ...] if is_multi_real else pred_data
                    ax.imshow(realization, aspect="auto", cmap=self.cmap, norm=norms[src])
                    title = display if n_real == 1 else f"{display} ({r + 1})"
                    ax.set_title(title, fontsize=7)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if not is_multi_real:
                        break  # deterministic: one column used, loop ends

        # ---- Colorbars (one per target source, manually placed at the right edge) ----
        # The rect parameter reserves right margin for colorbars and top margin for suptitle.
        cbar_w = 0.018
        cbar_gap = 0.005
        cbar_pad = 0.004
        right_margin = cbar_pad + n_tgt * cbar_w + max(0, n_tgt - 1) * cbar_gap

        fig.tight_layout(rect=(0, 0, 1.0 - right_margin, 0.93))

        for i, src in enumerate(target_srcs):
            # Get the image-row position of the corresponding groundtruth axis
            pos = gt_axes_per_src[src].get_position()
            cbar_left = 1.0 - right_margin + cbar_pad + i * (cbar_w + cbar_gap)
            cax = fig.add_axes((cbar_left, pos.y0, cbar_w, pos.height))
            mappable = ScalarMappable(norm=norms[src], cmap=self.cmap)
            mappable.set_array([])
            cbar = fig.colorbar(mappable, cax=cax, label="Temperature (K)")
            cbar.ax.tick_params(labelsize=7)

        # ---- Title and save ----
        fig.suptitle(f"Predictions for channel {channel}", fontsize=9)
        fig_stem = self.metric_results_dir / f"sample_{sample_index}_{channel}"
        fig.savefig(str(fig_stem) + ".svg", format="svg")
        fig.savefig(str(fig_stem) + ".pdf")
        plt.close(fig)

    @staticmethod
    def _set_coords_as_ticks(ax: Any, lats: np.ndarray, lons: np.ndarray, n_ticks: int = 5):
        """Set lat/lon coordinates as axis tick labels."""
        x_indices = np.linspace(0, len(lons[0]) - 1, n_ticks, dtype=int)
        y_indices = np.linspace(0, len(lats[:, 0]) - 1, n_ticks, dtype=int)
        ax.set_xticks(x_indices)
        ax.set_xticklabels([f"{lons[0, i]:.2f}" for i in x_indices], rotation=45)
        ax.set_yticks(y_indices)
        ax.set_yticklabels([f"{lats[i, 0]:.2f}" for i in y_indices])
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
