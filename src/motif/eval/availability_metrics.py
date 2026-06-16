"""Implements the AvailabilityMetricsEvaluation class.

This evaluation does not compute any new metric or load any prediction data.
Instead, it joins the per-sample metrics produced by the ``quantitative`` evaluation
(``quantitative/full_results.json``) with the predictions index (``self.samples_df``,
built in the parent class) in order to analyze how reconstruction quality depends on
the inputs that were actually available for each sample:

* the number of input observations in the sample;
* the signed time difference between the target and its temporally closest input;
* which input source modalities are present.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from motif.eval.abstract_evaluation_metric import AbstractMultisourceEvaluationMetric
from motif.eval.plot_style import (
    PANEL_HEIGHT,
    TALL_PANEL_HEIGHT,
    TWO_COL_WIDTH,
    apply_paper_style,
    get_model_palette,
    legend_below,
)

# Columns of the quantitative full_results.json that are identifiers rather than metrics.
_ID_COLUMNS = {"model_id", "sample_index", "source_name", "source_index", "channel"}

# Nicer y-axis labels for known metrics; unknown metrics fall back to upper-case.
_METRIC_LABELS = {
    "mae": "MAE",
    "mape": "MAPE",
    "mse": "MSE",
    "rmse": "RMSE",
    "crps": "CRPS",
    "corr": "Correlation",
    "ssim": "SSIM",
    "ssr": "SSR",
}


class AvailabilityMetricsEvaluation(AbstractMultisourceEvaluationMetric):
    """Evaluation class that relates the quantitative metrics to the availability of
    the inputs in each sample.

    It produces, for every metric, three ensembles of figures:

    * line plots of the metric against the number of input observations in the sample;
    * line plots of the metric against the (binned) signed time difference between the
      target and its temporally closest input;
    * bar plots and box plots of the metric for each input source, restricted to the
      samples that contain that source at least once.

    The class depends on the ``quantitative`` evaluation having been run first in the
    same eval run, as it reads ``quantitative/full_results.json``.
    """

    def __init__(
        self,
        model_data,
        parent_results_dir,
        dt_bin_width=0.5,
        num_workers=1,
        **kwargs,
    ):
        """
        Args:
            model_data (dict): Dictionary mapping model_ids to model specifications.
            parent_results_dir (str or Path): Parent directory for all results.
            dt_bin_width (float): Width (in hours) of the signed time-difference bins
                used for the "metric vs min dt" line plots. The bin edges are aligned
                to the target time (0), i.e. they fall at integer multiples of this
                value. Defaults to 0.5 (30 minutes).
            num_workers (int): Here for compatibility.
            **kwargs: Additional keyword arguments passed to the parent class
                (source_name_replacements, channel_replacements, checks_strictness).
        """
        super().__init__(
            id_name="availability_metrics",
            full_name="Metrics vs Source Availability",
            model_data=model_data,
            parent_results_dir=parent_results_dir,
            **kwargs,
        )
        self.num_workers = num_workers
        self.dt_bin_width = dt_bin_width

        # Output sub-directories, one per figure ensemble.
        self.vs_num_sources_dir = self.metric_results_dir / "vs_num_sources"
        self.vs_min_dt_dir = self.metric_results_dir / "vs_min_dt"
        self.per_source_barplot_dir = self.metric_results_dir / "per_source_barplot"
        self.per_source_boxplot_dir = self.metric_results_dir / "per_source_boxplot"
        for d in (
            self.vs_num_sources_dir,
            self.vs_min_dt_dir,
            self.per_source_barplot_dir,
            self.per_source_boxplot_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

        # Load the per-sample metrics from the quantitative evaluation.
        self.quant_results = None
        results_file = self.parent_results_dir / "quantitative" / "full_results.json"
        if results_file.exists():
            self.quant_results = pd.read_json(results_file, orient="records", lines=True)
        else:
            print(
                f"Quantitative results file {results_file} not found. "
                "The availability_metrics evaluation requires the quantitative "
                "evaluation to be run first; skipping."
            )

    def evaluate(self, **kwargs):
        """Main evaluation method. Joins the quantitative per-sample metrics with the
        availability features derived from self.samples_df, then produces the figures.
        """
        if self.quant_results is None:
            return

        apply_paper_style()

        # Per-(model_id, sample_index) metrics, aggregated over channel and source.
        per_sample, metric_cols = self._aggregate_metrics()
        # Per-sample availability features, computed once (availability is shared
        # across models under the sanity checks).
        n_input_obs, signed_closest_dt, input_sources = self._availability_features()

        self._plot_vs_num_sources(per_sample, metric_cols, n_input_obs)
        self._plot_vs_min_dt(per_sample, metric_cols, signed_closest_dt)
        self._plot_per_source(per_sample, metric_cols, input_sources)

    def _aggregate_metrics(self):
        """Aggregate the quantitative metrics over channels and target sources.

        Returns:
            per_sample (pd.DataFrame): one row per (model_id, sample_index) with one
                column per metric (the mean over channels and target sources).
            metric_cols (list of str): the metric column names, including derived rmse.
        """
        results = self.quant_results.copy()
        # Derive RMSE from MSE if available (quantitative stores mse, not rmse).
        if "mse" in results.columns:
            results["rmse"] = np.sqrt(results["mse"])
        metric_cols = [c for c in results.columns if c not in _ID_COLUMNS]
        per_sample = results.groupby(["model_id", "sample_index"])[metric_cols].mean().reset_index()
        return per_sample, metric_cols

    def _availability_features(self):
        """Compute the per-(model, sample) availability features from self.samples_df.

        The features are computed per model, not once per sample: only the target
        rows (avail == 0) are guaranteed identical across models, while with
        ``checks_strictness="targets_only"`` the inputs may legitimately differ
        between models. self.samples_df is already unique per
        (model_id, sample_index, source_name, source_index).

        Returns:
            n_input_obs (pd.DataFrame): columns [model_id, sample_index, n_input_obs].
            signed_closest_dt (pd.DataFrame): columns
                [model_id, sample_index, signed_closest_dt].
            input_sources (pd.DataFrame): columns [model_id, sample_index, source_name],
                one row per distinct input source name present in the (model, sample).
        """
        keys = ["model_id", "sample_index"]
        inputs = self.samples_df[self.samples_df["avail"] == 1]
        targets = self.samples_df[self.samples_df["avail"] == 0]

        # Number of input observations (counting (source_name, source_index) pairs).
        n_input_obs = inputs.groupby(keys).size().reset_index(name="n_input_obs")

        # Signed time difference of the input temporally closest to the target.
        target_dt = targets.groupby(keys)["dt"].mean().reset_index(name="target_dt")
        merged = inputs[keys + ["dt"]].merge(target_dt, on=keys)
        merged["signed"] = (merged["dt"] - merged["target_dt"]) / pd.Timedelta(hours=1)
        merged["abs"] = merged["signed"].abs()
        closest_idx = merged.groupby(keys)["abs"].idxmin()
        signed_closest_dt = (
            merged.loc[closest_idx, keys + ["signed"]]
            .rename(columns={"signed": "signed_closest_dt"})
            .reset_index(drop=True)
        )

        # Distinct input source names per (model, sample).
        input_sources = inputs[keys + ["source_name"]].drop_duplicates()

        return n_input_obs, signed_closest_dt, input_sources

    def _metric_label(self, metric):
        """Return a display label for a metric column name."""
        return _METRIC_LABELS.get(metric, metric.upper())

    def _save_fig(self, fig, path):
        """Save a figure as both SVG and PDF and close it."""
        fig.savefig(path)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)

    def _plot_vs_num_sources(self, per_sample, metric_cols, n_input_obs):
        """One line plot per metric: metric vs the number of input observations."""
        data = per_sample.merge(n_input_obs, on=["model_id", "sample_index"])
        palette = get_model_palette(data["model_id"].nunique())
        for metric in metric_cols:
            fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH, PANEL_HEIGHT))
            sns.lineplot(
                data=data,
                x="n_input_obs",
                y=metric,
                hue="model_id",
                estimator="mean",
                errorbar=("ci", 95),
                palette=palette,
                marker="o",
                ax=ax,
            )
            ax.set_xlabel("Number of input observations")
            ax.set_ylabel(self._metric_label(metric))
            ax.grid(axis="y")
            legend_below(ax)
            sns.despine(fig=fig)
            plt.tight_layout()
            self._save_fig(fig, self.vs_num_sources_dir / f"{metric}_vs_num_sources.svg")

    def _plot_vs_min_dt(self, per_sample, metric_cols, signed_closest_dt):
        """One line plot per metric: metric vs the signed time difference (binned) to
        the temporally closest input."""
        data = per_sample.merge(signed_closest_dt, on=["model_id", "sample_index"])
        if data.empty:
            return

        # Build bin edges aligned to the target time (0), at multiples of dt_bin_width.
        w = self.dt_bin_width
        dt_min, dt_max = data["signed_closest_dt"].min(), data["signed_closest_dt"].max()
        start = np.floor(dt_min / w) * w
        stop = np.ceil(dt_max / w) * w
        edges = np.arange(start, stop + w, w)
        if len(edges) < 2:
            edges = np.array([start, start + w])
        midpoints = (edges[:-1] + edges[1:]) / 2
        data = data.copy()
        bin_idx = pd.cut(data["signed_closest_dt"], bins=edges, labels=False, include_lowest=True)
        data["dt_bin"] = midpoints[bin_idx.astype(int)]

        palette = get_model_palette(data["model_id"].nunique())
        for metric in metric_cols:
            fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH, PANEL_HEIGHT))
            sns.lineplot(
                data=data,
                x="dt_bin",
                y=metric,
                hue="model_id",
                estimator="mean",
                errorbar=("ci", 95),
                palette=palette,
                marker="o",
                ax=ax,
            )
            ax.set_xlabel("Signed time difference to closest input (hours)")
            ax.set_ylabel(self._metric_label(metric))
            ax.axvline(0.0, color="grey", linestyle="--", linewidth=1.0)
            ax.grid(axis="y")
            legend_below(ax)
            sns.despine(fig=fig)
            plt.tight_layout()
            self._save_fig(fig, self.vs_min_dt_dir / f"{metric}_vs_min_dt.svg")

    def _plot_per_source(self, per_sample, metric_cols, input_sources):
        """One bar plot and one box plot per metric: the metric distribution for each
        input source, over the samples that contain that source at least once."""
        data = per_sample.merge(input_sources, on=["model_id", "sample_index"])
        if data.empty:
            return
        data = data.copy()
        data["source"] = data["source_name"].map(self._display_src_name)
        # Stable source ordering for consistent figures.
        source_order = sorted(data["source"].unique())
        palette = get_model_palette(data["model_id"].nunique())

        for metric in metric_cols:
            # Bar plot (mean +/- 95% CI), horizontal layout for long source labels.
            fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH, TALL_PANEL_HEIGHT))
            sns.barplot(
                data=data,
                x=metric,
                y="source",
                hue="model_id",
                order=source_order,
                errorbar=("ci", 95),
                palette=palette,
                ax=ax,
            )
            ax.set_xlabel(self._metric_label(metric))
            ax.set_ylabel("Input source")
            legend_below(ax)
            sns.despine(fig=fig)
            plt.tight_layout()
            self._save_fig(fig, self.per_source_barplot_dir / f"{metric}_per_source_bar.svg")

            # Box plot (full distribution).
            fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH, TALL_PANEL_HEIGHT))
            sns.boxplot(
                data=data,
                x=metric,
                y="source",
                hue="model_id",
                order=source_order,
                palette=palette,
                ax=ax,
            )
            ax.set_xlabel(self._metric_label(metric))
            ax.set_ylabel("Input source")
            legend_below(ax)
            sns.despine(fig=fig)
            plt.tight_layout()
            self._save_fig(fig, self.per_source_boxplot_dir / f"{metric}_per_source_box.svg")
