"""Implements the QuantativeEvaluation class."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import xarray as xr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from motif.datatypes import SourceIndex
from motif.eval.abstract_evaluation_metric import AbstractMultisourceEvaluationMetric
from motif.eval.plot_style import PANEL_HEIGHT, TWO_COL_WIDTH, apply_paper_style


def flatten_and_ignore_nans(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Given an array of predictions and targets, flattens them along all dimensions
    except the first one (realizations), and ignores NaNs in the target array."""
    # Create a mask of valid (non-NaN) entries in the target array
    valid_mask = ~np.isnan(target)
    # Apply the mask to both arrays and flatten them
    pred_flat = pred[:, valid_mask].reshape(pred.shape[0], -1)
    target_flat = target[valid_mask].flatten()
    return pred_flat, target_flat


class QuantitativeEvaluation(AbstractMultisourceEvaluationMetric):
    """Evaluation class that computes sample-wise metrics for a given set of models:
    - Mean Absolute Error (MAE)
    - Mean Square Error (MSE)
    - Continuous Ranked Probability Score (CRPS). For models with a single prediction,
        this is equivalent to the MAE.
    - Skill-Spread Ratio (SSR). This is only computed if the model has multiple
        realizations per sample.
    - Pearson Correlation Coefficient (Corr).
    - Structural Similarity Index (SSIM).
    The per-sample metrics are saved to disk in a JSON file. Then, the aggregated metrics
    are computed and saved to disk in a separate JSON file. Figures to compare the
    models are generated and saved to disk.
    """

    def __init__(
        self,
        model_data: dict[str, dict],
        parent_results_dir: str | Path,
        xlabel: str = "Model",
        num_workers: int = 1,
        **kwargs,
    ):
        """
        Args:
            model_data (dict): Dictionary mapping model_ids to model specifications.
            parent_results_dir (str or Path): Parent directory for all results.
            xlabel (str or None): Label for the x-axis in plots.
            num_workers (int): Number of workers for parallel processing.
            **kwargs: Additional keyword arguments passed to the AbstractMultisourceEvaluationMetric.
        """
        super().__init__(
            id_name="quantitative",
            full_name="Quantitative Evaluation",
            model_data=model_data,
            parent_results_dir=parent_results_dir,
            **kwargs,
        )
        self.num_workers = num_workers
        self.overall_metrics_dir = self.metric_results_dir / "overall"
        self.rmse_dir = self.metric_results_dir / "rmse"
        self.mae_dir = self.metric_results_dir / "mae"
        self.crps_dir = self.metric_results_dir / "crps"
        self.ssr_dir = self.metric_results_dir / "ssr"
        self.corr_dir = self.metric_results_dir / "corr"
        self.ssim_dir = self.metric_results_dir / "ssim"
        # Create the directories if they don't exist
        self.overall_metrics_dir.mkdir(parents=True, exist_ok=True)
        self.rmse_dir.mkdir(parents=True, exist_ok=True)
        self.mae_dir.mkdir(parents=True, exist_ok=True)
        self.crps_dir.mkdir(parents=True, exist_ok=True)
        self.ssr_dir.mkdir(parents=True, exist_ok=True)
        self.corr_dir.mkdir(parents=True, exist_ok=True)
        self.ssim_dir.mkdir(parents=True, exist_ok=True)

        # Within the label, replace double underscores with linebreaks and
        # single underscores with spaces, to get nicer labels in the plots.
        self.xlabel = xlabel.replace("__", "\n").replace("_", " ")

    def evaluate(self, **kwargs):
        """Main evaluation method that processes the data for all models."""

        # Evaluate all models and save the results
        results = self._evaluate_models()
        results_file = self.metric_results_dir / "full_results.json"
        results.to_json(results_file, orient="records", lines=True)
        print(f"Full results saved to: {results_file}")

        # Compute and save the aggregated results
        agg_results = self._save_aggregated_results(results)

        # Generate and save plots comparing the models
        self._plot_results(results, agg_results)
        return

    def _evaluate_models(self):
        """Evaluates all models at once and returns the results.
        Returns:
            results (pd.DataFrame): DataFrame containing the evaluation results for the model.
                Includes the columns 'model_id', 'sample_index', 'source_name', 'source_index',
                'channel', 'mae', 'mse', 'crps', 'ssr'.
        """
        # Collect all samples into a list for parallel processing
        samples = list(self.samples_iterator())

        if self.num_workers > 1:
            # Parallel processing
            results = []
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit all tasks
                futures = [executor.submit(self._process_sample, *sample) for sample in samples]
                # Process results with progress bar
                for future in tqdm(
                    as_completed(futures), desc="Evaluating samples", total=len(futures)
                ):
                    sample_results = future.result()
                    results.extend(sample_results)
        else:
            # Sequential processing
            results = []
            for sample in tqdm(samples, desc="Evaluating samples", total=self.n_samples):
                sample_results = self._process_sample(*sample)
                results.extend(sample_results)

        # Concatenate all results into a single DataFrame
        return pd.DataFrame(results)

    def _process_sample(
        self,
        sample_df: pd.DataFrame,
        true_obs: dict[str, dict[SourceIndex, xr.Dataset]],
        preds: dict[str, dict[SourceIndex, xr.Dataset]],
    ) -> list[dict]:
        """Process a single sample and return the results (helper method for parallel execution).

        Args:
            sample_df (pandas.DataFrame): DataFrame with sample metadata
            true_obs (dict): Dict mapping model_ids to dicts of source indices
                to true observation xarray datasets
            preds (dict): Dict mapping model_ids to dicts of source indices to prediction datasets

        Returns:
            list: List of result dictionaries for this sample
        """
        results = []
        sample_index = sample_df["sample_index"].iloc[0]

        # Process each model sequentially, then for each model process each source.
        for model_id, model_true_obs in true_obs.items():
            for src, true_obs_data in model_true_obs.items():
                source_name, source_index = src.name, src.index

                # We only evaluate sources that were masked, i.e. for which the availability flag
                # is 0 (1 meaning available but not masked, -1 meaning not available).
                if sample_df.loc[(model_id, source_name, source_index), "avail"] != 0:
                    continue
                    # Retrieve the list of channels for the source
                channels = list(true_obs_data.data_vars.keys())

                pred_data = preds[model_id][src]
                for channel in channels:
                    pred_data_channel = pred_data[channel].values
                    target_data_channel = true_obs_data[channel].values
                    # If there isn't a realization dimension, add one for consistency
                    if pred_data_channel.ndim == target_data_channel.ndim:
                        pred_data_channel = np.expand_dims(pred_data_channel, axis=0)
                    # Compute the metrics for the current channel
                    mae = self._compute_mae(pred_data_channel, target_data_channel)
                    mape = self._compute_mape(pred_data_channel, target_data_channel)
                    mse = self._compute_mse(pred_data_channel, target_data_channel)
                    crps = self._compute_crps(pred_data_channel, target_data_channel)
                    corr = self._compute_corr(pred_data_channel, target_data_channel)
                    ssim_val = self._compute_ssim(pred_data_channel, target_data_channel)
                    sample_results_dict = {
                        "model_id": model_id,
                        "sample_index": sample_index,
                        "source_name": source_name,
                        "source_index": source_index,
                        "channel": channel,
                        "mae": mae,
                        "mape": mape,
                        "mse": mse,
                        "crps": crps,
                        "corr": corr,
                        "ssim": ssim_val,
                    }
                    n_real = pred_data_channel.shape[0]
                    if n_real > 1:
                        # If there are multiple realizations, we can compute the SSR.
                        # We'll here just compute the MSE and the ensemble member variance,
                        # and aggregate them later to get the SSR.
                        ensemble_var, ensemble_mean_mse = self._compute_err_and_member_var(
                            pred_data_channel, target_data_channel
                        )
                        sample_results_dict["ssr"] = np.sqrt(
                            ((n_real + 1) / n_real) * (ensemble_var / ensemble_mean_mse)
                        )
                    results.append(sample_results_dict)

        return results

    def _save_aggregated_results(self, results: pd.DataFrame) -> pd.DataFrame:
        """Computes and saves the aggregated results for all models.

        Args:
            results (pd.DataFrame): DataFrame containing the evaluation results.
        """
        # Compute aggregated metrics for each model: mean and std.
        agg_results = (
            results.groupby("model_id")
            .agg(
                mae_mean=("mae", "mean"),
                mape_mean=("mape", "mean"),
                mse_mean=("mse", "mean"),
                crps_mean=("crps", "mean"),
                corr_mean=("corr", "mean"),
                ssim_mean=("ssim", "mean"),
            )
            .reset_index()
        )

        # Sort the aggregated results so that the models are in the same order
        # as in results
        unique_model_ids = results["model_id"].unique()
        agg_results = agg_results.set_index("model_id").loc[unique_model_ids].reset_index()

        # Compute bootstrap 95% confidence intervals for the mean metrics
        for model_id in agg_results["model_id"]:
            model_results = results[results["model_id"] == model_id]

            for metric in ["mae", "mape", "mse", "crps", "corr", "ssim"]:
                data = model_results[metric].values
                # Bootstrap confidence interval
                res = stats.bootstrap(
                    (data,), np.mean, n_resamples=1000, confidence_level=0.95, method="percentile"
                )
                agg_results.loc[agg_results["model_id"] == model_id, f"{metric}_ci_lower"] = (
                    res.confidence_interval.low
                )
                agg_results.loc[agg_results["model_id"] == model_id, f"{metric}_ci_upper"] = (
                    res.confidence_interval.high
                )

        # Compute RMSE from MSE
        agg_results["rmse_mean"] = np.sqrt(agg_results["mse_mean"])

        # Bootstrap CI for RMSE
        for model_id in agg_results["model_id"]:
            model_results = results[results["model_id"] == model_id]
            data = model_results["mse"].values
            res = stats.bootstrap(
                (data,),
                lambda x: np.sqrt(np.mean(x)),
                n_resamples=1000,
                confidence_level=0.95,
                method="percentile",
            )
            agg_results.loc[agg_results["model_id"] == model_id, "rmse_ci_lower"] = (
                res.confidence_interval.low
            )
            agg_results.loc[agg_results["model_id"] == model_id, "rmse_ci_upper"] = (
                res.confidence_interval.high
            )

        # Save the aggregated results to a JSON file
        agg_results_file = self.metric_results_dir / "aggregated_results.json"
        agg_results.to_json(agg_results_file, orient="records", lines=True)
        print(f"Aggregated results saved to: {agg_results_file}")
        return agg_results

    def _plot_results(self, results: pd.DataFrame, agg_results: pd.DataFrame):
        """Generates and saves plots comparing the models based on the evaluation results.
        Args:
            results (pd.DataFrame): DataFrame containing the evaluation results.
            agg_results (pd.DataFrame): DataFrame containing the aggregated evaluation results.
        """
        # Configure plotting style for publication quality
        apply_paper_style()

        # Scale figure width with the number of models to avoid overlapping x-tick labels.
        n_models = results["model_id"].nunique()
        fig_width = max(TWO_COL_WIDTH, n_models * 1.75)

        # First, we'll show plots of the metrics for each model, over all sources and channels.
        # This gives a general overview of the models' performance.
        # MAE: we'll make a boxplot to show the distribution of MAE values for each model.
        plt.figure(figsize=(fig_width, PANEL_HEIGHT))
        sns.boxplot(
            x="model_id",
            y="mae",
            data=results,
            showfliers=False,
            color="steelblue",
        )
        plt.xlabel(self.xlabel)
        plt.ylabel("MAE")
        plt.xticks(rotation=0)
        plt.grid(axis="y")
        sns.despine()
        plt.tight_layout()
        overall_mae_plot_file = self.overall_metrics_dir / "mae_boxplot_all_models.svg"
        plt.savefig(overall_mae_plot_file)
        plt.savefig(overall_mae_plot_file.with_suffix(".pdf"))
        plt.close()
        print(f"Overall MAE plot saved to: {overall_mae_plot_file}")

        # We'll also make a barplot showing the mean MAE with 95% CI error bars.
        plt.figure(figsize=(fig_width, PANEL_HEIGHT))
        sns.barplot(
            x="model_id",
            y="mae",
            data=results,
            errorbar=("ci", 95),
            color="steelblue",
        )
        plt.xlabel(self.xlabel)
        plt.ylabel("MAE")
        plt.xticks(rotation=0)
        plt.grid(axis="y")
        sns.despine()
        plt.tight_layout()
        overall_mae_barplot_file = self.overall_metrics_dir / "mae_barplot_all_models.svg"
        plt.savefig(overall_mae_barplot_file)
        plt.savefig(overall_mae_barplot_file.with_suffix(".pdf"))
        plt.close()
        print(f"Overall MAE bar plot saved to: {overall_mae_barplot_file}")

        # MAPE: barplot with 95% CI error bars.
        plt.figure(figsize=(fig_width, PANEL_HEIGHT))
        sns.barplot(
            x="model_id",
            y="mape",
            data=results,
            errorbar=("ci", 95),
            color="steelblue",
        )
        plt.xlabel(self.xlabel)
        plt.ylabel("MAPE (%)")
        plt.xticks(rotation=0)
        plt.grid(axis="y")
        sns.despine()
        plt.tight_layout()
        overall_mape_barplot_file = self.overall_metrics_dir / "mape_barplot_all_models.svg"
        plt.savefig(overall_mape_barplot_file)
        plt.savefig(overall_mape_barplot_file.with_suffix(".pdf"))
        plt.close()
        print(f"Overall MAPE bar plot saved to: {overall_mape_barplot_file}")

        # MAPE: boxplot
        plt.figure(figsize=(fig_width, PANEL_HEIGHT))
        sns.boxplot(
            x="model_id",
            y="mape",
            data=results,
            showfliers=False,
            color="steelblue",
        )
        plt.xlabel(self.xlabel)
        plt.ylabel("MAPE (%)")
        plt.xticks(rotation=0)
        plt.grid(axis="y")
        sns.despine()
        plt.tight_layout()
        overall_mape_plot_file = self.overall_metrics_dir / "mape_boxplot_all_models.svg"
        plt.savefig(overall_mape_plot_file)
        plt.savefig(overall_mape_plot_file.with_suffix(".pdf"))
        plt.close()

        # RMSE: we'll make a barplot, since the RMSE is a single value per model.
        fig, ax = plt.subplots(figsize=(fig_width, PANEL_HEIGHT))
        sns.barplot(
            x="model_id",
            y="rmse_mean",
            data=agg_results,
            errorbar=None,
            color="steelblue",
            ax=ax,
        )
        # Draw error bars manually for RMSE
        ax.errorbar(
            np.arange(len(agg_results)),
            agg_results["rmse_mean"],
            yerr=[
                agg_results["rmse_mean"] - agg_results["rmse_ci_lower"],
                agg_results["rmse_ci_upper"] - agg_results["rmse_mean"],
            ],
            fmt="none",
            c="black",
            capsize=5,
        )
        plt.xlabel(self.xlabel)
        plt.ylabel("RMSE")
        plt.xticks(rotation=0)
        ax.grid(axis="y")
        sns.despine()
        plt.tight_layout()
        overall_rmse_barplot_file = self.overall_metrics_dir / "rmse_barplot_all_models.svg"
        plt.savefig(overall_rmse_barplot_file)
        plt.savefig(overall_rmse_barplot_file.with_suffix(".pdf"))
        plt.close()
        print(f"Overall RMSE bar plot saved to: {overall_rmse_barplot_file}")

        # CRPS: boxplot
        plt.figure(figsize=(fig_width, PANEL_HEIGHT))
        sns.boxplot(
            x="model_id",
            y="crps",
            data=results,
            showfliers=False,
            color="steelblue",
        )
        plt.xlabel(self.xlabel)
        plt.ylabel("CRPS")
        plt.xticks(rotation=0)
        plt.grid(axis="y")
        sns.despine()
        plt.tight_layout()
        overall_crps_plot_file = self.overall_metrics_dir / "crps_all_models.svg"
        plt.savefig(overall_crps_plot_file)
        plt.savefig(overall_crps_plot_file.with_suffix(".pdf"))
        plt.close()
        print(f"Overall CRPS plot saved to: {overall_crps_plot_file}")

        # CRPS: barplot
        plt.figure(figsize=(fig_width, PANEL_HEIGHT))
        sns.barplot(
            x="model_id",
            y="crps",
            data=results,
            errorbar=("ci", 95),
            color="steelblue",
        )
        plt.xlabel(self.xlabel)
        plt.ylabel("CRPS")
        plt.xticks(rotation=0)
        plt.grid(axis="y")
        sns.despine()
        plt.tight_layout()
        overall_crps_barplot_file = self.overall_metrics_dir / "crps_barplot_all_models.svg"
        plt.savefig(overall_crps_barplot_file)
        plt.savefig(overall_crps_barplot_file.with_suffix(".pdf"))
        plt.close()
        print(f"Overall CRPS bar plot saved to: {overall_crps_barplot_file}")

        # Corr: boxplot
        plt.figure(figsize=(fig_width, PANEL_HEIGHT))
        sns.boxplot(
            x="model_id",
            y="corr",
            data=results,
            showfliers=False,
            color="steelblue",
        )
        plt.xlabel(self.xlabel)
        plt.ylabel("Correlation")
        plt.xticks(rotation=0)
        plt.grid(axis="y")
        sns.despine()
        plt.tight_layout()
        overall_corr_plot_file = self.overall_metrics_dir / "corr_all_models.svg"
        plt.savefig(overall_corr_plot_file)
        plt.savefig(overall_corr_plot_file.with_suffix(".pdf"))
        plt.close()
        print(f"Overall Correlation plot saved to: {overall_corr_plot_file}")

        # Corr: barplot
        plt.figure(figsize=(fig_width, PANEL_HEIGHT))
        sns.barplot(
            x="model_id",
            y="corr",
            data=results,
            errorbar=("ci", 95),
            color="steelblue",
        )
        plt.xlabel(self.xlabel)
        plt.ylabel("Correlation")
        plt.xticks(rotation=0)
        plt.grid(axis="y")
        sns.despine()
        plt.tight_layout()
        overall_corr_barplot_file = self.overall_metrics_dir / "corr_barplot_all_models.svg"
        plt.savefig(overall_corr_barplot_file)
        plt.savefig(overall_corr_barplot_file.with_suffix(".pdf"))
        plt.close()
        print(f"Overall Correlation bar plot saved to: {overall_corr_barplot_file}")

        # SSIM: boxplot
        plt.figure(figsize=(fig_width, PANEL_HEIGHT))
        sns.boxplot(
            x="model_id",
            y="ssim",
            data=results,
            showfliers=False,
            color="steelblue",
        )
        plt.xlabel(self.xlabel)
        plt.ylabel("SSIM")
        plt.xticks(rotation=0)
        plt.grid(axis="y")
        sns.despine()
        plt.tight_layout()
        overall_ssim_plot_file = self.overall_metrics_dir / "ssim_all_models.svg"
        plt.savefig(overall_ssim_plot_file)
        plt.savefig(overall_ssim_plot_file.with_suffix(".pdf"))
        plt.close()
        print(f"Overall SSIM plot saved to: {overall_ssim_plot_file}")

        # SSIM: barplot
        plt.figure(figsize=(fig_width, PANEL_HEIGHT))
        sns.barplot(
            x="model_id",
            y="ssim",
            data=results,
            errorbar=("ci", 95),
            color="steelblue",
        )
        plt.xlabel(self.xlabel)
        plt.ylabel("SSIM")
        plt.xticks(rotation=0)
        plt.grid(axis="y")
        sns.despine()
        plt.tight_layout()
        overall_ssim_barplot_file = self.overall_metrics_dir / "ssim_barplot_all_models.svg"
        plt.savefig(overall_ssim_barplot_file)
        plt.savefig(overall_ssim_barplot_file.with_suffix(".pdf"))
        plt.close()
        print(f"Overall SSIM bar plot saved to: {overall_ssim_barplot_file}")

        # SSR: Only for models that have multiple realizations
        if "ssr" in results.columns:
            # Boxplot
            plt.figure(figsize=(fig_width, PANEL_HEIGHT))
            sns.boxplot(
                x="model_id",
                y="ssr",
                data=results,
                showfliers=False,
                color="steelblue",
            )
            plt.xlabel(self.xlabel)
            plt.ylabel("SSR")
            plt.axhline(y=1, color="black", linestyle="--", linewidth=1)
            plt.xticks(rotation=0)
            plt.grid(axis="y")
            sns.despine()
            plt.tight_layout()
            overall_ssr_plot_file = self.overall_metrics_dir / "ssr_all_models.svg"
            plt.savefig(overall_ssr_plot_file)
            plt.savefig(overall_ssr_plot_file.with_suffix(".pdf"))
            plt.close()
            print(f"Overall SSR plot saved to: {overall_ssr_plot_file}")

            # Barplot
            plt.figure(figsize=(fig_width, PANEL_HEIGHT))
            sns.barplot(
                x="model_id",
                y="ssr",
                data=results,
                errorbar=("ci", 95),
                color="steelblue",
            )
            plt.xlabel(self.xlabel)
            plt.ylabel("SSR")
            plt.axhline(y=1, color="black", linestyle="--", linewidth=1)
            plt.xticks(rotation=0)
            plt.grid(axis="y")
            sns.despine()
            plt.tight_layout()
            overall_ssr_barplot_file = self.overall_metrics_dir / "ssr_barplot_all_models.svg"
            plt.savefig(overall_ssr_barplot_file)
            plt.savefig(overall_ssr_barplot_file.with_suffix(".pdf"))
            plt.close()
            print(f"Overall SSR bar plot saved to: {overall_ssr_barplot_file}")

        # Now, we'll separate the plots by pair (source_name, channel). In each plot,
        # the x-axis will be the model_id and the y-axis will be the metric.
        grouped_results = results.groupby(["source_name", "channel"])
        for (source_name, channel), group in grouped_results:
            # MAE plot
            plt.figure(figsize=(fig_width, PANEL_HEIGHT))
            sns.boxplot(
                x="model_id",
                y="mae",
                data=group,
                showfliers=False,
                color="steelblue",
            )
            plt.title(f"MAE for {source_name} - {channel}")
            plt.xlabel(self.xlabel)
            plt.ylabel("Mean Absolute Error")
            plt.xticks(rotation=0)
            plt.grid(axis="y")
            sns.despine()
            plt.tight_layout()
            mae_plot_file = self.mae_dir / f"mae_{source_name}_{channel}.svg"
            plt.savefig(mae_plot_file)
            plt.savefig(mae_plot_file.with_suffix(".pdf"))
            plt.close()
            print(f"MAE plot saved to: {mae_plot_file}")

            # RMSE
            rmse_per_model = group.groupby("model_id")["mse"].mean().reset_index()
            rmse_per_model["rmse"] = np.sqrt(rmse_per_model["mse"])
            plt.figure(figsize=(fig_width, PANEL_HEIGHT))
            sns.barplot(
                x="model_id",
                y="rmse",
                data=rmse_per_model,
                color="steelblue",
            )
            plt.title(f"RMSE for {source_name} - {channel}")
            plt.xlabel(self.xlabel)
            plt.ylabel("Root Mean Square Error")
            plt.xticks(rotation=0)
            plt.grid(axis="y")
            sns.despine()
            plt.tight_layout()
            rmse_plot_file = self.rmse_dir / f"rmse_{source_name}_{channel}.svg"
            plt.savefig(rmse_plot_file)
            plt.savefig(rmse_plot_file.with_suffix(".pdf"))
            plt.close()
            print(f"RMSE plot saved to: {rmse_plot_file}")

            # Corr per source/channel
            corr_per_model = group.groupby("model_id")["corr"].mean().reset_index()
            plt.figure(figsize=(fig_width, PANEL_HEIGHT))
            sns.barplot(
                x="model_id",
                y="corr",
                data=corr_per_model,
                color="steelblue",
            )
            plt.title(f"Correlation for {source_name} - {channel}")
            plt.xlabel(self.xlabel)
            plt.ylabel("Pearson Correlation")
            plt.xticks(rotation=0)
            plt.grid(axis="y")
            sns.despine()
            plt.tight_layout()
            corr_plot_file = self.corr_dir / f"corr_{source_name}_{channel}.svg"
            plt.savefig(corr_plot_file)
            plt.savefig(corr_plot_file.with_suffix(".pdf"))
            plt.close()
            print(f"Correlation plot saved to: {corr_plot_file}")

            # SSIM per source/channel
            ssim_per_model = group.groupby("model_id")["ssim"].mean().reset_index()
            plt.figure(figsize=(fig_width, PANEL_HEIGHT))
            sns.barplot(
                x="model_id",
                y="ssim",
                data=ssim_per_model,
                color="steelblue",
            )
            plt.title(f"SSIM for {source_name} - {channel}")
            plt.xlabel(self.xlabel)
            plt.ylabel("SSIM")
            plt.xticks(rotation=0)
            plt.grid(axis="y")
            sns.despine()
            plt.tight_layout()
            ssim_plot_file = self.ssim_dir / f"ssim_{source_name}_{channel}.svg"
            plt.savefig(ssim_plot_file)
            plt.savefig(ssim_plot_file.with_suffix(".pdf"))
            plt.close()
            print(f"SSIM plot saved to: {ssim_plot_file}")

    @staticmethod
    def _compute_mae(pred_data: np.ndarray, target_data: np.ndarray) -> float:
        """Computes the Mean Absolute Error (MAE) between the predicted ensemble mean
        and targets."""
        pred_flat, target_flat = flatten_and_ignore_nans(pred_data, target_data)
        # Average over realizations
        pred_flat = pred_flat.mean(axis=0)
        return np.abs((pred_flat - target_flat)).mean().item()

    @staticmethod
    def _compute_mape(pred_data: np.ndarray, target_data: np.ndarray) -> float:
        """Computes the Mean Absolute Percentage Error (MAPE) between the predicted
        ensemble mean and targets."""
        pred_flat, target_flat = flatten_and_ignore_nans(pred_data, target_data)
        # Average over realizations
        pred_flat = pred_flat.mean(axis=0)
        return (np.abs((pred_flat - target_flat) / target_flat)).mean().item() * 100

    @staticmethod
    def _compute_mse(pred_data: np.ndarray, target_data: np.ndarray) -> float:
        """Computes the Mean Square Error (MSE) between predicted ensemble mean
        and targets."""
        pred_flat, target_flat = flatten_and_ignore_nans(pred_data, target_data)
        # Average over realizations
        pred_flat = pred_flat.mean(axis=0)
        return ((pred_flat - target_flat) ** 2).mean().item()

    @staticmethod
    def _compute_corr(pred_data: np.ndarray, target_data: np.ndarray) -> float:
        """Computes the Pearson correlation coefficient between the predicted ensemble mean
        and targets. Returns NaN if the target has no variance.

        Args:
            pred_data (np.ndarray): Predicted data, of shape (M, ...) where M is the number of
                realizations (possibly 1 for deterministic models).
            target_data (np.ndarray): Target data, of shape (...) matching the shape of each
                realization in pred_data.
        """
        pred_flat, target_flat = flatten_and_ignore_nans(pred_data, target_data)
        # Average over realizations
        pred_flat = pred_flat.mean(axis=0)
        if target_flat.std() == 0 or pred_flat.std() == 0:
            return float("nan")
        return np.corrcoef(pred_flat, target_flat)[0, 1].item()

    @staticmethod
    def _compute_crps(pred_data: np.ndarray, target_data: np.ndarray) -> float:
        """Computes the Fair Continuous Ranked Probability Score (CRPS)
        between predictions and targets.
        For deterministic predictions, this is equivalent to the MAE.

        Args:
            pred_data (np.ndarray): Predicted data, of shape (M, ...) where M is the number of
                realizations (possibly 1 for deterministic models).
            target_data (np.ndarray): Target data, of shape (...) matching the shape of each
                realization in pred_data.
        """
        pred_data, target_data = flatten_and_ignore_nans(pred_data, target_data)
        # Compute the first term: the mean absolute error between predictions and target
        term1 = np.abs(pred_data - target_data).mean()
        # Compute the second term: the mean absolute error between all pairs of predictions
        term2 = np.abs(pred_data[:, None] - pred_data[None, :]).mean(axis=-1).sum()
        ens_size = pred_data.shape[0]
        if ens_size > 1:
            term2 = term2 / (ens_size * (ens_size - 1))
        else:
            term2 = term2 / (ens_size * ens_size)
        crps = term1 - 0.5 * term2
        return crps.item()

    @staticmethod
    def _compute_ssim(pred_data: np.ndarray, target_data: np.ndarray) -> float:
        """Computes the Structural Similarity Index (SSIM) between the predictions and
        targets. The SSIM is then averaged over the realizations.

        Args:
            pred_data (np.ndarray): Predicted data, of shape (M, ...) where M is the number of
                realizations (possibly 1 for deterministic models).
            target_data (np.ndarray): Target data, of shape (...) matching the shape of each
                realization in pred_data.
        """
        # Replace NaNs in target_data with zeros (SSIM does not handle NaNs)
        target_data = np.nan_to_num(target_data, nan=0.0)
        pred_data = np.nan_to_num(pred_data, nan=0.0)

        ssim_values = []
        data_range = target_data.max() - target_data.min()
        for i in range(pred_data.shape[0]):
            ssim_val = ssim(
                pred_data[i],
                target_data,
                data_range=data_range,
                gaussian_weights=True,
                win_size=11,
                sigma=1.5,
                K1=0.01,
                K2=0.03,
            )
            ssim_values.append(ssim_val)
        return float(np.mean(ssim_values))

    @staticmethod
    def _compute_err_and_member_var(
        pred_data: np.ndarray, target_data: np.ndarray
    ) -> tuple[float, float]:
        """Computes the unbiased MSE between the ensemble mean and the target,
        as well as the ensemble member variance.

        Args:
            pred_data (np.ndarray): Predicted data, of shape (M, ...) where M is the number of
                realizations.
            target_data (np.ndarray): Target data, of shape (...) matching the shape of each
                realization in pred_data.
        Returns:
            var (float): Ensemble member variance.
            mean_mse (float): Debiased MSE between the ensemble mean and the target.
        """
        pred_data, target_data = flatten_and_ignore_nans(pred_data, target_data)
        K = pred_data.shape[0]  # Number of ensemble members
        # Compute the ensemble mean
        ensemble_mean = pred_data.mean(axis=0)
        # Compute the finite-sample variance of the ensemble members
        var = ((pred_data - ensemble_mean[None, :]) ** 2).mean() * (K / (K - 1))
        # Compute the unbiased Mean Square error (MSE)
        mean_mse = ((ensemble_mean - target_data) ** 2).mean()
        return var, mean_mse
