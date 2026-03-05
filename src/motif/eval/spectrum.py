"""Implements the QuantativeEvaluation class."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from motif.eval.abstract_evaluation_metric import AbstractMultisourceEvaluationMetric


class RadiallyAveragedPSDEvaluation(AbstractMultisourceEvaluationMetric):
    """Evaluation class that computes the radially averaged power spectral density (PSD)
    for each model's predictions and the target data, and compares them in plots.
    """

    def __init__(self, model_data: dict[str, dict], parent_results_dir: str | Path, **kwargs):
        """
        Args:
            model_data (dict): Dictionary mapping model_ids to model specifications.
            parent_results_dir (str or Path): Parent directory for all results.
            **kwargs: Additional keyword arguments passed to the AbstractMultisourceEvaluationMetric.
        """
        super().__init__(
            id_name="spectrum",
            full_name="Radially Averaged Power Spectral Density Evaluation",
            model_data=model_data,
            parent_results_dir=parent_results_dir,
            **kwargs,
        )

    def evaluate(self, **kwargs) -> None:
        """Main evaluation method that processes the data for all models."""

        # Evaluate all models and save the results
        results = self._evaluate_models()
        results_file = self.metric_results_dir / "full_results.json"
        results.to_json(results_file, orient="records", lines=True)
        print(f"Full results saved to: {results_file}")

        # Generate and save plots comparing the models
        self._plot_results(results)
        return

    def _evaluate_models(self) -> pd.DataFrame:
        """Evaluates all models at once and returns the results.
        Returns:
            results (pd.DataFrame): DataFrame containing the evaluation results for the model.
                Includes the columns 'model_id', 'sample_index', 'source_name', 'source_index',
                'channel', 'mae', 'mse', 'crps', 'ssr'.
        """
        results = []  # List of dictionaries that we'll concatenate later into a DataFrame
        for sample_df, targets, preds in tqdm(
            self.samples_iterator(), desc="Evaluating samples", total=self.n_samples
        ):
            sample_index = sample_df["sample_index"].iloc[0]
            for src, target_data in targets.items():
                source_name, source_index = src.name, src.index
                src_tuple = (source_name, source_index)
                # We only evaluate sources that were masked, i.e. for which the availability flag
                # is 0 (1 meaning available but not masked, -1 meaning not available).
                if sample_df.loc[src_tuple, "avail"] != 0:
                    continue
                # Retrieve the list of channels for the source
                channels = list(target_data.data_vars.keys())
                # Evaluate each model's predictions against the target data
                # on every channel.
                for model_id in self.model_data:
                    pred_data = preds[model_id][src]
                    for channel in channels:
                        pred_data_channel = pred_data[channel].values
                        target_data_channel = target_data[channel].values
                        # For now, we'll skip the minority of cases where the target data
                        # contains NaNs.
                        if not np.isnan(target_data_channel).any():
                            # If there isn't a realization dimension, add one for consistency
                            if pred_data_channel.ndim == target_data_channel.ndim:
                                pred_data_channel = np.expand_dims(pred_data_channel, axis=0)
                            # Compute the PSD of the predictions and target
                            pred_psd, target_psd, freq = self._compute_psd(
                                pred_data_channel, target_data_channel
                            )
                            # Deduce the PSD gain (ratio of predicted to target PSD)
                            psd_gain = pred_psd / target_psd.clip(min=1e-10)
                            sample_results_dict = {
                                "model_id": model_id,
                                "sample_index": sample_index,
                                "source_name": source_name,
                                "source_index": source_index,
                                "channel": channel,
                                "pred_psd": list(pred_psd),
                                "target_psd": list(target_psd),
                                "psd_gain": list(psd_gain),
                                "freq": list(freq),
                            }
                            results.append(sample_results_dict)

        # Concatenate all results into a single DataFrame
        return (
            pd.DataFrame(results)
            .explode(column=["pred_psd", "target_psd", "psd_gain", "freq"])
            .reset_index(drop=True)
        )

    def _plot_results(self, results: pd.DataFrame) -> None:
        """Generates and saves plots comparing the models' PSDs and PSD gains.

        For each channel, produces a figure with two subplots:
          - Left: average radially averaged PSD for each model's predictions and the target.
          - Right: PSD gain (pred / target) for each model, with a reference line at y=1.

        Args:
            results (pd.DataFrame): DataFrame containing the evaluation results.
        """
        sns.set_theme(style="whitegrid")
        channels = results["channel"].unique()
        for channel in channels:
            channel_results = results[results["channel"] == channel]

            fig, (ax_psd, ax_gain) = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(f"Radially Averaged PSD - Channel: {channel}")

            # Left: predicted PSD per model
            sns.lineplot(
                data=channel_results,
                x="freq",
                y="pred_psd",
                hue="model_id",
                estimator="mean",
                errorbar=("ci", 95),
                ax=ax_psd,
            )
            # Target PSD is model-agnostic; use the first model's rows to avoid duplication
            first_model = channel_results["model_id"].iloc[0]
            target_rows = channel_results[channel_results["model_id"] == first_model]
            sns.lineplot(
                data=target_rows,
                x="freq",
                y="target_psd",
                estimator="mean",
                errorbar=("ci", 95),
                color="black",
                label="target",
                ax=ax_psd,
            )
            ax_psd.set_title("Power Spectral Density")
            ax_psd.set_xlabel("Frequency")
            ax_psd.set_ylabel("PSD")
            ax_psd.set_xscale("log")
            ax_psd.set_yscale("log")
            ax_psd.set_ylim(bottom=1e-2)
            ax_psd.grid(True, which="both", ls="--", lw=0.5)
            ax_psd.legend(title="Model")

            # Right: PSD gain per model + reference line at 1
            sns.lineplot(
                data=channel_results,
                x="freq",
                y="psd_gain",
                hue="model_id",
                estimator="mean",
                ax=ax_gain,
            )
            ax_gain.axhline(y=1.0, color="black", linestyle=":", linewidth=1.5)
            ax_gain.set_title("PSD Gain")
            ax_gain.set_xlabel("Frequency")
            ax_gain.set_ylabel("PSD Gain")
            ax_gain.set_xscale("log")
            ax_gain.set_yscale("log")
            ax_gain.set_ylim(bottom=1e-2, top=5)
            ax_gain.grid(True, which="both", ls="--", lw=0.5)
            ax_gain.legend(title="Model")

            plt.tight_layout()
            plot_file = self.metric_results_dir / f"psd_{channel}.png"
            plt.savefig(plot_file)
            plt.close()
            print(f"Plot saved to: {plot_file}")

    @staticmethod
    def _compute_psd(
        pred_data: np.ndarray, target_data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the radially averaged power spectral density (RAPSD) for the predictions
        and target.

        Args:
            pred_data (ndarray): Predicted data, of shape (M, H, W) where M is the number of
                realizations.
            target_data (ndarray): Target data, of shape (H, W) matching the shape of each
                realization in pred_data.
        Returns:
            pred_psd (ndarray): The RAPSD for the predictions, of shape (L,)
                where L is the number of frequency bins. The PSD is averaged
                over the realizations.
            target_psd (ndarray): The RAPSD for the target, of shape (L,).
            freq (ndarray): The corresponding frequencies, of shape (L,).
        """
        # Remove the mean to avoid a spike at the zero frequency
        pred_data = pred_data - np.mean(pred_data, axis=(-2, -1), keepdims=True)
        target_data = target_data - np.mean(target_data)

        # Compute the RAPSD for each realization and average over realizations
        pred_psd_avg = rapsd(pred_data[0], fft_method=np.fft)
        for i in range(1, pred_data.shape[0]):
            pred_psd = rapsd(pred_data[i], fft_method=np.fft)
            pred_psd_avg += pred_psd
        pred_psd_avg /= pred_data.shape[0]

        # Compute the RAPSD for the target and retrieve the frequencies
        target_psd, freq = rapsd(target_data, fft_method=np.fft, return_freq=True)

        return pred_psd_avg, target_psd, freq


def compute_centred_coord_array(M: int, N: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a 2D coordinate array, where the origin is at the center.
    Taken from the pysteps package.

    M : int
      The height of the array.
    N : int
      The width of the array.

    Returns
    -------
    out : ndarray
      The coordinate array.

    Examples
    --------
    >>> compute_centred_coord_array(2, 2)

    (array([[-2],\n
        [-1],\n
        [ 0],\n
        [ 1],\n
        [ 2]]), array([[-2, -1,  0,  1,  2]]))

    """

    if M % 2 == 1:
        s1 = np.s_[-int(M / 2) : int(M / 2) + 1]
    else:
        s1 = np.s_[-int(M / 2) : int(M / 2)]

    if N % 2 == 1:
        s2 = np.s_[-int(N / 2) : int(N / 2) + 1]
    else:
        s2 = np.s_[-int(N / 2) : int(N / 2)]

    YC, XC = np.ogrid[s1, s2]

    return YC, XC


def rapsd(
    field: np.ndarray,
    fft_method: Any | None = None,
    return_freq: bool = False,
    d: float = 1.0,
    normalize: bool = False,
    **fft_kwargs: Any,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Compute radially averaged power spectral density (RAPSD) from the given
    2D input field.
    Taken from the pysteps package.

    Args:
        field (ndarray): A 2d array of shape (m, n) containing the input field.
        fft_method (Callable): A module or object implementing the same methods as numpy.fft and
            scipy.fftpack. If set to None, field is assumed to represent the
            shifted discrete Fourier transform of the input field, where the
            origin is at the center of the array
            (see numpy.fft.fftshift or scipy.fftpack.fftshift).
        return_freq (bool): Whether to also return the Fourier frequencies.
        d (scalar): Sample spacing (inverse of the sampling rate). Defaults to 1.
            Applicable if return_freq is 'True'.
        normalize (bool): If True, normalize the power spectrum so that it sums to one.
            Whether to also return the Fourier frequencies.

    Returns:
        out (ndarray): One-dimensional array containing the RAPSD. The length of the array is
            int(l/2) (if l is even) or int(l/2)+1 (if l is odd), where l=max(m,n).
        freq (ndarray): If return_freq is True, also returns the corresponding frequencies.
    """

    if len(field.shape) != 2:
        raise ValueError(
            f"{len(field.shape)} dimensions are found, but the number of dimensions should be 2"
        )

    if np.sum(np.isnan(field)) > 0:
        raise ValueError("input field should not contain nans")

    m, n = field.shape

    yc, xc = compute_centred_coord_array(m, n)
    r_grid = np.sqrt(xc * xc + yc * yc).round()
    side_len = max(field.shape[0], field.shape[1])

    if side_len % 2 == 1:
        r_range = np.arange(0, int(side_len / 2) + 1)
    else:
        r_range = np.arange(0, int(side_len / 2))

    if fft_method is not None:
        psd = fft_method.fftshift(fft_method.fft2(field, **fft_kwargs))
        psd = np.abs(psd) ** 2 / psd.size
    else:
        psd = field

    result = []
    for r in r_range:
        mask = r_grid == r
        psd_vals = psd[mask]
        result.append(np.mean(psd_vals))

    result = np.array(result)

    if normalize:
        result /= np.sum(result)

    if return_freq:
        freq = np.fft.fftfreq(side_len, d=d)
        freq = freq[r_range]
        return result, freq
    else:
        return result
