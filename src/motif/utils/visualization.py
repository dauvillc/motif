from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor

from motif.data.grid_functions import crop_nan_border
from motif.datatypes import Batch, GenerativePrediction, MultisourceTensor, Prediction


def display_realizations(
    pred: Prediction, batch: Batch, save_filepath_prefix: str | Path, display_fraction: float = 1.0
):
    """Given multiple solutions of the flow matching process, creates one figure
    per sample to display the solutions and groundtruth.
    Args:
        pred: The model's predictions, in a Prediction object that also contains the
            availability flags for each source.
        save_filepath_prefix: Prefix of the filepath where the figure will be saved.
            The figure will be saved as save_filepath_prefix + "_{sample_idx}.png".
        display_fraction (float, optional): Fraction of the samples to display. Defaults to 1.0.
    """
    save_filepath_prefix = Path(save_filepath_prefix)
    save_filepath_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Retrieve the output tensors and availability flags from the Prediction object
    sol: MultisourceTensor = pred.pred  # Dict of {src: (R, T, B, C, ...)} or {src: (B, C, ...)}
    avail_flags = pred.avail  # Dict of {src: (B,)}

    # Make the deterministic case compatible with the non-deterministic case by considering
    # the deterministic output a a single-realization generative output.
    deterministic = not isinstance(pred, GenerativePrediction)
    if deterministic:
        sol = {source_index_pair: sol_s.unsqueeze(0) for source_index_pair, sol_s in sol.items()}

    # Extract batch size and number of realizations
    any_source_index_pair = next(iter(sol.keys()))
    n_realizations = sol[any_source_index_pair].shape[0]
    batch_size = sol[any_source_index_pair].shape[1]

    # Take a fraction of the samples, evenly spaced
    samples_to_display = np.linspace(0, batch_size - 1, int(display_fraction * batch_size)).astype(
        int
    )

    # For each sample
    for sample_idx in samples_to_display:
        # Get available sources for this sample (either masked or available)
        source_index_pairs = [
            source_index_pair
            for source_index_pair, flags in avail_flags.items()
            if flags[sample_idx].item() != -1  # Either masked (0) or available (1)
        ]

        if not source_index_pairs:
            continue  # Skip if no sources are available or masked

        # Create a figure with n_realizations + 1 columns (realizations + groundtruth)
        # and one row per source
        fig, axs = plt.subplots(
            nrows=len(source_index_pairs),
            ncols=n_realizations + 1,
            figsize=(3 * (n_realizations + 1), 3 * len(source_index_pairs)),
            squeeze=False,
        )

        # For each source-index pair
        for src_idx, src in enumerate(source_index_pairs):
            source_name = src.name
            is_masked = avail_flags[src][sample_idx].item() == 0

            coords: Tensor = batch[src].coords[sample_idx]  # (2, H, W), lat/lon
            coords_no_nan = crop_nan_border(coords, [coords])[0]

            # For each realization
            for r_idx in range(n_realizations):
                ax = axs[src_idx, r_idx]
                # Get original coordinates
                coords = batch[src].coords[sample_idx]  # (2, H, W), lat/lon
                # Create a version of the coordinates without the nan borders
                coords_no_nan = crop_nan_border(coords, [coords])[0]

                # Only show prediction if the source was masked
                if is_masked:
                    # Get prediction data
                    pred_tensor = sol[src][r_idx, sample_idx, 0]

                    # Display data based on dimensionality
                    if len(pred_tensor.shape) == 2:
                        # The images' borders may be NaN due to the batching system.
                        # Crop the NaN borders to display the images correctly.
                        pred_tensor = crop_nan_border(coords, [pred_tensor.unsqueeze(0)])[
                            0
                        ].squeeze(0)
                        pred_tensor = pred_tensor.detach().cpu().float().numpy()
                        ax.imshow(pred_tensor, cmap="viridis")
                        # Add coords as axis labels
                        h, w = pred_tensor.shape
                        x_vals = np.nanmean(
                            coords_no_nan[1, :, :w].detach().cpu().float().numpy(), axis=0
                        )
                        y_vals = np.nanmean(
                            coords_no_nan[0, :h, :].detach().cpu().float().numpy(), axis=1
                        )
                        step_x = max(1, w // 5)
                        step_y = max(1, h // 5)
                        ax.set_xticks(range(0, w, step_x))
                        ax.set_yticks(range(0, h, step_y))
                        ax.set_xticklabels([f"{val:.2f}" for val in x_vals[0::step_x]], rotation=45)
                        ax.set_yticklabels([f"{val:.2f}" for val in y_vals[0::step_y]])
                    else:  # For 0D or 1D data
                        if len(pred_tensor.shape) == 1:
                            pred_tensor = pred_tensor[0]
                        pred_tensor = pred_tensor.item()
                        ax.bar([0], [pred_tensor], color="orange")

                    ax.set_title(f"{source_name} Pred {r_idx + 1}")
                else:
                    ax.set_title(f"{source_name}")
                    ax.axis("off")

            # Display groundtruth in the last column
            ax = axs[src_idx, -1]
            # Get groundtruth
            true = batch[src].values[sample_idx, 0]

            if len(true.shape) == 2:
                true = crop_nan_border(coords, [true.unsqueeze(0)])[0].squeeze(0)
                true = true.detach().cpu().float().numpy()
                ax.imshow(true, cmap="viridis")
                h, w = true.shape
                x_vals = np.nanmean(coords_no_nan[1, :, :w].detach().cpu().float().numpy(), axis=0)
                y_vals = np.nanmean(coords_no_nan[0, :h, :].detach().cpu().float().numpy(), axis=1)
                step_x = max(1, w // 5)
                step_y = max(1, h // 5)
                ax.set_xticks(range(0, w, step_x))
                ax.set_yticks(range(0, h, step_y))
                ax.set_xticklabels([f"{val:.2f}" for val in x_vals[0::step_x]], rotation=45)
                ax.set_yticklabels([f"{val:.2f}" for val in y_vals[0::step_y]])
            else:
                if len(true.shape) == 1:
                    true = true[0]
                true = true.item()
                ax.bar([0], [true], color="orange")

            dt = batch[src].dt[sample_idx].item()
            ax.set_title(f"{source_name} dt={dt:.3f} GT")

        plt.tight_layout()
        # Save figure
        save_path = f"{save_filepath_prefix}_{sample_idx}.png"
        plt.savefig(save_path)
        plt.close(fig)
