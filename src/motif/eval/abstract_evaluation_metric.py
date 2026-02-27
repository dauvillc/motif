"""
Implements the AbstractEvaluationMetric class, which is a base class for all evaluation metrics.
"""

import abc
import re
from pathlib import Path
from typing import Generator, cast

import pandas as pd
import xarray as xr

from motif.datatypes import SourceIndex


def models_info_sanity_check(info_dfs: list[pd.DataFrame], strictness: str = "strict"):
    """Performs a sanity check on the predictions info dataframes, to
    ensure they were made on exactly the same data.
    Args:
        info_dfs (list of pandas.DataFrame): List of info dataframes, one for each model.
        strictness (str): "strict", "targets_only" or "none".
            Whether to perform strict checks on the info dataframes.
            If "targets_only", only checks that the target sources correspond and that there
            are no duplicates in the (sample_index, source_name, source_index) triples.
    """
    # Check that all info_dfs have the same columns
    if not all(info_df.columns.equals(info_dfs[0].columns) for info_df in info_dfs):
        raise ValueError(
            "Found models with different columns in their info_df. "
            "Please ensure all models are evaluated on the same data."
        )
    # Check that are no duplicates in the (sample_index, source_name, source_index) triples
    for info_df in info_dfs:
        if info_df.duplicated(subset=["sample_index", "source_name", "source_index"]).any():
            raise ValueError(
                "Found duplicates in the (sample_index, source_name, source_index) triples "
                "of the info_df. There may be a bug, as a a pair (src_name, src_index) should "
                "be unique within a sample."
            )

    if strictness == "strict":
        # Check that they have the same number of rows (samples and sources)
        if not all(len(info_df) == len(info_dfs[0]) for info_df in info_dfs):
            raise ValueError(
                "Found models with different number of rows in their info_df. "
                "This indicates different samples or different sources. "
            )
        # Check that you the sample_index, source_name, source_index columns are the same
        for col in ["sample_index", "source_name", "source_index"]:
            if not all(info_df[col].equals(info_dfs[0][col]) for info_df in info_dfs):
                raise ValueError(
                    f"Found models with different values in the {col} column of their info_df. "
                    "Please ensure all models are evaluated on the same data."
                )
        # Check the availability flags match
        for i in range(1, len(info_dfs)):
            if not info_dfs[i]["avail"].equals(info_dfs[0]["avail"]):
                raise ValueError(
                    "Found models with different availability flags in their info_df. "
                    "Make sure the models' masking selection is identical between"
                    "the prediction runs."
                )
    else:
        # Same checks, but only for the rows concerning the target sources (avail == 0)
        target_info_dfs = [
            info_df[info_df["avail"] == 0].reset_index(drop=True) for info_df in info_dfs
        ]
        if not all(len(info_df) == len(target_info_dfs[0]) for info_df in target_info_dfs):
            raise ValueError("Found different number of target rows in the info_dfs. ")
        for col in ["sample_index", "source_name", "source_index"]:
            if not all(info_df[col].equals(target_info_dfs[0][col]) for info_df in target_info_dfs):
                raise ValueError(
                    f"Found models with different values in the {col} column of their info_df "
                    "for the target rows."
                )


class AbstractMultisourceEvaluationMetric(abc.ABC):
    """Base class for all evaluation metrics."""

    def __init__(
        self,
        id_name: str,
        full_name: str,
        model_data: dict[str, dict],
        parent_results_dir: str | Path,
        source_name_replacements: list[tuple[str, str]] | None = None,
        channel_replacements: list[tuple[str, str]] | None = None,
        checks_strictness: str = "strict",
        num_workers: int = 1,
    ):
        """
        Args:
            id_name (str): Unique identifier for the metric. Must follow the
                rules of file naming.
            full_name (str): Full name of the metric, for display purposes.
            model_data (dict): Dictionary mapping model_ids to dictionaries containing:
                - info_df: DataFrame with metadata
                - root_dir: Path to predictions directory
                - results_dir: Path to results directory
                - run_id: Run ID
                - pred_name: Prediction name
            parent_results_dir (str or Path): Parent directory for all results
            source_name_replacements (List of tuple of str, optional): List of (pattern, replacement)
                substitutions to apply to source names for display purposes. The replacement
                is done using the re.sub function.
            channel_replacements (List of tuple of str, optional): List of (pattern, replacement)
                substitutions to apply to channel names for display purposes. The replacement
                is done using the re.sub function.
            checks_strictness (str): "strict", "targets_only" or "none".
                Whether to perform strict checks on the info dataframes of the models.
            num_workers (int): Here for compatibility.
        """
        self.id_name = id_name
        self.full_name = full_name
        self.model_data = model_data
        self.parent_results_dir = Path(parent_results_dir)
        self.source_name_replacements = source_name_replacements or []
        self.channel_replacements = channel_replacements or []

        # Create a directory for this evaluation metric
        self.metric_results_dir = self.parent_results_dir / id_name
        self.metric_results_dir.mkdir(parents=True, exist_ok=True)

        # Sort the info dataframes by sample_index, source_name, source_index
        info_dfs: list[pd.DataFrame] = [model_spec["info_df"] for model_spec in model_data.values()]
        for info_df in info_dfs:
            info_df.sort_values(by=["sample_index", "source_name", "source_index"], inplace=True)
            info_df.reset_index(drop=True, inplace=True)
        # Perform sanity checks on the model data
        if checks_strictness != "none":
            models_info_sanity_check(info_dfs, strictness=checks_strictness)

        # Isolate a model-agnostic DataFrame with the columns that don't depend on the model:
        # sample_index, source_name, source_index, avail, dt
        self.samples_df = (
            info_dfs[0][["sample_index", "source_name", "source_index", "avail", "dt"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        self.n_samples = self.samples_df["sample_index"].nunique()

    def samples_iterator(
        self, include_intermediate_steps: bool = False
    ) -> Generator[
        tuple[
            pd.DataFrame, dict[SourceIndex, xr.Dataset], dict[str, dict[SourceIndex, xr.Dataset]]
        ],
        None,
        None,
    ]:
        """Iterator over the samples in the evaluation.
        Args:
            include_intermediate_steps (bool): If True, includes the intermediate steps
                of the ODE solver for models that return them.
                In True, the predictions will have an additional leading dimension
                "integration_step" corresponding to the time steps of the ODE solver.
        Yields:
            sample_df (pandas.DataFrame): A DataFrame with the columns:
                - sample_index: Index of the sample (same for all rows)
                - source_name: Name of the source
                - source_index: Index of the source
                - avail: Availability flag (1 for available, 0 for target)
                - dt: Timestamp of the sample
                only the data for a single sample (sample_index) is yielded at a time.
            targets (dict): Dict (source_name, source_index) -> xarray.Dataset
                The targets are the same for all models.
            predictions (dict): Dict model_id -> Dict (source_name, source_index) -> xarray.Dataset
        The missing (source_name, source_index) pairs, i.e. those with an availability flag
        of -1, are removed from both the DataFrame and xarray datasets.
        """
        for sample_index in self.samples_df["sample_index"].unique():
            targets, predictions = self.load_data(sample_index)
            sample_df = self.samples_df[self.samples_df["sample_index"] == sample_index]
            sample_df = sample_df.set_index(["source_name", "source_index"])

            targets_dict = {}
            preds_dict = {model_id: {} for model_id in self.model_data}
            for src_tuple in sample_df.index:
                src = SourceIndex(name=src_tuple[0], index=src_tuple[1])
                # Get the availability flag to know whether this src is available
                avail = sample_df.loc[src_tuple, "avail"]
                if avail == -1:
                    continue
                targets_dict[src] = targets[src]
                for model_id in self.model_data:
                    if src not in predictions[model_id]:
                        # A source might included in the inputs/outputs of one the models
                        # but not in the others.
                        continue
                    # If the predictions include intermediate steps, keep only the last one
                    model_preds = predictions[model_id][src]
                    if not include_intermediate_steps and "integration_step" in model_preds.dims:
                        model_preds = model_preds.isel(integration_step=-1)
                    preds_dict[model_id][src] = model_preds

            # Remove the rows from the DataFrame that are not available
            sample_df = sample_df[sample_df["avail"] != -1]
            yield sample_df, targets_dict, preds_dict

    def load_data(
        self, sample_index: int
    ) -> tuple[dict[SourceIndex, xr.Dataset], dict[str, dict[SourceIndex, xr.Dataset]]]:
        """Loads the data for all models as xarray datasets stored in individual
        netCDF4 files.
        Args:
            sample_index (int): Index of the sample to load.
        Returns:
            targets (dict): Dict (src_name, src_index) -> xarray.Dataset
               The targets are the same for all models.
            predictions (dict): Dict model_id -> (src_name, src_index) -> xarray.Dataset
        """
        targets, predictions = {}, {}
        for i, (model_id, model_spec) in enumerate(self.model_data.items()):
            predictions[model_id] = {}
            info_df = model_spec["info_df"]
            info_df = info_df[info_df["sample_index"] == sample_index]
            root_dir = model_spec["root_dir"]

            # Isolate the unique pairs (source_name, source_index) for this model
            source_pairs = info_df[["source_name", "source_index"]].drop_duplicates()
            for _, row in source_pairs.iterrows():
                src_name, index = row["source_name"], row["source_index"]
                src = SourceIndex(name=src_name, index=index)
                # Load targets for the first model only (since they are the same for all)
                if i == 0:
                    target_path = (
                        root_dir / "targets" / src_name / str(index) / f"{sample_index}.nc"
                    )
                    targets_ds = xr.open_dataset(target_path)
                    targets[src] = apply_channel_name_replacements(
                        targets_ds, self.channel_replacements
                    )
                # Load predictions for all models
                pred_path = root_dir / "predictions" / src_name / str(index) / f"{sample_index}.nc"
                predictions_ds = xr.open_dataset(pred_path)
                predictions[model_id][src] = apply_channel_name_replacements(
                    predictions_ds, self.channel_replacements
                )

        return targets, predictions

    @abc.abstractmethod
    def evaluate(self, **kwargs):
        """
        Main evaluation method.

        Args:
            **kwargs: Additional keyword arguments including:
                - num_workers: Number of workers for parallel processing
        """
        pass

    def _display_src_name(self, src_name: str) -> str:
        """Applies the source name replacements to a source name for display purposes."""
        for pattern, replacement in self.source_name_replacements:
            src_name = re.sub(pattern, replacement, src_name)
        return src_name


def apply_channel_name_replacements(
    ds: xr.Dataset, channel_replacements: list[tuple[str, str]]
) -> xr.Dataset:
    """Applies the channel name replacements to the variables of an xarray dataset."""
    if not channel_replacements:
        return ds
    # For each channel, apply the replacements to its name and rename it
    # in the dataset
    rename_dict = {}
    for var in ds.data_vars:
        new_var = cast(str, var)
        for pattern, replacement in channel_replacements:
            new_var = re.sub(pattern, replacement, new_var)
        if new_var != var:
            rename_dict[var] = new_var
    if rename_dict:
        ds = ds.rename_vars(rename_dict)
    return ds
