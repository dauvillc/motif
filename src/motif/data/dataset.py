"""Implements the MultiSourceDataset class."""

import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from netCDF4 import Dataset
from tqdm import tqdm

from motif.data.data_aug import MultisourceDataAugmentation
from motif.data.grid_functions import crop_nan_border
from motif.data.source import Source
from motif.data.utils import (
    compute_sources_availability,
    load_nc_with_nan,
)
from motif.datatypes import RawSourceData, SourceIndex


def precompute_samples(
    ref_df: pd.DataFrame,
    df: pd.DataFrame,
    dt_min: pd.Timedelta,
    dt_max: pd.Timedelta,
    verbose: bool = True,
) -> list[Tuple[pd.Timestamp, pd.DataFrame]]:
    """Precomputes the samples for a (subset of the) dataframe of
    reference samples.
    Written outside the MotifDataset class so that it can be
    called in parallel processes.

    Args:
        ref_df (pd.DataFrame): The reference dataframe.
        df (pd.DataFrame): DataFrame containing all samples.
        dt_min (int): The minimum time delta.
        dt_max (int): The maximum time delta.
        verbose (bool): Whether to print progress messages.

    Returns:
        list of tuples: List of pairs (t0, sample_df).
    """
    sample_dfs = []
    iterator = ref_df.iterrows()
    if verbose:
        iterator = tqdm(iterator, total=len(ref_df), desc="Precomputing samples")
    for _, row in iterator:
        sid, t0 = row["sid"], row["time"]
        sample_df = df[df["sid"] == sid]
        # Only keep the rows where the time is within the time window
        # defined by the reference time t0 and dt_min, dt_max
        min_t = t0 + dt_min
        max_t = t0 + dt_max
        time_mask = (sample_df["time"] <= max_t) & (sample_df["time"] > min_t)
        sample_df = sample_df[time_mask]

        # Sort the rows by ascending time difference to the reference time,
        # so that if we keep the first k rows in getitem, we keep the k closest observations.
        sample_df = sample_df.sort_values("time", key=lambda x: abs(x - t0))

        sample_dfs.append((t0, sample_df))
    return sample_dfs


class MultiSourceDataset(torch.utils.data.Dataset):
    """A dataset that yields elements from multiple sources.

    For a given storm S and reference time t0, the dataset will yield observations of S
    that fall within the time window [t0 + dt_min, t0 + dt_max].
    The observations come from multiple sources, and for each source, multiple
    observations can be included in a sample.

    A sample from the dataset is a dict {(source_name, index): data} where
    source_name is the name of the source and index is an integer representing
    the observation index (0 = most recent, 1 = second most recent, etc.).
    Each data dict contains the following key-value pairs:
    - "dt" is a scalar tensor of shape (1,) containing the time delta between the reference time
        and the observation time, normalized by the length of the time window.
    - "characs" is a tensor of shape (n_charac_vars,) containing the source characteristics.
        Each data variable within a source has its own charac variables, which are all
        concatenated into a single tensor.
    - "coords" is a tensor of shape (2, H, W) containing the latitude and longitude at each pixel.
    - "landmask" is a tensor of shape (H, W) containing the land mask.
    - "dist_to_center" is a tensor of shape (H, W) containing the distance
        to the center of the storm.
    - "values" is a tensor of shape (n_variables, H, W) containing the variables for the source.
    """

    def __init__(
        self,
        dataset_dir: str,
        split: str,
        included_variables_dict: dict,
        dt_min: float,
        dt_max: float,
        dt_min_norm: float | None = None,
        dt_max_norm: float | None = None,
        groups_availability: dict = {},
        select_closest: bool = False,
        min_ref_time_delta: float = 0,
        num_workers: int = 0,
        data_augmentation: MultisourceDataAugmentation | None = None,
        limit_samples: int | float | None = None,
        seed: int = 42,
        select_most_recent: bool = True,
    ):
        """
        Args:
            dataset_dir (str): The directory containing the preprocessed dataset.
                The directory should contain the following:
                - train/val/test.json: pandas DataFrame containing the samples metadata.
                - processed/
                    - source_name/
                        - source_metadata.json: dictionary containing the metadata for the source.
                        - samples_metadata.json: dictionary containing the metadata for
                            the samples.
                        - *.nc: netCDF files containing the data for the source.
                - constants/
                    - source_name/
                        - data_means/stds.json: dictionaries containing the means and stds
                            for the data variables of the source.
                        - charac_vars_min_max.json: dictionaries {charac_var_name: {min, max}}
            split (str): The split of the dataset. Must be one of "train", "val", or "test".
            included_variables_dict (dict): A dictionary
                {source_name: (list of variables, list of input-only variables)}
                containing the variables (i.e. channels) to include for each source.
                Variables also included in the second list will be included in the yielded
                data but will be flagged as input-only in the Source object.
            dt_min (float): The minimum time delta (in hours) between the reference time
                and the observation time.
            dt_max (float): The maximum time delta (in hours) between the reference time
                and the observation time.
            dt_min_norm (float or None): The minimum time delta (in hours) used for
                normalizing the time delta. If None, dt_min is used.
            dt_max_norm (float or None): The maximum time delta (in hours) used for
                normalizing the time delta. If None, dt_max is used.
            groups_availability (Dict of str to dict): Dict defining groups of sources.
                For each group, a minimum and a maximum number of sources from that group
                can be set, which will be used to filter the samples.
                The root keys are group names, and serve no purpose beyond clarity.
                The structure is as follows:
                {
                    group_name_1: {
                        sources: [list of source names],
                        availability: (min_available, max_available)
                    },
                    group_name_2: ...
                }
                For example, {"g1": {sources: ["S1", "S2"], availability: (1, 3)}} means that
                from the group g1 ["S1", "S2"] must be available in each sample, and at most
                3 sources from that group will be included in each sample.
                Note 1: The maximum number can be set to None to have no limit.
                Note 2: Instead of writing the full source names, one can also use source types
                    to include all sources of a given type.
            select_closest (bool): If True, keeps the source that is closest to the reference time
                when selecting only a subset of sources within a group.
            min_ref_time_delta (float): The minimum time delta between two reference times
                of the same storm (ie between two samples of the same storm).
            num_workers (int): If > 1, number of workers to use for parallel loading of the data.
            data_augmentation (None or MultiSourceDataAugmentation): If not None, instance
                of MultiSourceDataAugmentation to apply to the data.
            limit_samples (int or float or None): If not None, limits the number of samples in
                the dataset.
                If an integer, will keep at most limit_samples samples.
                If a float between 0 and 1, will keep that fraction of the samples.
            seed (int): The seed to use for the random number generator.
            select_most_recent (bool): Same as select_closest, here for backwards compatibility.
        """

        self.dataset_dir = Path(dataset_dir)
        self.constants_dir = self.dataset_dir / "constants"
        self.processed_dir = self.dataset_dir / "prepared"
        self.split = split
        self.groups_availability = groups_availability
        self.min_ref_time_delta = min_ref_time_delta
        self.data_augmentation = data_augmentation
        self.rng = np.random.default_rng(seed)
        self.select_closest = select_closest or select_most_recent

        print(f"{split}: Browsing requested sources and loading metadata...")
        # Load and merge individual source metadata files
        sources_metadata = {}
        for source_name in included_variables_dict:
            source_metadata_path = self.processed_dir / source_name / "source_metadata.json"
            # If the source metadata file is not found, print a warning
            # and skip the source
            if not source_metadata_path.exists():
                print(f"Warning: {source_metadata_path} not found. Skipping source {source_name}.")
                continue
            with open(source_metadata_path, "r") as f:
                sources_metadata[source_name] = json.load(f)
        # If no source has been found, raise an error
        if len(sources_metadata) == 0:
            raise ValueError("Did not find any source metadata files.")
        # Filter the included variables for each source
        self.variables_dict = {
            source: included_variables_dict[source] for source in sources_metadata
        }

        self.sources: List[Source] = []
        self.sources_dict: Dict[str, Source] = {}
        for source_name, (all_vars, input_only, output_only) in self.variables_dict.items():
            # Update the source metadata with the included variables
            sources_metadata[source_name]["data_vars"] = all_vars
            sources_metadata[source_name]["input_only_vars"] = input_only
            sources_metadata[source_name]["output_only_vars"] = output_only
            # Create the source object
            self.sources.append(Source(**sources_metadata[source_name]))
            self.sources_dict[source_name] = self.sources[-1]
        # Load the samples metadata based on the split
        self.df = pd.read_csv(self.dataset_dir / f"{split}.csv", parse_dates=["time"])

        # Filter the dataframe to only keep the rows where source_name is the name of a source
        # in self.sources
        source_names = [source.name for source in self.sources]
        self.source_variables = {source.name: source.data_vars for source in self.sources}
        self.source_types = list(set([source.type for source in self.sources]))
        self.df = self.df[self.df["source_name"].isin(source_names)]
        self.df = self.df.reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError("No samples found for the given sources.")

        # Time delta settings
        self.dt_min = pd.Timedelta(dt_min, unit="h")
        self.dt_max = pd.Timedelta(dt_max, unit="h")
        if dt_min_norm is None:
            self.dt_min_norm = self.dt_min
        else:
            self.dt_min_norm = pd.Timedelta(dt_min_norm, unit="h")
        if dt_max_norm is None:
            self.dt_max_norm = self.dt_max
        else:
            self.dt_max_norm = pd.Timedelta(dt_max_norm, unit="h")
        self.dt_norm = self.dt_max_norm - self.dt_min_norm

        # ========================================================================================
        # BUILDING THE REFERENCE DATAFRAME
        # - We'll compute a dataframe D of shape (n_samples, n_sources) such that
        # D[i, s] = 1 if source i is available for sample s, and 0 otherwise.
        print(f"{split}: Computing sources availability...")
        available_sources = compute_sources_availability(
            self.df, self.dt_min, self.dt_max, num_workers=num_workers
        )
        # We can now build the reference dataframe:
        # - self.df contains the metadata for all elements from all sources,
        # - self.reference_df is a subset of self.df containing every (sid, time) pair
        #   that defines a sample which matches the availability criteria.
        # The criteria will fill a list of boolean masks, which will be combined
        # with a logical AND to filter the dataframe.
        masks = []
        # We'll begin by counting which sources are available for each sample.
        available_sources_count = available_sources.sum(axis=1)
        # Group availability criteria: for each group of sources, only keep the samples
        # where at least min_avail sources from the group are available.
        # Note: the max_avail criterion will be applied later, in __getitem__.
        # First, the user can specify source names, but they can also specify source types.
        # We need to convert the source types to source names.
        for group_name, group_avail_dict in self.groups_availability.items():
            group_sources = []
            for s in group_avail_dict["sources"]:
                if s in source_names:
                    group_sources.append(s)
                elif s in self.source_types:
                    # Add all sources of that type
                    type_sources = [source.name for source in self.sources if source.type == s]
                    group_sources.extend(type_sources)
                # If the source/type is not found, print a warning, but it could be that the
                # source is simply not in the dataset split, so we don't raise an error.
                else:
                    print(f"Warning: source or source type {s} not found among dataset sources.")
            group_avail_dict["sources"] = group_sources
        # Now apply the min_avail criterion for each group
        for group_avail_dict in self.groups_availability.values():
            # If some sources are never available, they won't be in available_sources.columns.
            min_avail = group_avail_dict["availability"][0]  # Minimum number of sources
            group = group_avail_dict["sources"]  # The sources in the group
            group = [s for s in group if s in available_sources.columns]
            group_avail_count = available_sources[group].sum(axis=1)
            masks.append(group_avail_count >= min_avail)
        # Combine the masks with a logical AND
        mask = np.logical_and.reduce(masks)
        # We can now build the reference dataframe:
        self.reference_df = self.df[mask][["sid", "time", "source_name", "intensity"]]
        self.reference_df["n_available_sources"] = available_sources_count[mask]

        # Avoid duplicated samples if two observations are at the exact same time
        self.reference_df = self.reference_df.drop_duplicates(["sid", "time"])
        self.reference_df = self.reference_df.reset_index(drop=True)

        # If required, sample the reference dataframe to keep only one sample
        # every min_ref_time_delta hours.
        if self.min_ref_time_delta > 0:
            # Assign each row to a time bin
            self.reference_df["time_bin"] = self.reference_df["time"].dt.floor(
                f"{self.min_ref_time_delta}H"
            )
            # Sort to ensure the first occurrence is the earliest in each bin
            self.reference_df = self.reference_df.sort_values(["sid", "time"])
            # Drop duplicates to keep the first occurrence in each bin per sid
            self.reference_df = self.reference_df.drop_duplicates(subset=["sid", "time_bin"])
            # Drop the 'time_bin' column as it's no longer needed
            self.reference_df = self.reference_df.drop(columns="time_bin").reset_index(drop=True)

        # If required, limit the number of samples in the dataset
        if limit_samples is not None:
            if isinstance(limit_samples, float):
                if not (0 < limit_samples < 1):
                    raise ValueError("If limit_samples is a float, it must be in (0, 1).")
                limit_samples = int(len(self.reference_df) * limit_samples)
            indices = self.rng.permutation(len(self.reference_df))[:limit_samples]
            self.reference_df = self.reference_df.iloc[indices].reset_index(drop=True)
            print(f"{split}: Limiting the dataset to {len(self.reference_df)} samples.")

        # If no samples are left, raise an error
        if len(self.reference_df) == 0:
            raise ValueError(f"No samples left after filtering for split {split}.")

        # ========================================================================================
        # Pre-computation to speed up the data loading: instead of isolating the rows
        # of self.df in __getitem__, we'll pre-compute them now.
        # We'll obtain a list [sample_df_1, sample_df_2, ...] where each element corresponds to a
        # a row of self.reference_df (i.e. an item in self.__getitem__).
        print(f"{split}: Pre-computing the samples...")
        if num_workers <= 1:
            self.samples = precompute_samples(
                self.reference_df,
                self.df,
                self.dt_min,
                self.dt_max,
            )
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Divide the full reference df in num_workers chunks
                chunks = np.array_split(self.reference_df, num_workers)
                results = executor.map(
                    precompute_samples,
                    chunks,
                    [self.df] * len(chunks),
                    [self.dt_min] * len(chunks),
                    [self.dt_max] * len(chunks),
                    [True] + [False] * (len(chunks) - 1),  # only the first chunk is verbose
                )
                self.samples = [item for sublist in results for item in sublist]
        # Sort the samples by reference time
        self.samples = sorted(self.samples, key=lambda x: x[0])

        # ========================================================================================
        # Load the data means and stds
        self.data_means: Dict[str, Dict[str, float]] = {}
        self.data_stds: Dict[str, Dict[str, float]] = {}
        for source in self.sources:
            # Data vars means and stds
            with open(self.constants_dir / source.name / "data_means.json", "r") as f:
                self.data_means[source.name] = json.load(f)
            with open(self.constants_dir / source.name / "data_stds.json", "r") as f:
                self.data_stds[source.name] = json.load(f)

        # Characteristic variables: pre-build the tensors of characteristics for each source,
        # since those do not depend on the sample.
        self.charac_vars_tensors = {}
        charac_vars_min, charac_vars_max = defaultdict(list), defaultdict(list)
        for source in self.sources:
            charac_vars = source.get_charac_values()
            self.charac_vars_tensors[source.name] = torch.tensor(charac_vars, dtype=torch.float32)
            # Also pre-load the min and max of the charac variables for the source. We'll fill
            # the self.charac_vars_min and self.charac_vars_max lists with the min and max
            # of the source's charac variables, in the same order as in the tensors we just built;
            # this way we can easily normalize the charac variables.
            with open(self.constants_dir / source.name / "charac_vars_min_max.json", "r") as f:
                source_characs_min_max = json.load(f)
            # iter_charac_variables() yields the items in the same order as get_charac_values().
            for charac_var_name, data_var_name, _ in source.iter_charac_variables():
                charac_vars_min[source.name].append(source_characs_min_max[charac_var_name]["min"])
                charac_vars_max[source.name].append(source_characs_min_max[charac_var_name]["max"])
        # convert to tensors
        self.charac_vars_min: Dict[str, torch.Tensor] = {
            source_name: torch.tensor(mins, dtype=torch.float32)
            for source_name, mins in charac_vars_min.items()
        }
        self.charac_vars_max: Dict[str, torch.Tensor] = {
            source_name: torch.tensor(maxs, dtype=torch.float32)
            for source_name, maxs in charac_vars_max.items()
        }

    def __getitem__(self, idx: int) -> Tuple[int, Dict[SourceIndex, RawSourceData]]:
        """Returns the element at the given index.

        Args:
            idx (int): The index of the element to retrieve.

        Returns:
            idx (int): The index of the element.
            sample (dict): A dictionary of the form {(source_name, index): data}
                where index is an integer representing the observation index
                (0 = most recent, 1 = second most recent, etc.)
        """
        t0, sample_df = self.samples[idx]

        # If selecting the sources randomly, shuffle the rows of the sample dataframe
        # so that keeping the first k of a group keeps a random subset of it.
        if not self.select_closest:
            sample_df = sample_df.sample(frac=1, random_state=int(self.rng.integers(0, 1_000_000)))

        # Group availability criterion: if there are too many sources from a given group,
        # we only keep the maximum allowed number of sources from that group.
        # (Random if select_closest is False due to above)
        for group_avail_dict in self.groups_availability.values():
            max_avail = group_avail_dict["availability"][1]  # How many sources to keep
            group = group_avail_dict["sources"]  # The sources in the group
            if max_avail is not None:
                group_mask = sample_df["source_name"].isin(group)
                group_samples = sample_df[group_mask]
                if len(group_samples) > max_avail:
                    # Keep only the first max_avail samples from the group
                    to_drop = group_samples.index[max_avail:]
                    sample_df = sample_df.drop(index=to_drop)

        # Sort the sample dataframe by time, most recent first
        sample_df = sample_df.sort_values("time", ascending=False)

        # For each source, try to load the element at the given time
        output = {}
        sources_cnt = defaultdict(int)  # {source_name: number of obs from that source added}

        for _, row in sample_df.iterrows():
            source_name = row["source_name"]
            source = self.sources_dict[source_name]

            # Retrieve the time delta and normalize it
            time = row["time"]
            dt = time - t0
            DT = torch.tensor(
                (dt - self.dt_min_norm).total_seconds() / self.dt_norm.total_seconds(),
                dtype=torch.float32,
            )

            # Load the npy file containing the data for the given storm, time, and source
            filepath = Path(row["data_path"])
            with Dataset(filepath) as ds:
                # Coordinates
                lat = load_nc_with_nan(ds["latitude"])
                lon = load_nc_with_nan(ds["longitude"])
                # Make sure the longitude is in the range [-180, 180]
                lon = np.where(lon > 180, lon - 360, lon)
                C = np.stack([lat, lon], axis=0)
                C = torch.tensor(C, dtype=torch.float32)
                # Load the land mask and distance to the center of the storm
                LM = torch.tensor(load_nc_with_nan(ds["land_mask"]), dtype=torch.float32)
                D = torch.tensor(load_nc_with_nan(ds["dist_to_center"]), dtype=torch.float32)

                # Characterstic variables
                if source.n_charac_variables() == 0:
                    CA = None
                else:
                    CA = torch.tensor(source.get_charac_values(), dtype=torch.float32)

                # Values
                # Load the variables in the order specified in the source
                V = np.stack([load_nc_with_nan(ds[var]) for var in source.data_vars], axis=0)
                V = torch.tensor(V, dtype=torch.float32)

                # Availability mask: 1 if value is not NaN, -1 otherwise
                # (We keep 0 to indicate masked values).
                AM = (~torch.isnan(V)[0]).float() * 2 - 1

                # Normalize the characs and values tensors
                CA, V = self.normalize(V, source, characs=CA)

                if source.dim == 2:
                    # For images only.
                    # The values can contain borders that are fully NaN (due to the sources
                    # originally having multiple channels that are not aligned geographically).
                    # Compute how much we can crop them, and apply that cropping to all spatial
                    # tensors to keep the spatial alignment.
                    V, C, LM, D, AM = crop_nan_border(V, [V, C, LM, D, AM])

                # Assemble the output dict for that source
                output_source = RawSourceData(
                    dt=DT,
                    coords=C,
                    landmask=LM,
                    dist_to_center=D,
                    values=V,
                    avail_mask=AM,
                    characs=CA,  # May be None
                )

                # Add the processed observation to the list for this source
                output[SourceIndex(source_name, sources_cnt[source_name])] = output_source
                sources_cnt[source_name] += 1

        if len(output) <= 1:
            sid, t0 = self.reference_df.iloc[idx][["sid", "time"]]
            print(sample_df)
            print(f"t0 = {t0}, sid = {sid}")
            raise ValueError(f"Sample {sid} at time {t0} has only {len(output)} sources available.")

        # Apply data augmentation if required
        if self.data_augmentation is not None:
            output = self.data_augmentation(output)
        return idx, output

    def normalize(
        self,
        values: torch.Tensor,
        source: Source | str,
        characs: torch.Tensor | None = None,
        denormalize: bool = False,
        leading_dims: int = 0,
        device: torch.device | None = None,
    ) -> Tuple[torch.Tensor | None, torch.Tensor]:
        """Normalizes the characs and values tensors associated with a given
        source, and optionally a specific data variable.
        Args:
            values (torch.Tensor): tensor of shape (C, ...) if leading_dims=0,
                or (D1, D2, ..., Dn, C, ...) for leading_dims=n, where ... are the spatial dimensions.
            source (Source or str): Source object representing the source, or name
                of the source.
            characs (torch.Tensor, optional): tensor of shape (n_charac_vars,)
                containing the characteristic variables.
            denormalize (bool, optional): If True, denormalize the characs and values tensors
                instead of normalizing them.
            leading_dims (int, optional): Number of leading dimensions before the channel dimension.
                For example, if leading_dims=0, the tensor has shape (C, ...),
                if leading_dims=1, the tensor has shape (D1, C, ...),
                if leading_dims=2, the tensor has shape (D1, D2, C, ...), etc.
            device (torch.device, optional): Device to use for the normalization.
        Returns:
            normalized_characs: normalized charac variables, or None if characs is None.
            normalized_values: normalized values.
        """
        if isinstance(source, Source):
            source_name = source.name
        else:
            source_name = source
            source = self.sources_dict[source_name]

        # Characteristic variables normalization.
        if characs is not None:  # Values normalization.
            characs = characs.to(torch.float32)
            min, max = self.charac_vars_min[source_name], self.charac_vars_max[source_name]
            if device is not None:
                min = min.to(device)
                max = max.to(device)
            if denormalize:
                normalized_characs = characs * (max - min) + min
            else:
                normalized_characs = (characs - min) / (max - min)
        else:
            normalized_characs = None

        data_vars = source.data_vars

        # Create shape with leading 1s for each leading dimension, followed by channel dimension
        # and trailing 1s for spatial dimensions
        shape = [1] * leading_dims + [-1] + [1] * (len(values.shape) - leading_dims - 1)

        data_means = np.array([self.data_means[source_name][var] for var in data_vars]).reshape(
            shape
        )
        data_stds = np.array([self.data_stds[source_name][var] for var in data_vars]).reshape(shape)
        data_means = torch.tensor(data_means, dtype=values.dtype)
        data_stds = torch.tensor(data_stds, dtype=values.dtype)

        if device is not None:
            data_means = data_means.to(device)
            data_stds = data_stds.to(device)
        if denormalize:
            normalized_values = values * data_stds + data_means
        else:
            normalized_values = (
                values - data_means
            ) / data_stds  # Fixed: was dividing by data_means

        return normalized_characs, normalized_values

    def __len__(self) -> int:
        return len(self.reference_df)

    def _get_source_names(self) -> List[str]:
        """Returns a list of the source names (before splitting the sources)."""
        return [source.name for source in self.sources]

    def _get_n_sources(self) -> int:
        """Returns the number of original sources (before splitting the sources)."""
        return len(self.sources)

    def get_n_sources(self) -> int:
        """Returns the number of sources."""
        return len(self.sources)

    def get_source_types_charac_vars(self) -> Dict[str, List[str]]:
        """Returns a dict {source_type: charac variables}."""
        # Browse all sources and collect their types and charac variables
        source_types_charac_vars = {}
        for source in self.sources:
            # If the source type has been seen before, make sure the charac
            # vars were the same. All sources from the same type should have
            # the same charac variables.
            if source.type in source_types_charac_vars:
                if source.charac_vars != source_types_charac_vars[source.type]:
                    raise ValueError(
                        f"Sources of type {source.type} have different charac variables."
                    )
            else:
                source_types_charac_vars[source.type] = source.charac_vars
        return source_types_charac_vars

    def get_n_charac_variables(self) -> Dict[str, int]:
        """Returns a dict {source_name: number of charac variables}."""
        return {source.name: source.n_charac_variables() for source in self.sources}
