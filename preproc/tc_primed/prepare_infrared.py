#!/usr/bin/env python3

"""
Prepares the infrared data for the TC Primed dataset.
"""

import json
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import xarray as xr
from global_land_mask import globe
from netCDF4 import Dataset
from omegaconf import OmegaConf
from tqdm import tqdm
from xarray.backends import NetCDF4DataStore

from motif.data.grid_functions import grid_distance_to_point

# Local imports
from preproc.tc_primed.utils import list_tc_primed_sources

# Which value of infrared_availability_flag indicates which source
# in TC-PRIMED. 0 means unavailable.
TCIRAR_OR_HURSAT = [None, "tcirar", "hursat"]


def initialize_metadata(dest_dir):
    """Initializes the metadata for either TCIRAR or HURSAT infrared source.
    Args:
        dest_dir (Path): Destination directory.
    Returns:
        bool: True if the metadata was successfully written, False otherwise.
    """

    # Variables that give information about the storm of the sample.
    storm_vars = ["storm_latitude", "storm_longitude", "intensity"]
    # Dict that contains the source's metadata.
    source_metadata = {
        "source_name": f"tc_primed_ir_{dest_dir.name.split('_')[-1]}",
        "source_type": "infrared",
        "dim": 2,
        "data_vars": ["IRWIN"],
        "storm_metadata_vars": storm_vars,
        "is_on_regular_grid": True,
        "charac_vars": {},
    }
    # Characteristic variable: only the resolution changes
    # between TCIRAR and HURSAT.
    if "tc_primed_ir_tcirar" in dest_dir.name:
        source_metadata["charac_vars"]["resolution_km"] = {"IRWIN": 4.0}
    elif "tc_primed_ir_hursat" in dest_dir.name:
        source_metadata["charac_vars"]["resolution_km"] = {"IRWIN": 8.0}
    with open(dest_dir / "source_metadata.json", "w") as f:
        json.dump(source_metadata, f)
    return True


def process_ir_file(file, dest_dirs, check_exist=False):
    """Processes a single infrared file.s
    Args:
        file (str): Path to the input file.
        dest_dirs (dict): Destination directories for TCIRAR and HURSAT sources.
        check_exist (bool): Flag to check if the file already exists.
    Returns:
        dict or None: Sample metadata, or None if the sample is discarded.
        tcirar_or_hursat (str): 'tcirar' or 'hursat' depending on the source
            that was used for this sample.
    """
    with Dataset(file) as raw_sample:
        # Access data group
        ds = raw_sample["infrared"]
        # Check the infrared availability flag. We only keep the samples
        # for which infrared data is available.
        tcirar_or_hursat = TCIRAR_OR_HURSAT[ds["infrared_availability_flag"][0].item()]
        if tcirar_or_hursat is None:
            return None, None  # Discard sample
        dest_dir = dest_dirs[tcirar_or_hursat]

        # Open the dataset using xarray and select the brightness temperature variable
        ds = xr.open_dataset(NetCDF4DataStore(ds), decode_times=False)
        ds = ds[["IRWIN", "latitude", "longitude"]]

        # Assemble the sample's metadata
        raw_overpass_meta = raw_sample["overpass_metadata"]
        season = raw_overpass_meta["season"][0].item()
        basin = raw_overpass_meta["basin"][0]
        storm_number = raw_overpass_meta["cyclone_number"][-1].item()
        sid = f"{season}{basin}{storm_number}"  # Unique storm identifier
        time = pd.to_datetime(raw_overpass_meta["time"][0], origin="unix", unit="s")
        # Storm latitude and longitude
        overpass_storm_metadata = raw_sample["overpass_storm_metadata"]
        storm_lat = overpass_storm_metadata["storm_latitude"][0].item()
        storm_lon = overpass_storm_metadata["storm_longitude"][0].item()
        storm_lon = (storm_lon + 180) % 360 - 180  # Standardize longitude values
        intensity = overpass_storm_metadata["intensity"][0].item()

        sample_metadata = {
            "source_name": f"tc_primed_ir_{tcirar_or_hursat}",
            "source_type": "infrared",
            "sid": sid,
            "time": time,
            "season": season,
            "storm_latitude": storm_lat,
            "storm_longitude": storm_lon,
            "intensity": intensity,
            "basin": basin,
            "dim": 2,  # Spatial dimensionality
        }
        dest_file = dest_dir / f"{sid}_{time.strftime('%Y%m%dT%H%M%S')}.nc"
        sample_metadata["data_path"] = dest_file

        # Check if the file already exists. If it does, skip processing.
        if check_exist and dest_file.exists():
            return sample_metadata, tcirar_or_hursat

        # Standardize longitude values to [-180, 180]
        ds["longitude"] = (ds["longitude"] + 180) % 360 - 180

        # Check for null variables and discard the sample if any are found.
        for variable in ds.variables:
            if ds[variable].isnull().all():
                warnings.warn(f"Variable {variable} is null for sample {file}.")
                return None, None

        # Compute the land-sea mask
        land_mask = globe.is_land(
            ds["latitude"].values,
            ds["longitude"].values,
        )
        # Compute the distance between each grid point and the center of the storm.
        dist_to_center = grid_distance_to_point(
            ds["latitude"].values,
            ds["longitude"].values,
            storm_lat,
            storm_lon,
        )
        # Add the land mask and distance to center as new variables.
        ds["land_mask"] = (("lat", "lon"), land_mask)
        ds["dist_to_center"] = (("lat", "lon"), dist_to_center)

        # Save processed data in the netCDF format
        ds = ds[["IRWIN", "latitude", "longitude", "land_mask", "dist_to_center"]]
        ds = ds.drop_encoding()  # Drop any encoding forwaded by xarray from TC-PRIMED.
        ds = ds.drop_attrs()  # Drop global attributes that may cause issues.
        ds.to_netcdf(dest_file)

        return sample_metadata, tcirar_or_hursat


def process_chunk(file_list, dest_dirs, check_exist, verbose=False):
    """Processes a chunk of infrared files."""
    local_metadata, tcirar_or_hursat = [], []
    iterator = tqdm(file_list, desc="Processing chunk") if verbose else file_list
    for file in iterator:
        meta, src = process_ir_file(file, dest_dirs, check_exist)
        if meta is not None:
            local_metadata.append(meta)
            tcirar_or_hursat.append(src)
    return local_metadata, tcirar_or_hursat


@hydra.main(config_path="../../configs/", config_name="preproc", version_base=None)
def main(cfg):
    """Main function to process IR data."""
    cfg = OmegaConf.to_container(cfg, resolve=True)

    # Setup paths
    tc_primed_path = Path(cfg["paths"]["raw_datasets"]) / "tc_primed"
    dest_path = Path(cfg["paths"]["preprocessed_dataset"]) / "prepared"
    check_exist = cfg.get("check_exist", False)
    include_seasons = cfg.get("include_seasons", None)

    # Get the list of overpass files in TC-PRIMED
    _, tc_primed_files, _ = list_tc_primed_sources(
        tc_primed_path, source_type="satellite", include_seasons=include_seasons
    )
    ir_files = tc_primed_files["infrared"]

    # There are TWO infrared sources: "TCIRAR" when the data comes from
    # the "TC-IRAR" dataset (4km resolution), and "HURSAT" when it comes
    # from the "HURSAT" dataset (8km res).
    tcirar_dest_dir = dest_path / "tc_primed_ir_tcirar"
    hursat_dest_dir = dest_path / "tc_primed_ir_hursat"
    tcirar_dest_dir.mkdir(parents=True, exist_ok=True)
    hursat_dest_dir.mkdir(parents=True, exist_ok=True)
    dest_dirs = {"tcirar": tcirar_dest_dir, "hursat": hursat_dest_dir}

    # Initialize metadata for both sources
    if not initialize_metadata(tcirar_dest_dir):
        print("Failed to initialize metadata for TCIRAR source.")
        return
    if not initialize_metadata(hursat_dest_dir):
        print("Failed to initialize metadata for HURSAT source.")
        return

    # Process all files, and keep count of discarded files and save the metadata of
    # each sample.
    samples_metadata = {"tcirar": [], "hursat": []}
    num_workers = cfg.get("num_workers", 1)

    if num_workers > 1:
        chunks = np.array_split(ir_files, num_workers)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for i, chunk in enumerate(chunks):
                futures.append(
                    executor.submit(
                        process_chunk,
                        list(chunk),
                        dest_dirs,
                        check_exist,
                        verbose=(i == 0),
                    )
                )
            for future in as_completed(futures):
                meta_chunk, tcirar_or_hursat = future.result()
                for sample_metadata, src in zip(meta_chunk, tcirar_or_hursat):
                    samples_metadata[src].append(sample_metadata)
    else:
        for file in tqdm(ir_files, desc="Processing files"):
            sample_metadata, tcirar_or_hursat = process_ir_file(file, dest_dirs, check_exist)
            if sample_metadata is not None:
                samples_metadata[tcirar_or_hursat].append(sample_metadata)

    for src, dest_dir in dest_dirs.items():
        if len(samples_metadata[src]) == 0:
            # For both sources, if no samples were successfully processed, warn the user
            # and delete the source directory.
            warnings.warn(f"No samples were processed for source {src}. Deleting directory.")
            for file in dest_dir.iterdir():
                file.unlink()
            dest_dir.rmdir()
        else:
            # Concatenate the samples metadata into a DataFrame and save it
            pd.DataFrame(samples_metadata[src]).to_csv(
                dest_dir / "samples_metadata.csv", index=False
            )


if __name__ == "__main__":
    main()
