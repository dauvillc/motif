"""
Prepares the SAR data over tropical cyclones and matches them with the TC-PRIMED dataset.
"""

import json
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import xarray as xr
from omegaconf import OmegaConf
from shapely.wkt import loads as wkt_loads
from tqdm import tqdm

from motif.data.grid_functions import grid_distance_to_point

DATA_VARS = ["wind_speed"]


def initialize_radar_metadata(dest_dir):
    """
    Initializes the metadata for the sar source.
    Args:
        dest_dir (Path): Destination directory.
    Returns:
        bool: True if the metadata was successfully written, False otherwise.
    """
    # Variables that give information about the storm of the sample.
    storm_vars = ["storm_latitude", "storm_longitude", "intensity"]
    # Dict that contains the source's metadata.
    source_metadata = {
        "source_name": "sar_cband",
        "source_type": "sar",
        "dim": 2,
        "data_vars": DATA_VARS,
        "storm_metadata_vars": storm_vars,
        "charac_vars": {},  # No characteristic variables for SAR
    }
    with open(dest_dir / "source_metadata.json", "w") as f:
        json.dump(source_metadata, f)
    return True


def process_file(
    file,
    file_info,
    dest_dir,
    check_older=None,
):
    """Processes a single SAR TC file.
    Args:
        file (str): Path to the overpass file in netCDF4 format.
        file_info (pd.Series): Information about the file from the acquisition metadata.
        dest_dir (Path): Destination directory.
        check_older (timedelta or None): if a timedelta dt, checks if there is a pre-existing
            file younger than dt. If so, skips processing.
    Returns:
        dict or None: Sample metadata, or None if the sample is discarded.
    """
    ds = xr.open_dataset(file, decode_times=False)

    # 1. Extract data variables and coordinates
    ds = ds[["lat", "lon", "mask_flag"] + DATA_VARS]
    # - Remove the channel dimension in data variables
    for var in DATA_VARS:
        ds[var] = (("lat", "lon"), ds[var].values[0])
    # - If any variable is fully missing data, skip
    if any(ds[var].isnull().all() for var in DATA_VARS):
        return None

    # 2. Assemble the sample's metadata
    sid = file_info["sid"]  # YYYYBBNNN
    season, basin = sid[:4], sid[4:6]
    # - Retrieve the time and storm metadata
    time = file_info["acquisition_start_time"]
    storm_center_loc = wkt_loads(file_info["track_point"])
    storm_lat, storm_lon = storm_center_loc.y, storm_center_loc.x
    storm_lon = (storm_lon + 180) % 360 - 180  # Standardize longitude values
    intensity = file_info["vmax (m/s)"] * 1.94384  # convert from m/s to knots

    sample_metadata = {
        "source_name": "sar_cband",
        "source_type": "sar",
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

    # Check if the file already exists and is younger than the max timedelta
    if dest_file.exists() and check_older is not None:
        mtime = pd.to_datetime(dest_file.stat().st_mtime, unit="s")
        if pd.Timestamp.now() - mtime < check_older:
            return sample_metadata

    # 3. Process the data
    # - The "lat" and "lon" variables are given at 1D arrays; convert to a 2D meshgrid
    lat_1d = ds["lat"].values
    lon_1d = ds["lon"].values
    lon_1d = (lon_1d + 180) % 360 - 180  # Standardize longitude values
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
    ds = ds.assign_coords(
        {"latitude": (("lat", "lon"), lat_2d), "longitude": (("lat", "lon"), lon_2d)}
    )
    # - Retrieve the land-sea mask
    land_mask = ds["mask_flag"].values == 0  # only 0 means valid
    # Compute the distance between each grid point and the center of the storm.
    dist_to_center = grid_distance_to_point(
        ds["latitude"].values,
        ds["longitude"].values,
        storm_lat,
        storm_lon,
    )
    # Add the land mask and distance to center as new variables.
    ds["land_mask"] = (("lat", "lon"), land_mask[0])
    ds["dist_to_center"] = (("lat", "lon"), dist_to_center)

    # Save processed data in the netCDF format
    ds = ds[DATA_VARS + ["latitude", "longitude", "land_mask", "dist_to_center"]]
    # Encoding: no compression, float32 for all variables.
    encoding = {
        var: {
            "dtype": "float32",
            "zlib": False,
            "_FillValue": float("nan"),
        }
        for var in ds.data_vars
    }
    ds = ds.drop_encoding()
    ds.to_netcdf(dest_file, encoding=encoding)

    return sample_metadata


@hydra.main(config_path="../../configs/", config_name="preproc", version_base=None)
def main(cfg):
    """Main function to process the radar data."""
    cfg = OmegaConf.to_container(cfg, resolve=True)

    # Setup paths
    sar_dir = Path(cfg["paths"]["raw_datasets"]) / "sar_cyclobs"
    dest_dir = Path(cfg["paths"]["preprocessed_dataset"]) / "prepared" / "sar_cband"
    dest_dir.mkdir(parents=True, exist_ok=True)

    include_seasons = cfg.get("include_seasons", None)
    check_older = cfg.get("check_older", None)
    check_older = pd.to_timedelta(check_older) if check_older is not None else None

    # Load the cyclobs SAR acquisitions metadata
    acq_metadata_path = sar_dir / "sar_acquisitions_metadata.csv"
    acq_df = pd.read_csv(acq_metadata_path, parse_dates=["acquisition_start_time"])
    acq_df = acq_df.reset_index(drop=True)
    # The "sid" column gives the sid in the format bbNNYYYY where
    # bb: basin code, NN: storm number, YYYY: year
    # - We'll extract the components and reformat to YYYYBBNNN
    #  for example, al022024 -> 2024AL2
    acq_df["season"] = acq_df["sid"].str[-4:]
    acq_df["basin"] = acq_df["sid"].str[:2].str.upper()
    # For the storm number, remove zero padding
    acq_df["storm_number"] = acq_df["sid"].str[2:4].astype(int).astype(str)
    acq_df["sid"] = acq_df["season"] + acq_df["basin"] + acq_df["storm_number"]
    # - Filter by seasons if specified
    if include_seasons is not None:
        acq_df = acq_df[acq_df["season"].isin(include_seasons)].reset_index(drop=True)
        print(f"Filtered acquisitions to include seasons {include_seasons}: {len(acq_df)} files.")

    # Process each row in the acquisitions metadata, corresponding to a SAR file
    if not initialize_radar_metadata(dest_dir):
        # Remove directory if metadata could not be initialized
        dest_dir.rmdir()
        print("Failed to initialize SAR metadata.")
        return

    # Process all files, and keep count of discarded files and save the metadata of
    # each sample.
    discarded, samples_metadata = 0, []
    num_workers = cfg.get("num_workers", 1)
    chunksize = cfg.get("chunksize", 64)

    # Extract the file names from the "data_url" column
    sar_files = acq_df["data_url"].apply(lambda url: url.split("/")[-1])

    if num_workers <= 1:
        for k, row in tqdm(acq_df.iterrows(), total=len(acq_df), desc="Processing SAR files"):
            file = sar_dir / sar_files[k]
            sample_metadata = process_file(
                file,
                row,
                dest_dir,
                check_older,
            )
            if sample_metadata is None:
                discarded += 1
            else:
                samples_metadata.append(sample_metadata)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            iterator = executor.map(
                process_file,
                [sar_dir / sar_files[k] for k, row in acq_df.iterrows()],
                [row for _, row in acq_df.iterrows()],
                repeat(dest_dir),
                repeat(check_older),
                chunksize=chunksize,
            )
            for sample_metadata in tqdm(
                iterator,
                desc="Processing SAR files",
                total=len(acq_df),
            ):
                if sample_metadata is None:
                    discarded += 1
                else:
                    samples_metadata.append(sample_metadata)

    # If all files were discarded, remove all files inside the directory
    # and remove the directory itself.
    if discarded == len(acq_df):
        for file in dest_dir.iterdir():
            file.unlink()
        dest_dir.rmdir()
        print("Found no valid samples for SAR, removing directory.")
    else:
        if discarded > 0:
            percent = discarded / len(acq_df) * 100
            print(f"Discarded {discarded} samples for SAR ({percent:.2f}%)")
        # Concatenate the samples metadata into a DataFrame and save it
        samples_metadata = pd.DataFrame(samples_metadata)
        samples_metadata_path = dest_dir / "samples_metadata.csv"
        samples_metadata.to_csv(samples_metadata_path, index=False)


if __name__ == "__main__":
    main()
