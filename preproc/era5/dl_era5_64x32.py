#!/usr/bin/env python3
"""
Downloads ERA5 (1.5°) WeatherBench2 Zarr dataset from Google Cloud Storage.

Usage:
    python dl_era5_64x32.py --output /path/to/local/era5_1p5deg.zarr --workers 16
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import fsspec
from tqdm import tqdm


def download_zarr(src_path: str, dst_path: str, workers: int = 8):
    """Recursively copies a Zarr store from a GCS bucket to local disk."""
    # Open source and destination filesystems
    fs_src = fsspec.filesystem("gs", token="anon")
    fs_dst = fsspec.filesystem("file")

    # List all files (keys) in the Zarr store
    print("Listing source files ...")
    files = fs_src.find(src_path)
    print(f"Found {len(files)} objects.")

    # Parallel copy with tqdm progress bar

    def copy_one(src_file):
        rel_path = src_file.replace(src_path, "", 1).lstrip("/")
        dst_file = os.path.join(dst_path, rel_path)
        if fs_dst.exists(dst_file):
            return  # Skip existing files (resume)
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        with fs_src.open(src_file, "rb") as fsrc, open(dst_file, "wb") as fdst:
            fdst.write(fsrc.read())

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(tqdm(pool.map(copy_one, files), total=len(files), desc="Copying"))

    print(f"Dataset downloaded in {dst_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", required=True, help="Destination folder for the local Zarr store."
    )
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel threads.")
    args = parser.parse_args()

    src = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"
    download_zarr(src, args.output, args.workers)


if __name__ == "__main__":
    main()
