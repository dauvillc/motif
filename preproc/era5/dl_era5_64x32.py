#!/usr/bin/env python3
"""
Downloads ERA5 (1.5°) WeatherBench2 Zarr dataset from Google Cloud Storage.

Usage:
    python preproc/era5/dl_era5_64x32.py paths=local era5_download.workers=16
"""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fsspec
import hydra
from omegaconf import OmegaConf
from tqdm import tqdm


def download_zarr(src_path: str, dst_path: str, workers: int = 8):
    """Recursively copies a Zarr store from a GCS bucket to local disk."""
    fs_src = fsspec.filesystem("gs", token="anon")
    fs_dst = fsspec.filesystem("file")

    print("Listing source files ...")
    files = fs_src.find(src_path)
    print(f"Found {len(files)} objects.")

    def copy_one(src_file):
        rel_path = src_file.replace(src_path, "", 1).lstrip("/")
        dst_file = os.path.join(dst_path, rel_path)
        if fs_dst.exists(dst_file):
            return
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        with fs_src.open(src_file, "rb") as fsrc, open(dst_file, "wb") as fdst:
            fdst.write(fsrc.read())

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(tqdm(pool.map(copy_one, files), total=len(files), desc="Copying"))

    print(f"Dataset downloaded in {dst_path}")


@hydra.main(config_path="../../configs/", config_name="preproc", version_base=None)
def main(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    download_cfg = cfg["era5_download"]
    src = download_cfg["gcs_uri"]
    dst = str(Path(cfg["paths"]["era5_weatherbench"]))
    workers = download_cfg["workers"]
    download_zarr(src, dst, workers)


if __name__ == "__main__":
    main()
