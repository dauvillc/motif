#!/usr/bin/env python3
"""
Bulk-download files from the NOAA TC-PRIMED public S3 bucket.

Examples
========
# All 2015 Atlantic storms (year index 28: 1987 + 28 = 2015)
python preproc/tc_primed/download_tc_primed.py paths=jz \
    tc_primed_download.year=28 tc_primed_download.basin=AL

# Everything in v01r01/final (~1.6 TB)
python preproc/tc_primed/download_tc_primed.py paths=jz tc_primed_download.workers=32
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import hydra
from botocore import UNSIGNED
from botocore.client import Config
from omegaconf import OmegaConf
from tqdm import tqdm

BUCKET_NAME = "noaa-nesdis-tcprimed-pds"


def list_keys(prefix: str):
    """Recursively list object keys under `prefix` in the public bucket."""
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"], obj["Size"]


def download_one(s3, key: str, size: int, dest_root: str, pbar: tqdm, prefix: str):
    """Download a single object unless it already exists locally at the same size."""
    local_path = os.path.join(dest_root, os.path.relpath(key, start=prefix))
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if os.path.exists(local_path) and os.path.getsize(local_path) == size:
        pbar.update(size)
        return

    def callback(bytes_transferred):
        pbar.update(bytes_transferred)

    s3.download_file(BUCKET_NAME, key, local_path, Callback=callback)


@hydra.main(config_path="../../configs/", config_name="preproc", version_base=None)
def main(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    download_cfg = cfg["tc_primed_download"]
    year = download_cfg["year"]
    basin = download_cfg["basin"]
    workers = download_cfg["workers"]

    if basin is not None and year is None:
        raise ValueError("tc_primed_download.basin requires tc_primed_download.year to be set.")

    dest_base = Path(cfg["paths"]["tc_primed"])
    if year is not None:
        absolute_year = 1987 + year
        prefix = f"v01r01/final/{absolute_year}/"
        dest_root = dest_base / str(absolute_year)
    else:
        prefix = "v01r01/final/"
        dest_root = dest_base
    if basin is not None:
        prefix = os.path.join(prefix, basin) + "/"
        dest_root = dest_root / basin

    print(f"Downloading from s3://{BUCKET_NAME}/{prefix} to {dest_root}/")

    s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    objects = list(list_keys(prefix))
    total_size = sum(size for _, size in objects)

    print(f"Found {len(objects):,} files – {total_size / 1e9:,.2f} GB.")

    with tqdm(
        total=total_size, desc="Downloading", unit="B", unit_scale=True, unit_divisor=1024
    ) as pbar:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(download_one, s3_client, key, size, dest_root, pbar, prefix)
                for key, size in objects
            ]
            for future in as_completed(futures):
                future.result()

    print("Done.")


if __name__ == "__main__":
    main()
