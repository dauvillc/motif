#!/usr/bin/env python3
"""
download_tcprimed.py
--------------------

Bulk-download files from the NOAA TC-PRIMED public S3 bucket.

Examples
========
# 1)  All 2015 Atlantic storms (year index 28: 1987 + 28 = 2015)
python download_tcprimed.py \
    --year 28 \
    --basin AL \
    --dest /scratch/$USER/tcprimed

# 2)  Everything in v01r01/final (≈1.6 TB – be sure you really want it!)
python download_tcprimed.py \
    --dest /scratch/$USER/tcprimed \
    --workers 32
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.client import Config
from tqdm import tqdm

BUCKET_NAME = "noaa-nesdis-tcprimed-pds"


def list_keys(prefix: str):
    """
    Recursively list object keys under `prefix` in the public bucket.
    """
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"], obj["Size"]


def download_one(s3, key: str, size: int, dest_root: str, pbar: tqdm, prefix: str):
    """
    Download a single object unless it already exists locally at the same size.
    """
    local_path = os.path.join(dest_root, os.path.relpath(key, start=prefix))
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Skip if file exists and the size matches
    if os.path.exists(local_path) and os.path.getsize(local_path) == size:
        pbar.update(size)
        return

    # Download with progress callback
    def callback(bytes_transferred):
        pbar.update(bytes_transferred)

    s3.download_file(BUCKET_NAME, key, local_path, Callback=callback)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download TC-PRIMED data (public S3).")
    parser.add_argument(
        "--year",
        type=int,
        help="Year index to download (0=1987, 1=1988, etc.). If not provided, downloads all data.",
    )
    parser.add_argument(
        "--basin",
        type=str,
        choices=["AL", "EP", "CP", "WP", "SH", "IO"],
        help="Basin code to filter by (e.g., 'AL' for Atlantic). Requires --year to be set.",
    )
    parser.add_argument("--dest", required=True, help="Root destination directory for the dataset.")
    parser.add_argument(
        "--workers", type=int, default=8, help="Parallel download threads (default: 8)."
    )
    args = parser.parse_args()

    # Construct prefix based on arguments
    if args.year is not None:
        absolute_year = 1987 + args.year
        prefix = f"v01r01/final/{absolute_year}/"
        dest_root = Path(args.dest) / str(absolute_year)
    else:
        prefix = "v01r01/final/"
        dest_root = Path(args.dest)
    if args.basin is not None:
        if args.year is None:
            parser.error("--basin requires --year to be set.")
        prefix = os.path.join(prefix, args.basin) + "/"
        dest_root = dest_root / args.basin
    print(f"Downloading from s3://{BUCKET_NAME}/{prefix} to {dest_root}/")

    # Anonymous (unsigned) S3 client
    s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    # Find everything we're going to grab
    objects = list(list_keys(prefix))
    total_size = sum(size for _, size in objects)

    print(f"Found {len(objects):,} files – {total_size / 1e9:,.2f} GB.")

    # Multi-threaded download with progress bar showing bytes
    with tqdm(
        total=total_size, desc="Downloading", unit="B", unit_scale=True, unit_divisor=1024
    ) as pbar:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(download_one, s3_client, key, size, dest_root, pbar, prefix)
                for key, size in objects
            ]
            for future in as_completed(futures):
                future.result()  # Raise any exceptions

    print("Done.")
