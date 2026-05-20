import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# Constants
API_URL = "https://cyclobs.ifremer.fr/app/api/getData"
MAX_WORKERS = 4
MAX_RETRIES = 3
BACKOFF_FACTOR = 2  # Seconds to wait between retries (multiplied by attempt)


def get_sar_acquisitions_metadata(
    instrument: str = "C-Band_SAR", include_cols: str = "all"
) -> pd.DataFrame:
    """
    Queries the CyclObs API to get the list of available SAR acquisitions.

    Args:
        instrument: The instrument filter. Defaults to "C-Band_SAR".
        include_cols: Columns to include. "all" provides comprehensive metadata.

    Returns:
        pd.DataFrame: A DataFrame containing the metadata.
    """
    params = {
        "instrument": instrument,
        "include_cols": include_cols,
    }

    print(f"Querying CyclObs API at {API_URL} for instrument='{instrument}'...")

    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()
        df = pd.read_csv(response.url)
        print(f"Found {len(df)} acquisitions.")
        return df

    except requests.exceptions.RequestException as e:
        print(f"Error querying API: {e}")
        return pd.DataFrame()


def download_file(url: str, output_dir: Path) -> str:
    """
    Downloads a single file safely using retries and temporary files.

    Process:
    1. Checks if final file exists.
    2. Downloads to {filename}.tmp.
    3. Renames to {filename} only upon success.

    Args:
        url: The direct download URL.
        output_dir: The target directory as a Path object.

    Returns:
        str: Status message.
    """
    if not isinstance(url, str) or not url.startswith("http"):
        return "Invalid URL"

    # Extract filename and create Path objects
    filename = url.split("/")[-1]
    final_path = output_dir / filename
    temp_path = output_dir / f"{filename}.tmp"

    # 1. Skip if the final file already exists (Resume capability)
    if final_path.exists():
        return f"Skipped (Exists): {filename}"

    # Retry loop
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()

                # 2. Write to temporary file
                with temp_path.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

            # 3. Atomic rename upon success
            temp_path.rename(final_path)
            return f"Downloaded: {filename}"

        except (requests.exceptions.RequestException, IOError) as e:
            # Clean up partial temp file if it exists
            if temp_path.exists():
                temp_path.unlink()

            if attempt == MAX_RETRIES:
                return f"Failed {filename} after {MAX_RETRIES} attempts: {e}"

            # Exponential backoff sleep
            sleep_time = BACKOFF_FACTOR**attempt
            time.sleep(sleep_time)

    return f"Failed: {filename}"


def main():
    """
    Main execution function.
    Parses arguments, sets up directory, and manages parallel downloads.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Download SAR data from CyclObs for ML pipelines.")
    parser.add_argument("--dest", type=Path, help="Destination directory for downloaded files.")
    args = parser.parse_args()

    # Ensure download directory exists
    download_dir = args.dest
    download_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving data to: {download_dir.resolve()}")

    # 1. Get metadata
    df_metadata = get_sar_acquisitions_metadata()

    if df_metadata.empty:
        print("No data found or API error.")
        return

    # Save metadata locally
    metadata_path = download_dir / "sar_acquisitions_metadata.csv"
    df_metadata.to_csv(metadata_path, index=False)
    print(f"Metadata saved to {metadata_path}")

    # 2. Prepare downloads
    if "data_url" not in df_metadata.columns:
        print("Error: 'data_url' column missing.")
        return

    # Filter valid URLs
    urls_to_download = [url for url in df_metadata["data_url"] if pd.notna(url)]
    total_files = len(urls_to_download)

    print(f"Starting parallel download of {total_files} files with {MAX_WORKERS} threads...")

    # 3. Execute Parallel Downloads
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_url = {
            executor.submit(download_file, url, download_dir): url for url in urls_to_download
        }

        with tqdm(total=total_files, unit="file", desc="Downloading") as pbar:
            for future in as_completed(future_to_url):
                result = future.result()
                pbar.update(1)

                # Optional: specific error logging
                if result.startswith("Failed"):
                    tqdm.write(result)

    print("\nDownload complete.")


if __name__ == "__main__":
    main()
