"""Download SAR data from CyclObs for ML pipelines."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import hydra
import pandas as pd
import requests
from omegaconf import OmegaConf
from tqdm import tqdm

API_URL = "https://cyclobs.ifremer.fr/app/api/getData"
MAX_RETRIES = 3
BACKOFF_FACTOR = 2


def get_sar_acquisitions_metadata(
    instrument: str = "C-Band_SAR", include_cols: str = "all"
) -> pd.DataFrame:
    """Queries the CyclObs API to get the list of available SAR acquisitions."""
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
    """Downloads a single file safely using retries and temporary files."""
    if not isinstance(url, str) or not url.startswith("http"):
        return "Invalid URL"

    filename = url.split("/")[-1]
    final_path = output_dir / filename
    temp_path = output_dir / f"{filename}.tmp"

    if final_path.exists():
        return f"Skipped (Exists): {filename}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()

                with temp_path.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

            temp_path.rename(final_path)
            return f"Downloaded: {filename}"

        except (requests.exceptions.RequestException, OSError) as e:
            if temp_path.exists():
                temp_path.unlink()

            if attempt == MAX_RETRIES:
                return f"Failed {filename} after {MAX_RETRIES} attempts: {e}"

            time.sleep(BACKOFF_FACTOR**attempt)

    return f"Failed: {filename}"


@hydra.main(config_path="../../configs/", config_name="preproc", version_base=None)
def main(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    workers = cfg["sar_download"]["workers"]
    download_dir = Path(cfg["paths"]["sar_cyclobs"])
    download_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving data to: {download_dir.resolve()}")

    df_metadata = get_sar_acquisitions_metadata()

    if df_metadata.empty:
        print("No data found or API error.")
        return

    metadata_path = download_dir / "sar_acquisitions_metadata.csv"
    df_metadata.to_csv(metadata_path, index=False)
    print(f"Metadata saved to {metadata_path}")

    if "data_url" not in df_metadata.columns:
        print("Error: 'data_url' column missing.")
        return

    urls_to_download = [url for url in df_metadata["data_url"] if pd.notna(url)]
    total_files = len(urls_to_download)

    print(f"Starting parallel download of {total_files} files with {workers} threads...")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_url = {
            executor.submit(download_file, url, download_dir): url for url in urls_to_download
        }

        with tqdm(total=total_files, unit="file", desc="Downloading") as pbar:
            for future in as_completed(future_to_url):
                result = future.result()
                pbar.update(1)
                if result.startswith("Failed"):
                    tqdm.write(result)

    print("\nDownload complete.")


if __name__ == "__main__":
    main()
