import argparse
import time
from pathlib import Path
from typing import Optional, Set, Tuple
from urllib.parse import parse_qs, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# --- Configuration ---
BASE_URL: str = "https://www.star.nesdis.noaa.gov/socd/mecb/sar/"
MAIN_PAGE_URL: str = "https://www.star.nesdis.noaa.gov/socd/mecb/sar/sarwinds_tropical.php"
START_YEAR: int = 2016
END_YEAR: int = 2025

# HTTP Headers to mimic a real web browser.
HEADERS: dict = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


def get_soup(url: str) -> Optional[BeautifulSoup]:
    """
    Fetches the content of a URL and parses it into a BeautifulSoup object.
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        return BeautifulSoup(response.content, "html.parser")
    except requests.RequestException as e:
        print(f"[Error] Fetching {url}: {e}")
        return None


def download_file(url: str, dest_path: Path) -> bool:
    """
    Downloads a single file from a URL using a temporary file strategy.
    Returns True on success, False on failure.
    """
    # Define temp path
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    if temp_path.exists():
        temp_path.unlink()  # Remove any existing temp file

    try:
        # Download to .tmp file
        with requests.get(url, headers=HEADERS, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(temp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        # Rename to final on success
        temp_path.rename(dest_path)
        return True

    except Exception:
        # Cleanup incomplete temp file
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        return False


def main() -> None:
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Download NOAA SAR 3km NetCDF files for tropical cyclones (2016-2025)."
    )
    parser.add_argument(
        "--dest",
        type=str,
        help="Destination directory (default: ./noaa_sar_winds_3km)",
    )
    args = parser.parse_args()

    download_root = Path(args.dest)

    # 1. Setup root directory
    if not download_root.exists():
        download_root.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {download_root}")

    print(f"Crawling main page: {MAIN_PAGE_URL}")
    soup: Optional[BeautifulSoup] = get_soup(MAIN_PAGE_URL)
    if not soup:
        return

    # 2. Identify storm pages
    storm_links: Set[Tuple[int, str]] = set()
    for a in soup.find_all("a", href=True):
        href: str = a["href"]
        if "year=" in href and "storm=" in href:
            full_url: str = urljoin(MAIN_PAGE_URL, href)
            parsed = urlparse(full_url)
            qs = parse_qs(parsed.query)
            year_list = qs.get("year", [])

            if year_list:
                try:
                    y = int(year_list[0])
                    if START_YEAR <= y <= END_YEAR:
                        storm_links.add((y, full_url))
                except ValueError:
                    continue

    sorted_storms = sorted(list(storm_links), key=lambda x: x[0], reverse=True)
    print(f"Found {len(sorted_storms)} storm pages.")

    # Phase 1: Collect all file URLs
    print("\nPhase 1: Scanning storm pages for files...")
    download_list = []

    for year, storm_url in sorted_storms:
        parsed_url = urlparse(storm_url)
        qs = parse_qs(parsed_url.query)
        storm_id = qs.get("storm", ["unknown"])[0]

        # Create storm folder
        storm_dir = download_root / str(year) / storm_id
        if not storm_dir.exists():
            storm_dir.mkdir(parents=True, exist_ok=True)

        # Get storm page content
        print(f"  Scanning: {storm_id} ({year})")
        storm_soup = get_soup(storm_url)
        if not storm_soup:
            continue

        # Find 3km NetCDF links
        for a in storm_soup.find_all("a", href=True):
            text = a.get_text().strip().lower()
            href = a["href"]

            if href.endswith(".nc") and "3km" in text:
                file_url = urljoin(storm_url, href)
                parsed_file = urlparse(file_url)
                filename = Path(parsed_file.path).name
                dest_path = storm_dir / filename
                download_list.append((file_url, dest_path))

        # Tiny sleep to be polite while scraping the pages
        time.sleep(0.5)

    # Save all URLs to file
    urls_file = download_root / "all_urls.txt"
    with open(urls_file, "w") as f:
        for file_url, dest_path in download_list:
            f.write(f"{file_url}\n")
    print(f"\nAll URLs saved to: {urls_file}")

    # Filter out already-downloaded files
    files_to_download = [(url, path) for url, path in download_list if not path.exists()]
    num_skipped = len(download_list) - len(files_to_download)

    print(f"\nTotal files found: {len(download_list)}")
    print(f"Already downloaded: {num_skipped}")
    print(f"To download: {len(files_to_download)}")

    if not files_to_download:
        print("\nAll files already downloaded.")
        return

    # Phase 2: Download files with progress bar
    print("\nPhase 2: Downloading files...")
    successes = 0
    failures = 0
    failed_urls = []

    with tqdm(files_to_download, desc="Downloading", unit="file") as pbar:
        for file_url, dest_path in pbar:
            if download_file(file_url, dest_path):
                successes += 1
            else:
                failures += 1
                failed_urls.append(file_url)
            pbar.set_postfix(success=successes, fail=failures)

    # Save failed URLs to file
    if failed_urls:
        failed_file = download_root / "failed_downloads.txt"
        with open(failed_file, "w") as f:
            for url in failed_urls:
                f.write(f"{url}\n")
        print(f"\nFailed URLs saved to: {failed_file}")

    # Print summary
    print(f"\nDownload complete: {successes} succeeded, {failures} failed")


if __name__ == "__main__":
    main()
