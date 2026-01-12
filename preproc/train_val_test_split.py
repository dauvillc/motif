"""Splits the data into training, validation, and test sets."""

from pathlib import Path

import hydra
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm


@hydra.main(config_path="../configs/", config_name="preproc", version_base=None)
def main(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    # Path to the preprocessed dataset
    preprocessed_dir = Path(cfg["paths"]["preprocessed_dataset"])
    # Path to the preprocessed dataset
    regridded_dir = preprocessed_dir / "prepared"
    train_file = preprocessed_dir / "train.csv"
    val_file = preprocessed_dir / "val.csv"
    test_file = preprocessed_dir / "test.csv"

    # Each subdirectory in the regridded directory corresponds to a source, and contains
    # a file "samples_metadata.csv". We'll load all of these files and assemble them
    # into a single DataFrame, which we'll then split into training, validation, and test sets.
    metadata = []
    for source_dir in tqdm(list(regridded_dir.iterdir()), desc="Loading metadata"):
        metadata_file = source_dir / "samples_metadata.csv"
        metadata.append(pd.read_csv(metadata_file, parse_dates=["time"]))
    print("Concatenating metadata")
    metadata = pd.concat(metadata, ignore_index=True)

    # In some cases the same source may have multiple images for the same storm
    # very close in time (e.g. at less than an hour interval). Those images would
    # be almost identical, so we'll filter the data to keep only one image every
    # min_time_between_same_source minutes for each storm and source.
    metadata = metadata.sort_values(["sid", "source_name", "time"])
    delta = pd.Timedelta(minutes=cfg["min_time_between_same_source"])

    def keep_one_every_delta(g):
        """Keep only one sample every delta time in the group g."""
        kept = []
        last_kept_time = pd.Timestamp.min
        for idx, t in zip(g.index, g["time"]):
            if t >= last_kept_time + delta:
                kept.append(idx)
                last_kept_time = t
        return g.loc[kept]

    metadata = (
        metadata.groupby(["sid", "source_name"], group_keys=False)
        .apply(keep_one_every_delta)
        .reset_index(drop=True)
    )

    # We'll sort the samples by storm ID, then by time and finally by source.
    # This isn't required for the pipeline to work, but it makes it easier to
    # inspect the data.
    metadata = metadata.sort_values(["sid", "time", "source_name"], ascending=[True, False, True])
    metadata = metadata.reset_index(drop=True)

    # Load the seasons to use for the validation and test sets
    val_seasons, test_seasons = cfg["val_seasons"], cfg["test_seasons"]
    # We'll use the SID to determine the season of each sample. The season is the
    # first four characters of the SID.
    metadata["season"] = metadata["sid"].str[:4].astype(int)
    # Split the data into training, validation, and test sets based on the seasons.
    val_mask = metadata["season"].isin(val_seasons)
    test_mask = metadata["season"].isin(test_seasons)
    train_mask = ~(val_mask | test_mask)
    train = metadata[train_mask]
    val = metadata[val_mask]
    test = metadata[test_mask]

    # Sort and reset indices (cleaner for later inspection)
    train = train.sort_values(["sid", "time", "source_name"], ascending=[True, False, True])
    val = val.sort_values(["sid", "time", "source_name"], ascending=[True, False, True])
    test = test.sort_values(["sid", "time", "source_name"], ascending=[True, False, True])
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)

    # Save the split metadata to disk as preprocessed_dir/train.json, ...
    print("Saving split metadata")
    train.to_csv(train_file, index=False)
    val.to_csv(val_file, index=False)
    test.to_csv(test_file, index=False)
    print(f"Training samples: {len(train)}")
    print(f"Validation samples: {len(val)}")
    print(f"Test samples: {len(test)}")


if __name__ == "__main__":
    main()
