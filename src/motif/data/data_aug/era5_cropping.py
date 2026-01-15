"""Implements the ERA5GridRecentering class."""

import torch


class ERA5Cropping:
    """Designed for samples containing patches of ERA5 data centered on a storm.
    Randomly crops the ERA5 patches to avoid overfitting to the storm center.

    Expects samples from the dataset as dicts {(source_name, index): data} where
    source_name is the name of the source and index is an integer representing
    the observation index (0 = most recent, 1 = second most recent, etc.).
    Each data dict contains the following key-value pairs:
    - "dt" is a scalar tensor of shape (1,) containing the time delta between the reference time
        and the element's time, normalized by dt_max.
    - "characs" is a tensor of shape (n_charac_vars,) containing the source characteristics.
        Each data variable within a source has its own charac variables, which are all
        concatenated into a single tensor.
    - "coords" is a tensor of shape (2, H, W) containing the latitude and longitude at each pixel.
    - "landmask" is a tensor of shape (H, W) containing the land mask.
    - "dist_to_center" is a tensor of shape (H, W) containing the distance
        to the center of the storm.
    - "values" is a tensor of shape (n_variables, H, W) containing the variables for the source.
    """

    def __init__(self, crop_frac_min=0.6, crop_frac_max=1.0, min_size=61, seed=501):
        """
        Args:
            crop_frac_min (float): Minimum fraction of the original size to crop to.
            crop_frac_max (float): Maximum fraction of the original size to crop to.
            min_size (int): Minimum size (in pixels) of the cropped patch.
            seed (int): Random seed for reproducibility.
        """
        self.crop_frac_min = crop_frac_min
        self.crop_frac_max = crop_frac_max
        self.min_size = min_size
        self.gen = torch.Generator()
        self.gen.manual_seed(seed)

    def __call__(self, data):
        """Regrids the target ERA5 patches to a wider grid.
        Args:
            data (dict): Dict containing the data of all sources.
        Returns:
            dict: The data with the ERA5 target patches regridded.
        """
        for (src, src_idx), src_data in data.items():
            if src != "tc_primed_era5":
                continue
            _, H, W = src_data["values"].shape
            crop_frac_w = (
                torch.empty(1)
                .uniform_(self.crop_frac_min, self.crop_frac_max, generator=self.gen)
                .item()
            )
            crop_frac_h = (
                torch.empty(1)
                .uniform_(self.crop_frac_min, self.crop_frac_max, generator=self.gen)
                .item()
            )
            new_W = max(int(W * crop_frac_w), self.min_size)
            new_H = max(int(H * crop_frac_h), self.min_size)
            left = torch.randint(0, W - new_W + 1, (1,), generator=self.gen).item()
            top = torch.randint(0, H - new_H + 1, (1,), generator=self.gen).item()
            right = left + new_W
            bottom = top + new_H
            for key in ["coords", "values"]:
                src_data[key] = src_data[key][:, top:bottom, left:right]
            for key in ["landmask", "dist_to_center"]:
                src_data[key] = src_data[key][top:bottom, left:right]
        return data
