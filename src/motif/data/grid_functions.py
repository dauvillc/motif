"""Implements functions to manipulate gridded data."""

import numpy as np
import torch
from haversine import haversine_vector

EARTH_RADIUS = 6371228.0  # Earth radius in meters


def crop_nan_border(src_image, tgt_images):
    """Computes the smallest rectangle to which a source image can be
    cropped without losing any non-NaN values; then crops the target images
    to that rectangle.

    Args:
        src_image (torch.Tensor): The source image of shape (C, H, W).
        tgt_images (list of torch.Tensor): Target images of shape (C, H, W) or (H, W).

    Returns:
        list of torch.Tensor: The cropped target images.
    """
    row_full_nan = torch.isnan(src_image).all(dim=(0, 2)).int()
    col_full_nan = torch.isnan(src_image).all(dim=(0, 1)).int()
    first_row = torch.argmax(~row_full_nan).item()
    last_row = row_full_nan.size(0) - torch.argmax(~row_full_nan.flip(dims=[0])).item() + 1
    first_col = torch.argmax(~col_full_nan).item()
    last_col = col_full_nan.size(0) - torch.argmax(~col_full_nan.flip(dims=[0])).item() + 1

    tgt_images_cropped = []
    for tgt_image in tgt_images:
        if tgt_image.ndim == 3:
            tgt_image_cropped = tgt_image[:, first_row:last_row, first_col:last_col]
        else:
            tgt_image_cropped = tgt_image[first_row:last_row, first_col:last_col]
        tgt_images_cropped.append(tgt_image_cropped)
    return tgt_images_cropped


def crop_nan_border_numpy(src_image, tgt_images):
    """Computes the smallest rectangle to which a source image can be
    cropped without losing any non-NaN values; then crops the target images
    to that rectangle. NumPy implementation.

    Args:
        src_image (numpy.ndarray): The source image of shape (H, W).
        tgt_images (list of numpy.ndarray): Target images of shape (..., H, W),
            where ... is an arbitrary number of leading dimensions.

    Returns:
        list of numpy.ndarray: The cropped target images.
    """
    # Find rows and columns that are all NaN
    row_full_nan = np.all(np.isnan(src_image), axis=1)
    col_full_nan = np.all(np.isnan(src_image), axis=0)

    # Find first and last non-NaN rows and columns
    non_nan_rows = np.where(~row_full_nan)[0]
    non_nan_cols = np.where(~col_full_nan)[0]

    if len(non_nan_rows) == 0 or len(non_nan_cols) == 0:
        # Return originals if all NaN
        return tgt_images

    first_row = non_nan_rows[0]
    last_row = non_nan_rows[-1] + 1  # Add 1 for exclusive upper bound in slicing
    first_col = non_nan_cols[0]
    last_col = non_nan_cols[-1] + 1  # Add 1 for exclusive upper bound in slicing

    # Crop target images
    tgt_images_cropped = []
    for tgt_image in tgt_images:
        tgt_image_cropped = tgt_image.T[first_col:last_col, first_row:last_row].T
        tgt_images_cropped.append(tgt_image_cropped)
    return tgt_images_cropped


def grid_distance_to_point(grid_lat, grid_lon, lat, lon):
    """Computes the distance between a point and all points of a grid.
    Args:
        grid_lat (numpy.ndarray): The latitude grid, of shape (H, W).
        grid_lon (numpy.ndarray): The longitude grid, of shape (H, W).
        lat (float): The latitude of the point.
        lon (float): The longitude of the point.
    Returns:
        numpy.ndarray: array of shape (H, W) containing the distances between
            the point and all points of the grid.
    """
    # Stack the latitude and longitude grids
    grid_latlon = np.stack([grid_lat, grid_lon], axis=-1)  # (H, W, 2)
    # Haversine expects an array of shape (N_points, 2)
    grid_latlon = grid_latlon.reshape(-1, 2)  # (H * W, 2)
    return haversine_vector([(lat, lon)], grid_latlon, comb=True).reshape(grid_lat.shape)


def distance_to_overlap(
    target_coords, *ref_coords, downsample_factor_ref=5, downsample_factor_target=2
):
    """For every point in a target 2D grid of latitudes and longitudes,
    and given reference 2D grids of coordinates,
    computes the distance to the closest point in any of the reference grids.
    Args:
        target_coords (torch.Tensor): tensor of shape (B, 2, H, W) containing the
            latitudes and longitudes of the target grid.
        *ref_coords (torch.Tensor): tensors of shape (B, 2, H, W) containing the
            latitudes and longitudes of the reference grids.
        downsample_factor_ref (int): Factor by which to downsample the reference grids
            before computing the distances. This greatly speeds up the computation
            and reduces memory usage, at the cost of some precision.
        downsample_factor_target (int): Factor by which to downsample the target grid
            before computing the distances.
    Returns:
        torch.Tensor: tensor of shape (B, H, W) containing the distances
            (in km) to the closest point in any of the reference grids.
            If a point in the target grid is NaN, the distance is NaN.
    """
    # Downsample the grids
    if downsample_factor_ref > 1:
        ref_coords = [
            ref[:, :, ::downsample_factor_ref, ::downsample_factor_ref] for ref in ref_coords
        ]
    if downsample_factor_target > 1:
        factor = downsample_factor_target
        H0, W0 = target_coords.shape[2], target_coords.shape[3]
        target_coords = target_coords[:, :, ::factor, ::factor]
    # Reshape the tensors to (B, H*W, 2)
    B, _, H, W = target_coords.shape
    target_coords = target_coords.reshape(B, 2, -1).permute(0, 2, 1)
    ref_coords = [ref.reshape(B, 2, -1).permute(0, 2, 1) for ref in ref_coords]
    # Concatenate the reference coordinates along the points dimension
    ref_coords = torch.cat(ref_coords, dim=1)  # (B, N_ref_points, 2)
    # Compute the distance between each point in the target grid and each
    # point in the reference grids. We'll use the euclidean distance on the
    # lat/lon coordinates as an approximation of the haversine distance.
    # This is valid for small distances (up to a few hundred km).
    dists = torch.cdist(target_coords, ref_coords, p=2)  # (B, N_target_points, N_ref_points)
    dists = torch.nan_to_num(dists, nan=float("inf"))
    min_dists, _ = dists.min(dim=-1)  # (B, N_target_points)
    min_dists = min_dists.view(B, H, W)
    # Convert from degrees to km (approximation)
    min_dists = min_dists * (2 * np.pi * EARTH_RADIUS / 360) / 1000.0
    # Set distances to inf where the target coordinates are NaN
    target_nan_mask = torch.isnan(target_coords).any(dim=-1).view(B, H, W)
    min_dists = min_dists.masked_fill(target_nan_mask, float("inf"))
    # If downsampled, upsample back to original size
    if downsample_factor_target > 1:
        min_dists = torch.nn.functional.interpolate(
            min_dists.unsqueeze(1),
            size=(H0, W0),
            mode="nearest",
        ).squeeze(1)
    return min_dists
