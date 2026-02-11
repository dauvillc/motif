"""Implements functions to manipulate gridded data."""

import logging
import math
import traceback
import warnings
from typing import cast

import numpy as np
import xarray as xr
from numpy import nan as NA
from pyresample.area_config import create_area_def
from pyresample.bilinear._numpy_resampler import NumpyBilinearResampler
from pyresample.geometry import AreaDefinition, DynamicAreaDefinition, SwathDefinition
from pyresample.image import ImageContainerNearest
from pyresample.utils import check_and_wrap


class DisableLogger:
    """Context manager to disable logging temporarily."""

    def __enter__(self):
        logging.basicConfig(level=logging.ERROR, force=True)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.basicConfig(level=logging.INFO, force=True)


class ResamplingError(ValueError):
    """Exception raised when resampling operations fail."""

    pass


# Disable some warnings that pyresample and CRS are raising
# but can be safely ignored
warnings.simplefilter(action="ignore")


EARTH_RADIUS = 6371228.0  # Earth radius in meters


def pad_dataset(ds: xr.Dataset, max_size: tuple[int, int]) -> xr.Dataset:
    """Pads the dataset ds to the size max_size, by adding missing values
    at the bottom and right of the dataset.
    If the dataset is already larger than max_size, crops it to max_size.

    Args:
        ds (xarray.Dataset): The dataset to pad.
        max_size (tuple[int, int]): The maximum size of the dataset, as a tuple (scan, pixel).
    """
    # Retrieve the size of the dataset
    sizes = ds.sizes
    size = (sizes["scan"], sizes["pixel"])
    # Compute the padding values
    pad_scan = max_size[0] - size[0]
    pad_pixel = max_size[1] - size[1]
    # Set all -9999.9 values to NaN
    ds = ds.where(ds != -9999.9)
    # Along the scan dimension, pad the dataset with NaN values if pad_scan > 0
    # or crop it if pad_scan < 0
    if pad_scan > 0:
        ds = ds.pad(scan=(0, pad_scan), mode="constant", constant_values=NA)
    elif pad_scan < 0:
        ds = ds.isel(scan=slice(None, max_size[0]))
    # Same for the pixel dimension
    if pad_pixel > 0:
        ds = ds.pad(pixel=(0, pad_pixel), mode="constant", constant_values=NA)
    elif pad_pixel < 0:
        ds = ds.isel(pixel=slice(None, max_size[1]))
    return ds


def reverse_spatially(ds: xr.Dataset, dim_x: str, dim_y: str) -> xr.Dataset:
    """Reverses the dataset ds spatially, i.e. possibly reverses the spatial dimensions
    so that the latitude is decreasing from the top to the bottom of the image,
    and the longitude is increasing from left to right.

    Args:
        ds (xarray.Dataset): The dataset to reverse.
        dim_x (str): The name of the longitude dimension.
        dim_y (str): The name of the latitude dimension.
    """
    # The latitude variable in TC-PRIMEd is in degrees north, so it should be
    # decreasing from the top to the bottom of the image.
    if ds.latitude[0, 0] < ds.latitude[1, 0]:
        return ds.isel(**{dim_y: slice(None, None, -1)})  # type: ignore
    if ds.longitude[0, 0] > ds.longitude[0, 1]:
        return ds.isel(**{dim_x: slice(None, None, -1)})  # type: ignore
    return ds


def regrid(
    ds: xr.Dataset,
    target_resolution: float,
    has_channel_dimension: bool = False,
    target_area: AreaDefinition | None = None,
    return_area: bool = False,
) -> xr.Dataset | tuple[xr.Dataset, AreaDefinition]:
    """Regrids the dataset ds to a regular grid with a given target resolution.
    Uses an equiangular (Plate Carrée) projection.

    Args:
        ds: Dataset to regrid. Must include the variables 'latitude'
            and 'longitude' and have exactly two dimensions.
        target_resolution: The target resolution in km (converted to degrees
            at the equator).
        has_channel_dimension: Whether the variables other than latitude
            and longitude have a trailing channel dimension.
        target_area: If provided, the target area
            definition to use for regridding. If None, a new area definition is created
            based on the target resolution and the extent of the swath.
        return_area: Whether to return the target area definition along with
            the regridded dataset.

    Returns:
        xr.Dataset: The regridded dataset.
        (DynamicAreaDefinition, optional): The target area definition,
            returned if return_area is True.
    """
    # Get the dimensions from the latitude variable
    dims = list(ds.latitude.dims)
    if len(dims) != 2:
        raise ValueError("Dataset must have exactly two dimensions")
    dim_y, dim_x = dims

    lon, lat = check_and_wrap(ds.longitude.values, ds.latitude.values)

    # Resampling can provoke errors for cases where the longitudes
    # cross the antimeridian. We have however the advantage that
    # we can assume that the longitudes here span over strictly less than 180°.
    # Therefore, if there are both negative and positive longitudes, they
    # either cross the 0° meridian or the 180° meridian.
    # - If they cross the 0° meridian, nothing to do it'll be handled correctly.
    # - If they cross the 180° meridian, we'll shift the longitudes to be all positive,
    #   then resample, then shift back.
    lons_were_shifted = False
    if np.any(lon < 0) and np.any(lon > 0):
        # Check which meridian is being crossed
        lon_span = lon.max() - lon.min()
        if lon_span > 180:
            # Crossing the 180° meridian
            lon = np.where(lon < 0, lon + 360, lon) - 180
            lons_were_shifted = True

    swath = SwathDefinition(lons=lon, lats=lat)
    radius_of_influence = 100000  # 100 km

    if target_area is None:
        # Use an equiangular (Plate Carrée) projection
        proj_dict = {
            "proj": "longlat",
            "a": EARTH_RADIUS,
        }

        # Convert target resolution from km to degrees
        # We calculate the meters per degree based on the Earth radius provided
        circumference = 2 * math.pi * EARTH_RADIUS
        meters_per_degree = circumference / 360.0
        res_degrees = (target_resolution * 1000) / meters_per_degree

        with DisableLogger():
            target_area_def = cast(
                DynamicAreaDefinition,
                create_area_def(
                    area_id="dynamic_equiangular",
                    projection=proj_dict,
                    resolution=res_degrees,
                    units="degrees",
                    shape=None,
                ),
            )

        # Create the grid lat/lon values from the swath
        target_area = target_area_def.freeze(swath, antimeridian_mode="modify_extents")

    # Retrieve the variables to regrid
    variables = [var for var in ds.variables if dim_y in ds[var].dims and dim_x in ds[var].dims]
    variables = [var for var in variables if var not in ["latitude", "longitude"]]
    ds = ds.reset_coords()[variables]

    # Create the resampler
    resampler = NumpyBilinearResampler(swath, target_area, radius_of_influence)
    # Individually resample each variable
    resampled_vars = {}
    for var in variables:
        try:
            resampled_vars[var] = resampler.resample(
                ds[var].values,
                fill_value=float("nan"),  # type: ignore
            )
        except Exception as e:
            print("Longitude:", lon)
            print("Latitude:", lat)
            traceback.print_tb(e.__traceback__)
            traceback.print_exc()
            raise ResamplingError(f"Error resampling variable {var}") from e

    # Rebuild the dataset
    if has_channel_dimension:
        result = {var: (("lat", "lon", "channel"), resampled_vars[var]) for var in variables}
    else:
        result = {var: (("lat", "lon"), resampled_vars[var]) for var in variables}

    # Add the latitude and longitude variables as coordinates
    lons, lats = target_area.get_lonlats()
    # Shift back longitudes if they were shifted
    if lons_were_shifted:
        lons += 180.0
        lons = np.where(lons > 180.0, lons - 360.0, lons)

    coords = {
        "latitude": (["lat", "lon"], lats),
        "longitude": (["lat", "lon"], lons),
    }
    # Rebuild the dataset
    result = xr.Dataset(result, coords=coords)
    if return_area:
        return result, target_area
    return result


def regrid_to_grid(
    ds: xr.Dataset, grid_lat: np.ndarray, grid_lon: np.ndarray, has_channel_dimension: bool = False
) -> xr.Dataset:
    """Regrids the dataset ds to a given target grid defined by its latitude
    and longitude arrays.

    Args:
        ds (xarray.Dataset): Dataset to regrid. Must include the variables 'latitude'
            and 'longitude' and have exactly two dimensions.
        grid_lat (numpy.ndarray): The latitude grid, of shape (H, W).
        grid_lon (numpy.ndarray): The longitude grid, of shape (H, W).
        has_channel_dimension (bool): Whether the variables other than latitude
            and longitude have a trailing channel dimension.

    Returns:
        xr.Dataset: The regridded dataset.
    """
    # Get the dimensions from the latitude variable
    dims = list(ds.latitude.dims)
    if len(dims) != 2:
        raise ValueError("Dataset must have exactly two dimensions")
    dim_y, dim_x = dims

    lon, lat = check_and_wrap(ds.longitude.values, ds.latitude.values)

    swath = SwathDefinition(lons=lon, lats=lat)
    radius_of_influence = 100000  # 100 km

    # Define the target area object for pyresample
    target_swath = SwathDefinition(lons=grid_lon, lats=grid_lat)

    # Retrieve the variables to regrid
    variables = [var for var in ds.variables if dim_y in ds[var].dims and dim_x in ds[var].dims]
    variables = [var for var in variables if var not in ["latitude", "longitude"]]
    ds = ds.reset_coords()[variables]

    # Individually resample each variable
    resampled_vars = {}
    for var in variables:
        try:
            resampler = ImageContainerNearest(
                ds[var].values,
                swath,
                radius_of_influence,
                fill_value=float("nan"),  # type: ignore
            )
            resampled_vars[var] = resampler.resample(
                target_swath,
            ).image_data
        except Exception as e:
            print("Longitude:", lon)
            print("Latitude:", lat)
            traceback.print_tb(e.__traceback__)
            traceback.print_exc()
            raise ResamplingError(f"Error resampling variable {var}") from e
    # Rebuild the dataset
    if has_channel_dimension:
        result = {var: ((dim_y, dim_x, "channel"), resampled_vars[var]) for var in variables}
    else:
        result = {var: ((dim_y, dim_x), resampled_vars[var]) for var in variables}

    # Add the latitude and longitude variables as coordinates
    coords = {
        "latitude": ((dim_y, dim_x), grid_lat),
        "longitude": ((dim_y, dim_x), grid_lon),
    }
    # Rebuild the dataset
    result = xr.Dataset(result, coords=coords)
    return result
