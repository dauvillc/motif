"""Implements a customized collate_fn that receives samples from a MultiSourceDataset."""

from typing import Dict, List, Tuple

import torch
from torch.nn.functional import pad

from motif.datatypes import BatchWithSampleIndexes, RawSourceData, SourceData, SourceIndex


def maximum_of_shapes(shape_A: Tuple[int, ...], shape_B: Tuple[int, ...]) -> Tuple[int, ...]:
    """Returns the maximum shape dim per dim between shape_A and shape_B."""
    if len(shape_A) == 0:
        return shape_B
    if len(shape_B) == 0:
        return shape_A
    return tuple(max(a, b) for a, b in zip(shape_A, shape_B))


def multi_source_collate_fn(
    indices_samples: List[Tuple[int, Dict[SourceIndex, RawSourceData]]],
) -> BatchWithSampleIndexes:
    """Collates multiple samples from a MultiSourceDataset, which returns pairs
    (sample_index, sample).
    Adds an entry "avail" to each source, which is a tensor of shape (B,) whose value
    is 1 if the source is present and -1 otherwise (0 is kept to mean masked sources).

    A given source may be present in some samples but not in others. Padding will be applied
    to fill in the missing sources. Besides, for a given source the spatial shape may differ
    across samples; again padding will be applied to match the maximum shape across samples.

    Args:
        indices_samples (list): list of pairs (sample_index, sample).
    Returns:
        indices: A list of the sample indices.
        samples: A dictionary D such that D[(source_name, index)] contains the information
            for all samples for the source source_name with observation index index,
            with the entries described above.
    """
    # Separate indices and samples
    indices = [idx for idx, _ in indices_samples]
    samples = [sample for _, sample in indices_samples]

    # Step 1: Collect all unique SourceIndex keys across all samples
    all_source_indices = set()
    for sample in samples:
        all_source_indices.update(sample.keys())

    # Step 2: For each source, find the maximum spatial shape across all samples
    max_spatial_shapes: Dict[SourceIndex, Tuple[int, ...]] = {}
    for src_idx in all_source_indices:
        max_shape: Tuple[int, ...] = ()
        for sample in samples:
            if src_idx in sample:
                # Get spatial shape from values (skip channel dimension)
                spatial_shape = sample[src_idx].values.shape[1:]
                max_shape = maximum_of_shapes(max_shape, spatial_shape)
        max_spatial_shapes[src_idx] = max_shape

    # Step 3: Build batched tensors for each source
    batch_dict: Dict[SourceIndex, SourceData] = {}

    for src_idx in all_source_indices:
        max_spatial_shape = max_spatial_shapes[src_idx]

        # Lists to collect tensors from all samples
        values_list = []
        coords_list = []
        dt_list = []
        avail_mask_list = []
        dist_to_center_list = []
        landmask_list = []
        characs_list = []
        avail_list = []

        # We need to find the source's dtype, device, number of channels
        # and number of characs if any.
        ref_data: RawSourceData | None = None
        # Browse the samples until finding one that contains the source
        # (there must be at least one, otherwise we wouldn't be in this loop)
        for sample in samples:
            if src_idx in sample:
                ref_data = sample[src_idx]
                break
        assert ref_data is not None
        ref_dtype = ref_data.values.dtype
        ref_device = ref_data.values.device
        ref_n_channels = ref_data.values.shape[0]
        ref_n_characs = ref_data.characs.shape[0] if ref_data.characs is not None else None

        # Iterate through samples to build batch
        for sample in samples:
            if src_idx in sample:
                # Source is present in this sample
                raw_data = sample[src_idx]

                # Pad spatial fields to max_spatial_shape
                values_padded = _pad_spatial(raw_data.values, max_spatial_shape)
                coords_padded = _pad_spatial(raw_data.coords, max_spatial_shape)
                dist_to_center_padded = _pad_spatial(raw_data.dist_to_center, max_spatial_shape)
                landmask_padded = _pad_spatial(raw_data.landmask, max_spatial_shape)
                avail_mask_padded = _pad_spatial(
                    raw_data.avail_mask, max_spatial_shape, fill_value=-1.0
                )

                values_list.append(values_padded)
                coords_list.append(coords_padded)
                dt_list.append(raw_data.dt)
                avail_mask_list.append(avail_mask_padded)
                dist_to_center_list.append(dist_to_center_padded)
                landmask_list.append(landmask_padded)
                characs_list.append(raw_data.characs)
                avail_list.append(torch.tensor(1.0, dtype=torch.float32))

            else:
                # Source is missing in this sample - create full NaN tensors
                # values has shape (C, ...), where C is the channel dimension
                n_channels = ref_n_channels
                values_shape = (n_channels,) + max_spatial_shape
                values_nan = torch.full(
                    values_shape,
                    float("nan"),
                    dtype=ref_dtype,
                    device=ref_device,
                )

                # coords has shape (2, ...)
                coords_shape = (2,) + max_spatial_shape
                coords_nan = torch.full(
                    coords_shape,
                    float("nan"),
                    dtype=ref_dtype,
                    device=ref_device,
                )

                # Spatial fields have shape (...)
                spatial_nan = torch.full(
                    max_spatial_shape,
                    float("nan"),
                    dtype=ref_dtype,
                    device=ref_device,
                )

                # The availability mask is not filled with NaNs but with -1, which already
                # indicates missing values anyway.
                avail_mask_nan = torch.full(
                    max_spatial_shape,
                    -1.0,
                    dtype=torch.float32,
                    device=ref_device,
                )

                # dt is a scalar
                dt_nan = torch.tensor(
                    float("nan"),
                    dtype=ref_dtype,
                    device=ref_device,
                )

                # characs has shape (characs_dim,) if characs are present, else None
                if ref_n_characs is not None:
                    characs_nan = torch.full(
                        (ref_n_characs,),
                        float("nan"),
                        dtype=ref_dtype,
                        device=ref_device,
                    )
                else:
                    characs_nan = None

                values_list.append(values_nan)
                coords_list.append(coords_nan)
                dt_list.append(dt_nan)
                avail_mask_list.append(avail_mask_nan)
                dist_to_center_list.append(spatial_nan)
                landmask_list.append(spatial_nan.clone())
                characs_list.append(characs_nan)
                avail_list.append(torch.tensor(-1.0, dtype=torch.float32))

        # Stack all tensors along batch dimension
        values_batch = torch.stack(values_list, dim=0)
        coords_batch = torch.stack(coords_list, dim=0)
        dt_batch = torch.stack(dt_list, dim=0)
        avail_mask_batch = torch.stack(avail_mask_list, dim=0)
        dist_to_center_batch = torch.stack(dist_to_center_list, dim=0)
        landmask_batch = torch.stack(landmask_list, dim=0)
        avail_batch = torch.stack(avail_list, dim=0)

        # Handle characs: if there are characs, stack the tensors; otherwise keep as None
        if ref_n_characs is not None:
            characs_batch = torch.stack(characs_list, dim=0)
        else:
            characs_batch = None

        # Create SourceData object
        batch_dict[src_idx] = SourceData(
            values=values_batch,
            coords=coords_batch,
            dt=dt_batch,
            avail=avail_batch,
            avail_mask=avail_mask_batch,
            dist_to_center=dist_to_center_batch,
            landmask=landmask_batch,
            characs=characs_batch,
        )

    return indices, batch_dict


def _pad_spatial(
    tensor: torch.Tensor, target_spatial_shape: Tuple[int, ...], fill_value: float = float("nan")
) -> torch.Tensor:
    """Pads the spatial dimensions of a tensor to match target_spatial_shape.

    The tensor can have leading non-spatial dimensions (e.g., channel dimension).
    Only the trailing dimensions matching the rank of target_spatial_shape are padded.

    Args:
        tensor: Input tensor with shape (..., *spatial_dims).
        target_spatial_shape: Target shape for the spatial dimensions.
        fill_value: Value to use for padding. Default is NaN.

    Returns:
        Padded tensor with shape (..., *target_spatial_shape).
    """
    # Determine how many dimensions are spatial
    spatial_rank = len(target_spatial_shape)
    current_spatial_shape = tensor.shape[-spatial_rank:]

    # Check if padding is needed
    if current_spatial_shape == target_spatial_shape:
        return tensor

    # Build padding tuple in reversed order (torch.nn.functional.pad convention)
    # Format: (last_dim_left, last_dim_right, ..., first_dim_left, first_dim_right)
    pad_tuple = []
    for current_size, target_size in zip(
        reversed(current_spatial_shape), reversed(target_spatial_shape)
    ):
        pad_tuple.append(0)  # left padding
        pad_tuple.append(target_size - current_size)  # right padding

    return pad(tensor, pad_tuple, value=fill_value)
