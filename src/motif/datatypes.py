from dataclasses import dataclass
from typing import Dict, List, Tuple

from torch import Tensor


# Sources within samples are identified by a tuple of (source_name, source_index)
# If a source S appears in multiple samples in a batch, each occurrence is identified
# by a different source_index (0 being the most recent).
@dataclass
class SourceIndex:
    name: str
    index: int

    def __iter__(self):
        return iter((self.name, self.index))

    def __getitem__(self, key):
        if key == 0:
            return self.name
        elif key == 1:
            return self.index
        else:
            raise IndexError("SourceIndex only has two elements: name and index.")

    def __hash__(self) -> int:
        return hash((self.name, self.index))


@dataclass
class RawSourceData:
    """Represents the data for a single source as output by the dataset,
    before being collated into a batch."""

    # ... represents the spatial dimensions
    values: Tensor  # (B, C, ...)
    coords: Tensor  # (B, 2, ...)
    dt: Tensor  # (B,)
    avail_mask: Tensor  # (B, ...)
    dist_to_center: Tensor  # (B, ...)
    landmask: Tensor  # (B, ...)
    characs: Tensor | None  # (B, characs_dim) or None


@dataclass
class SourceData(RawSourceData):
    """Represents the data for a single source after being collated into a batch,
    but before being preprocessed by the lightning module."""

    avail: Tensor  # (B,)
    characs: Tensor | None  # (B, characs_dim) or None
    diffusion_t: Tensor | None = None  # (B, diffusion_dim) or None
    pred_mean: Tensor | None = None  # (B, C, ...) or None

    def clone(self) -> "SourceData":
        """Returns a copy of this SourceData with all tensors cloned."""
        return SourceData(
            values=self.values.clone(),
            coords=self.coords.clone(),
            dt=self.dt.clone(),
            avail=self.avail.clone(),
            avail_mask=self.avail_mask.clone(),
            dist_to_center=self.dist_to_center.clone(),
            landmask=self.landmask.clone(),
            characs=self.characs.clone() if self.characs is not None else None,
            diffusion_t=self.diffusion_t.clone() if self.diffusion_t is not None else None,
            pred_mean=self.pred_mean.clone() if self.pred_mean is not None else None,
        )

    def shallow_clone(self) -> "SourceData":
        """Returns a copy of this SourceData with the same references to tensors."""
        return SourceData(
            values=self.values,
            coords=self.coords,
            dt=self.dt,
            avail=self.avail,
            avail_mask=self.avail_mask,
            dist_to_center=self.dist_to_center,
            landmask=self.landmask,
            characs=self.characs,
            diffusion_t=self.diffusion_t,
            pred_mean=self.pred_mean,
        )


@dataclass
class SourceEmbedding:
    """Represents the data of a source after being processed by the
    embedding layers."""

    embedding: Tensor  # (B, h, w, dim)
    coords: Tensor  # (B, h, w, coords_dim)
    conditioning: Tensor | None  # (B, h, w, cond_dim) or None


# A batch of data from multiple sources, indexed by SourceIndex
Batch = Dict[SourceIndex, SourceData]
# A batch of data from multiple sources, but with also include the index
# of each sample in the batch within the dataset.
BatchWithSampleIndexes = Tuple[List[int], Dict[SourceIndex, SourceData]]

# Same as above, but after preprocessing by the lightning module.
# Actually the same type, but helps prevent confusion between the two stages of the data pipeline.
PreprocessedBatch = Dict[SourceIndex, SourceData]

# A dict of source embeddings
SourceEmbeddingDict = Dict[SourceIndex, SourceEmbedding]

# A dict of one tensor per source, used for the output of torch layers.
MultisourceTensor = Dict[SourceIndex, Tensor]


@dataclass
class Prediction:
    """Represents the model's predictions, i.e. the output of the predict_step() method."""

    # pred is the final output of the model for each source (including non-masked sources).
    # It can have different shapes depending on the model architecture:
    # - Deterministic models: (B, C, ...) where ... are the spatial dimensions
    # - Flow matching: (R, T, B, C, ...) where R is the number of realizations,
    #      and T is the number of time steps.
    pred: MultisourceTensor  # (B, C, ...) or (R, B, C, ...) or (R, T, B, C, ...)
    avail: MultisourceTensor  # (B, ...)


@dataclass
class GenerativePrediction(Prediction):
    """Represents the model's predictions for a flow matching model."""

    time_grid: Tensor  # (T,) or None
    pred_mean: MultisourceTensor | None = None  # (B, C, ...) or None
