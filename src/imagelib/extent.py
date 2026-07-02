"""Limits classes for storing image spatial extents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Union

import numpy as np


@dataclass
class Limits:
    min: float
    max: float

    def __post_init__(self):
        self.min = float(self.min)
        self.max = float(self.max)
        true_min = min(self.min, self.max)
        true_max = max(self.min, self.max)
        self.min = true_min
        self.max = true_max

    def size(self) -> float:
        return self.max - self.min

    def __iter__(self):
        yield self.min
        yield self.max

    def __repr__(self):
        return f"Limits({self.min}, {self.max})"

    def __hash__(self):
        return hash((self.min, self.max))


LimitsLike = Union["Limits", tuple[float, float], Sequence[float]]
LimitsNDInput = Union[
    "LimitsND",
    Sequence[LimitsLike],  # tuple of tuples / list of Limits
    Sequence[float],  # flat tuple
    np.ndarray,  # (N, 2) or flat (2N,)
]


@dataclass
class LimitsND:
    limits: list[Limits] = field(default_factory=list)

    def __post_init__(self):
        raw = self.limits.limits if isinstance(self.limits, LimitsND) else self.limits

        # already a LimitsND-shaped list of Limits objects
        if all(isinstance(item, Limits) for item in raw):
            self.limits = list(raw)
            return

        arr = np.asarray(raw, dtype=float)

        if arr.ndim == 1:
            if arr.size % 2 != 0:
                raise ValueError(
                    f"Flat input must have an even number of elements, got {arr.size}"
                )
            arr = arr.reshape(-1, 2)
        elif arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"Expected shape (N, 2) or flat (2N,), got {arr.shape}")

        self.limits = [Limits(lo, hi) for lo, hi in arr]

    @property
    def ndim(self) -> int:
        return len(self.limits)

    def __getitem__(self, index) -> Limits:
        return self.limits[index]

    def sizes(self) -> np.ndarray:
        return np.array([limit.size() for limit in self.limits])

    @classmethod
    def from_extent(cls, extent: Extent) -> LimitsND:
        """Create a LimitsND object from a legacy Extent object."""
        limits = []
        for dim in range(extent.ndim):
            limits.append(Limits(extent.start(dim), extent.end(dim)))
        return cls(limits)

    @classmethod
    def from_shape(cls, shape: tuple[int, ...]) -> LimitsND:
        """Create a LimitsND object from a shape tuple, where each dimension's limits are (0, size-1)."""
        limits = []
        for size in shape:
            limits.append(Limits(0, size - 1))
        return cls(limits)

    def __iter__(self):
        return iter(self.limits)

    def __len__(self):
        return len(self.limits)

    def origin(self) -> np.ndarray:
        """Return the origin (min values) of the limits as a numpy array."""
        return np.array([limit.min for limit in self.limits])

    def __hash__(self):
        return hash(tuple(self))

    def make_grid(self, shape: tuple[int, ...]) -> tuple[np.ndarray, ...]:
        """Create a meshgrid of coordinates for the given shape, using the limits.

        Returns a grid of shape (*shape, ndim), where the last dimension contains the coordinates for each axis.

        If the limits represent (z, y, x), then the shape should be (nz, ny, nx), and the output will be (nz, ny, nx, zyx).
        """
        if len(shape) != self.ndim:
            raise ValueError(
                f"Shape length {len(shape)} does not match number of dimensions {self.ndim}"
            )
        grids = []
        for dim, (limit, size) in enumerate(zip(self.limits, shape)):
            grids.append(np.linspace(limit.min, limit.max, size))
        return np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1)


class Extent(tuple):
    """Legacy flat-tuple encoding of spatial limits: (dim0_min, dim0_max, dim1_min, dim1_max, ...).

    Kept only to decode HDF5 files written before the switch to LimitsND.
    """

    def __new__(cls, initializer):
        initializer = [float(value) for value in initializer]
        assert len(initializer) % 2 == 0, "Extent must have an even number of elements."

        return super(Extent, cls).__new__(cls, initializer)

    @property
    def ndim(self):
        return len(self) // 2

    def sort(self):
        """Sorts the extent such that dim0_min < dim0_max, dim1_min < dim1_max, ..."""
        initializer = []
        for val0, val1 in zip(self[::2], self[1::2]):
            initializer.append(min(val0, val1))
            initializer.append(max(val0, val1))

        return Extent(initializer)

    def start(self, dim):
        """Returns the start value of the given dimension."""
        return self[dim * 2]

    def end(self, dim):
        """Returns the end value of the given dimension."""
        return self[dim * 2 + 1]

    def __hash__(self):
        return hash(tuple(self))


def compute_limits_after_slicing(current_shape, limits: LimitsND, key) -> LimitsND:
    """Compute the new LimitsND after applying a numpy-style index key.

    Handles slices (including ``:``), integers (dimension removal),
    ``None`` / ``np.newaxis`` (new axis insertion), and ``Ellipsis``.
    """
    key = _expand_ellipsis(key, limits.ndim)

    new_limits = []
    original_dim = 0

    for key_element in key:
        if key_element is None:
            new_limits.append(Limits(0, 0))  # new axis carries no spatial meaning
        elif isinstance(key_element, int):
            original_dim += 1  # integer indexing removes this dimension
        else:
            indices = range(current_shape[original_dim])[key_element]
            new_limits.append(
                _limits_for_selected_indices(
                    limits[original_dim], current_shape[original_dim], indices
                )
            )
            original_dim += 1

    return LimitsND(new_limits)


def select_axis_values_after_slicing(values, key, fill) -> tuple:
    """Reorder a per-axis sequence to match a numpy-style index key.

    `values` has one entry per array dimension. Integer indexing drops the
    corresponding entry, ``None`` / ``np.newaxis`` inserts a new entry equal to
    ``fill``, and slices keep the entry unchanged.
    """
    key = _expand_ellipsis(key, len(values))
    result = []
    original_dim = 0
    for key_element in key:
        if key_element is None:
            result.append(fill)
        elif isinstance(key_element, int):
            original_dim += 1
        else:
            result.append(values[original_dim])
            original_dim += 1
    return tuple(result)


def _limits_for_selected_indices(dim_limits: Limits, num_pixels, indices) -> Limits:
    """Return the Limits spanning a sequence of selected pixel indices."""
    if len(indices) == 0:
        return Limits(dim_limits.min, dim_limits.min)
    return Limits(
        _index_to_coordinate(dim_limits, num_pixels, indices[0]),
        _index_to_coordinate(dim_limits, num_pixels, indices[-1]),
    )


def _expand_ellipsis(key, ndim):
    """Expand Ellipsis to explicit slice(None) objects.

    None (np.newaxis) items are preserved and do NOT count as consuming an array
    dimension — only slices and integers consume dimensions.
    """
    if not isinstance(key, tuple):
        key = (key,)

    n_consuming = sum(1 for k in key if k is not None and k is not Ellipsis)

    if Ellipsis not in key:
        return key + (slice(None),) * (ndim - n_consuming)

    ellipsis_pos = key.index(Ellipsis)
    n_missing = ndim - n_consuming
    return key[:ellipsis_pos] + (slice(None),) * n_missing + key[ellipsis_pos + 1 :]


def _index_to_coordinate(dim_limits: Limits, num_pixels, pixel_index) -> float:
    """Convert a pixel index to its physical coordinate (linear interpolation)."""
    if num_pixels <= 1:
        return dim_limits.min
    return dim_limits.min + pixel_index * dim_limits.size() / (num_pixels - 1)
