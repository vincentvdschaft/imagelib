"""Extent class for storing image extents."""

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
        raw = self.limits

        # already a LimitsND-shaped list of Limits objects
        if all(isinstance(item, Limits) for item in raw):
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
        """Create a LimitsND object from an Extent object."""
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

    def to_extent(self) -> Extent:
        """Convert the LimitsND object back to an Extent object."""
        extent_coords = []
        for limit in self.limits:
            extent_coords.append(limit.min)
            extent_coords.append(limit.max)
        return Extent(extent_coords)

    def __iter__(self):
        return iter(self.limits)

    def __len__(self):
        return len(self.limits)

    def origin(self) -> np.ndarray:
        """Return the origin (min values) of the limits as a numpy array."""
        return np.array([limit.min for limit in self.limits])

    def __hash__(self):
        return hash(tuple(self))


class Extent(tuple):
    """Wrapper class for extent data.
    Stores an extent as a tuple of floats (x0, x1, y0, y1).
    Since a tuple is immutable, this class is immutable. All functions that return an
    extent return a new extent.
    """

    def __new__(cls, initializer):
        initializer = [float(value) for value in initializer]
        assert len(initializer) % 2 == 0, "Extent must have an even number of elements."

        return super(Extent, cls).__new__(cls, initializer)

    def flip(self, dim):
        """Flip the extent along the given dimension.

        Parameters
        ----------
        dim : int
            The dimension to flip. 0 for x, 1 for y, etc...

        Returns
        -------
        Extent
            The flipped extent.
        """
        initializer = list(self)
        dim0 = dim * 2
        dim1 = dim0 + 1
        initializer[dim0], initializer[dim1] = initializer[dim1], initializer[dim0]
        return Extent(initializer)

    @property
    def ndim(self):
        return len(self) // 2

    def yflip(self):
        """Flip the y components of the extent."""
        return self.flip(1)

    def xflip(self):
        """Flip the x components of the extent."""
        return self.flip(0)

    def sort(self):
        """Sorts the extent such that x0 < x1 and y0 < y1."""
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

    @property
    def x0(self):
        """Returns the x0 value of the extent."""
        return self[0]

    @property
    def x1(self):
        """Returns the x1 value of the extent."""
        return self[1]

    @property
    def y0(self):
        """Returns the y0 value of the extent."""
        assert len(self) >= 4
        return self[2]

    @property
    def y1(self):
        """Returns the y1 value of the extent."""
        assert len(self) >= 4
        return self[3]

    def dim_size(self, dim):
        """Returns the size of the given dimension."""
        self_sorted = self.sort()
        return self_sorted[dim * 2 + 1] - self_sorted[dim * 2]

    @property
    def dim_sizes(self):
        """Returns (width, height) of the extent."""
        return [self.dim_size(n) for n in range(self.ndim)]

    @property
    def dim_scales(self):
        """Returns the dimension sizes where zeros are replaced with ones to avoid division by zero."""
        return [size if size != 0 else 1 for size in self.dim_sizes]

    @property
    def origin(self):
        """Returns the origin (x0, y0, ...) of the extent."""
        self_sorted = self.sort()
        return tuple(self_sorted[dim * 2] for dim in range(self.ndim))

    @property
    def width(self):
        """Returns the width of the extent."""
        return self.dim_size(0)

    @property
    def height(self):
        """Returns the height of the extent."""
        return self.dim_size(1)

    @property
    def aspect(self):
        """Returnss the aspect ratio (width/height)."""
        assert self.ndim == 2
        return self.height / self.width

    @property
    def xlims(self):
        """Returns (x0, x1) of the extent."""
        return self[0], self[1]

    @property
    def xlims_flipped(self):
        """Returns (x1, x0) of the extent."""
        return self[1], self[0]

    @property
    def ylims(self):
        """Returns (y0, y1) of the extent."""
        return self[2], self[3]

    @property
    def ylims_flipped(self):
        """Returns (y1, y0) of the extent."""
        return self[3], self[2]

    @property
    def zlims(self):
        """Returns (z0, z1) of the extent."""
        assert self.ndim >= 3
        return self[4], self[5]

    @property
    def zlims_flipped(self):
        """Returns (z1, z0) of the extent."""
        assert self.ndim >= 3
        return self[5], self[4]

    @property
    def origin(self):
        """Returns the origin (x0, y0, ...) of the extent."""
        return tuple(self.start(dim) for dim in range(self.ndim))

    def __mul__(self, value):
        return Extent(element * value for element in self)

    def __rmul__(self, value):
        return self * value

    def __truediv__(self, value):
        return Extent(element / value for element in self)

    def __floordiv__(self, value):
        return Extent(element // value for element in self)

    def __add__(self, value):
        return Extent(element + value for element in self)

    def expand(self, points):
        """Expand the extent to include the given points.

        Parameters
        ----------
        points : np.ndarray
            An array of shape (N, 2) where N is the number of points.

        Returns
        -------
        new_extent : Extent
            A new extent that includes the given points.
        """
        assert isinstance(points, np.ndarray), "Points must be a numpy array."
        points = points.reshape(-1, 2)
        x0 = min(points[:, 0].min(), self[0])
        x1 = max(points[:, 0].max(), self[1])
        y0 = min(points[:, 1].min(), self[2])
        y1 = max(points[:, 1].max(), self[3])
        return Extent(x0, x1, y0, y1)

    def __sub__(self, value):
        if _is_number(value):
            return self + (-value)
        return self + (-value[0], -value[1])

    def __hash__(self):
        return hash(tuple(self))

    @classmethod
    def from_bbox(cls, x, y, width, height):
        """Create an extent from a bounding box.

        Parameters
        ----------
        x : float
            The x-coordinate of the bottom-left corner.
        y : float
            The y-coordinate of the bottom-left corner.
        width : float
            The width of the bounding box.
        height : float
            The height of the bounding box.

        Returns
        -------
        Extent
            The created extent.
        """
        return cls((x, x + width, y, y + height))


def _is_number(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def compute_extent_after_slicing(current_shape, extent: Extent, key) -> Extent:
    """Compute the new spatial extent after applying a numpy-style index key.

    Handles slices (including ``:``) , integers (dimension removal),
    ``None`` / ``np.newaxis`` (new axis insertion), and ``Ellipsis``.
    """
    key = _expand_ellipsis(key, extent.ndim)

    new_extent_coords = []
    original_dim = 0

    for key_element in key:
        if key_element is None:
            new_extent_coords.extend([0, 0])  # new axis carries no spatial meaning
        elif isinstance(key_element, int):
            original_dim += 1  # integer indexing removes this dimension
        else:
            indices = range(current_shape[original_dim])[key_element]
            new_extent_coords.extend(
                _extent_coords_for_selected_indices(
                    extent.start(original_dim),
                    extent.end(original_dim),
                    current_shape[original_dim],
                    indices,
                )
            )
            original_dim += 1

    return Extent(new_extent_coords).sort()


def _extent_coords_for_selected_indices(dim_start, dim_end, num_pixels, indices):
    """Return [coord_start, coord_end] for a sequence of selected pixel indices."""
    if len(indices) == 0:
        return [dim_start, dim_start]
    return [
        _index_to_coordinate(dim_start, dim_end, num_pixels, indices[0]),
        _index_to_coordinate(dim_start, dim_end, num_pixels, indices[-1]),
    ]


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


def _index_to_coordinate(dim_start, dim_end, num_pixels, pixel_index):
    """Convert a pixel index to its physical coordinate (linear interpolation)."""
    if num_pixels <= 1:
        return dim_start
    return dim_start + pixel_index * (dim_end - dim_start) / (num_pixels - 1)
