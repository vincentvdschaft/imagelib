"""Extent class for storing image extents."""

import numpy as np


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
    def size(self):
        """Returnss (width, height) of the extent."""
        return [self.dim_size(n) for n in range(self.ndim)]

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
