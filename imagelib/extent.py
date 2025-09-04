"""Extent class for storing image extents."""

import numpy as np


class Extent(tuple):
    """Wrapper class for extent data.
    Stores an extent as a tuple of floats (x0, x1, y0, y1).
    Since a tuple is immutable, this class is immutable. All functions that return an
    extent return a new extent.
    """

    def __new__(cls, *args, **kwargs):
        if kwargs:
            initializer = [kwargs["x0"], kwargs["x1"], kwargs["y0"], kwargs["y1"]]
        elif len(args) == 4:
            initializer = [
                float(args[0]),
                float(args[1]),
                float(args[2]),
                float(args[3]),
            ]
        # Reduce numpy arrays and such to list of floats
        elif len(args) == 1:
            initializer = [float(x) for x in args[0]]
            assert len(initializer) == 4
        else:
            raise ValueError("Extent must have 4 elements.")

        return super(Extent, cls).__new__(cls, initializer)

    def yflip(self):
        """Flip the y components of the extent."""
        return Extent(self[0], self[1], self[3], self[2])

    def xflip(self):
        """Flip the x components of the extent."""
        return Extent(self[1], self[0], self[2], self[3])

    def sort(self):
        """Sorts the extent such that x0 < x1 and y0 < y1."""
        x0 = min(self[0], self[1])
        x1 = max(self[0], self[1])
        y1 = max(self[2], self[3])
        y0 = min(self[2], self[3])
        return Extent(x0, x1, y0, y1)

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
        return self[2]

    @property
    def y1(self):
        """Returns the y1 value of the extent."""
        return self[3]

    @property
    def width(self):
        """Returns the width of the extent."""
        self_sorted = self.sort()
        return self_sorted[1] - self_sorted[0]

    @property
    def height(self):
        """Returns the height of the extent."""
        self_sorted = self.sort()
        return self_sorted[3] - self_sorted[2]

    @property
    def size(self):
        """Returnss (width, height) of the extent."""
        return self.width, self.height

    @property
    def aspect(self):
        """Returnss the aspect ratio (width/height)."""
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

    def set_x0(self, value):
        """Set the x0 value of the extent."""
        return Extent(value, self[1], self[2], self[3])

    def set_x1(self, value):
        """Set the x1 value of the extent."""
        return Extent(self[0], value, self[2], self[3])

    def set_y0(self, value):
        """Set the y0 value of the extent"""
        return Extent(self[0], self[1], value, self[3])

    def set_y1(self, value):
        """Set the y1 value of the extent."""
        return Extent(self[0], self[1], self[2], value)

    def __mul__(self, value):
        return Extent(
            self[0] * value, self[1] * value, self[2] * value, self[3] * value
        )

    def __rmul__(self, value):
        return self * value

    def __truediv__(self, value):
        return Extent(
            self[0] / value, self[1] / value, self[2] / value, self[3] / value
        )

    def __floordiv__(self, value):
        return Extent(
            self[0] // value, self[1] // value, self[2] // value, self[3] // value
        )

    def __add__(self, value):
        if _is_number(value):
            value = float(value)
            return Extent(
                self[0] + value, self[1] + value, self[2] + value, self[3] + value
            )

        try:
            x, y = value[0], value[1]
        except (TypeError, IndexError) as e:
            raise ValueError(
                "Extent can only be added to a number or a 2-tuple."
            ) from e
        return Extent(x + self[0], x + self[1], y + self[2], y + self[3])

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
        return cls(x, x + width, y, y + height)


def _is_number(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False
