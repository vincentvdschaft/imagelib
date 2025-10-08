from pathlib import Path

import matplotlib.image
import numpy as np

from .ndextent import Extent
from .saving import load_hdf5_image, save_hdf5_image


class NDImage:
    def __init__(self, array, extent, metadata=None):
        extent = Extent(extent).sort()
        _check_ndimage_initializers(array, extent)
        self.array = np.asarray(array)
        self.extent = extent
        self._metadata = {}
        if metadata is not None:
            self.update_metadata(metadata)

    @property
    def extent(self):
        return Extent(self._extent)

    @extent.setter
    def extent(self, value):
        self._extent = Extent(value).sort()

    def __repr__(self):
        return f"NDImage(array={self.shape}, extent={self.extent!r})"

    # ==========================================================================
    # Numpy array interface
    # ==========================================================================
    # --- 1️⃣ Allow automatic conversion to numpy array ---
    def __array__(self, dtype=None):
        return np.asarray(self.array, dtype=dtype)

    # --- 2️⃣ Intercept numpy ufuncs like +, -, *, sin, etc. ---
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # unwrap NDImage objects to their arrays
        unwrapped = []
        for inp in inputs:
            if isinstance(inp, NDImage):
                unwrapped.append(inp.array)
            else:
                unwrapped.append(inp)

        # apply the ufunc
        result = getattr(ufunc, method)(*unwrapped, **kwargs)

        # if result is an ndarray (not scalar), wrap it back
        if isinstance(result, np.ndarray):
            return NDImage(result, self.extent, metadata=self.metadata)
        else:
            return result

    # --- 3️⃣ Optional: handle numpy high-level functions like np.concatenate ---
    def __array_function__(self, func, types, args, kwargs):
        # if NDImage not supported, fall back to ndarray
        if not all(issubclass(t, NDImage) for t in types):
            return NotImplemented

        if func is np.concatenate:
            arrays = [a.array for a in args[0]]
            new_array = np.concatenate(arrays, **kwargs)
            return NDImage(new_array, self.extent, metadata=self.metadata)
        return NotImplemented

    # ==========================================================================
    # Sizes and dimensions
    # ==========================================================================
    def pixel_size(self, dim):
        """Returns the pixel size in the given dimension."""
        return (
            self.extent.dim_size(dim) / (self.shape[dim] - 1)
            if self.shape[dim] > 1
            else 0.0
        )

    @property
    def pixel_width(self):
        """Returns the pixel width."""
        return self.pixel_size(0)

    @property
    def pixel_height(self):
        """Returns the pixel height."""
        return self.pixel_size(1)

    # ==========================================================================
    # Forward properties
    # ==========================================================================
    @property
    def shape(self):
        return self.array.shape

    @property
    def ndim(self):
        return self.array.ndim

    @property
    def size(self):
        return self.array.size

    @property
    def dtype(self):
        return self.array.dtype

    # ==========================================================================
    # Metadata handling
    # ==========================================================================

    @property
    def metadata(self):
        """Return metadata of image."""
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        """Set metadata of image."""
        assert isinstance(value, dict), "Metadata must be a dictionary."
        self._metadata = value

    def add_metadata(self, key, value):
        """Add metadata to image."""
        self._metadata[key] = value
        return self

    def update_metadata(self, metadata):
        """Update metadata of image."""
        self._metadata.update(metadata)
        return self

    def append_metadata(self, key, value):
        """Add metadata assuming the key is a list."""

        if key not in self.metadata:
            self._metadata[key] = []
        elif not isinstance(self.metadata[key], list):
            raise ValueError(f"Metadata key {key} is not a list.")

        self._metadata[key].append(value)
        return self

    def clear_metadata(self):
        """Clear metadata of image."""
        self.metadata = {}

    # ==========================================================================
    # Grids
    # ==========================================================================
    def vals(self, dim):
        """Returns the coordinate values along the given dimension."""
        return np.linspace(
            self.extent.start(dim), self.extent.end(dim), self.shape[dim]
        )

    def x_vals(self):
        """Returns the x coordinate values."""
        return self.vals(0)

    def y_vals(self):
        """Returns the y coordinate values."""
        return self.vals(1)

    def z_vals(self):
        """Returns the z coordinate values."""
        return self.vals(2)

    @property
    def grid(self):
        """Returns the coordinate grid."""
        return np.stack(
            np.meshgrid(*[self.vals(dim) for dim in range(self.ndim)], indexing="ij"),
            axis=-1,
        )

    @property
    def flatgrid(self):
        """Returns the flattened coordinate grid."""
        return self.grid.reshape(-1, self.ndim)

    def __getitem__(self, key):
        """Slicing the image."""
        new_grid = self.grid[key]
        new_array = self.array[key]

        new_extent_initializer = []
        for dim in range(self.ndim):
            minval = np.min(new_grid[..., dim])
            maxval = np.max(new_grid[..., dim])
            new_extent_initializer.append(minval)
            new_extent_initializer.append(maxval)

        return NDImage(
            new_array, Extent(new_extent_initializer).sort(), metadata=self.metadata
        )

    # ==========================================================================
    # Functions
    # ==========================================================================
    def save(self, path, cmap="gray"):
        """Save image to HDF5 file."""
        path = Path(path)
        if path.suffix in [".png", ".jpg", ".jpeg", ".bmp"]:
            matplotlib.image.imsave(path, self.array.T, cmap=cmap)
            return self
        assert path.suffix == ".hdf5", "File must be HDF5 format."

        save_hdf5_image(
            path=path,
            array=self.array,
            extent=self.extent,
            metadata=self.metadata,
        )
        return self

    @classmethod
    def load(cls, path):
        """Load image from HDF5 file."""
        path = Path(path)
        assert path.suffix == ".hdf5", "File must be HDF5 format."
        return load_hdf5_image(path)

    def get_window(self, extent: Extent):
        """Returns a new image that contains only the pixels in the window.

        Parameters
        ----------
        extent : Extent
            The extent of the window to extract (in the image units, not pixel indices).
        """
        extent = Extent(extent)

        slices = []

        for dim in range(self.ndim):
            limits = []
            for limit in (extent.start(dim), extent.end(dim)):
                index = int(
                    np.ceil(
                        (limit - self.extent.start(dim))
                        / self.extent.dim_size(dim)
                        * (self.shape[dim] - 1)
                    )
                )
                limit = np.clip(index, 0, self.shape[dim] - 1)
                limits.append(limit)
            slices.append(slice(*limits))

        return self[tuple(slices)]


# ==============================================================================
# Helper functions
# ==============================================================================
def _check_ndimage_initializers(array, extent: Extent):
    assert array.ndim == extent.ndim, (
        "The array and extent must have the same number of dimensions. "
        f"Got {array.ndim} and {extent.ndim}."
    )
    for dim in range(array.ndim):
        if array.shape[dim] == 1:
            assert extent.dim_size(dim) == 0.0, (
                "Dimensions with one element must have zero size in the extent"
            )
