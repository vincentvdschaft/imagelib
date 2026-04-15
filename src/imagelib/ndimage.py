from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.image
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import uniform_filter1d

from .clahe import clahe
from .dynamic_range import apply_dynamic_range_curve
from .extent import Extent, compute_extent_after_slicing
from .match_histograms import match_histograms
from .saving import load_hdf5_image, save_hdf5_image


class NDImage:
    def __init__(self, array, extent=None, metadata=None):
        if extent is None:
            extent = self._extent_from_array(array)
        extent = Extent(extent).sort()
        _check_ndimage_initializers(array, extent)
        self.array = np.asarray(array).copy()
        self.array.setflags(write=True)
        self._extent = extent
        self._metadata = {}
        if metadata is not None:
            self.update_metadata(metadata)

    @staticmethod
    def _extent_from_array(array) -> Extent:
        ndim = array.ndim
        extent_initializer = []
        for dim in range(ndim):
            extent_initializer.append(0)
            extent_initializer.append(array.shape[dim] - 1)
        return Extent(extent_initializer)

    @property
    def extent(self) -> Extent:
        return Extent(self._extent)

    def __repr__(self) -> str:
        return f"NDImage(array={self.shape}, extent={self.extent!r})"

    # ==========================================================================
    # Numpy array interface
    # ==========================================================================
    # --- 1️⃣ Allow automatic conversion to numpy array ---
    def __array__(self, dtype=None) -> np.ndarray:
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
    @property
    def size(self) -> int:
        """Returns the total number of pixels in the image."""
        return self.array.size

    def pixel_size(self, dim) -> float:
        """Returns the pixel size in the given dimension."""
        return (
            self.extent.dim_size(dim) / (self.shape[dim] - 1)
            if self.shape[dim] > 1
            else 0.0
        )

    @property
    def pixel_sizes(self) -> np.ndarray:
        return compute_pixel_sizes(self.extent, self.shape)

    @property
    def pixel_width(self) -> float:
        """Returns the pixel width."""
        return self.pixel_size(0)

    @property
    def pixel_height(self) -> float:
        """Returns the pixel height."""
        return self.pixel_size(1)

    @property
    def extent_imshow(self) -> Extent:
        """Returns the extent adjusted for imshow."""
        return correct_extent_for_imshow(self.extent, self.shape)

    # ==========================================================================
    # Forward properties
    # ==========================================================================
    @property
    def shape(self) -> tuple:
        return self.array.shape

    @property
    def ndim(self) -> int:
        return self.array.ndim

    @property
    def dtype(self) -> np.dtype:
        return self.array.dtype

    @property
    def T(self) -> NDImage:
        """Returns the transposed image."""
        axes = list(reversed(range(self.ndim)))
        return self.transpose(axes)

    # ==========================================================================
    # Metadata handling
    # ==========================================================================

    @property
    def metadata(self) -> dict:
        """Return metadata of image."""
        return self._metadata

    @metadata.setter
    def metadata(self, value) -> None:
        """Set metadata of image."""
        assert isinstance(value, dict), "Metadata must be a dictionary."
        self._metadata = value

    def add_metadata(self, key, value) -> NDImage:
        """Add metadata to image."""
        self._metadata[key] = value
        return self

    def update_metadata(self, metadata) -> NDImage:
        """Update metadata of image."""
        self._metadata.update(metadata)
        return self

    def append_metadata(self, key, value) -> NDImage:
        """Add metadata assuming the key is a list."""

        if key not in self.metadata:
            self._metadata[key] = []
        elif not isinstance(self.metadata[key], list):
            raise ValueError(f"Metadata key {key} is not a list.")

        self._metadata[key].append(value)
        return self

    def clear_metadata(self) -> NDImage:
        """Clear metadata of image."""
        self.metadata = {}
        return self

    # ==========================================================================
    # Grids
    # ==========================================================================
    def vals(self, dim) -> np.ndarray:
        """Returns the coordinate values along the given dimension."""
        return np.linspace(
            self.extent.start(dim), self.extent.end(dim), self.shape[dim]
        )

    def x_vals(self) -> np.ndarray:
        """Returns the x coordinate values."""
        return self.vals(0)

    def y_vals(self) -> np.ndarray:
        """Returns the y coordinate values."""
        return self.vals(1)

    def z_vals(self) -> np.ndarray:
        """Returns the z coordinate values."""
        return self.vals(2)

    @property
    def grid(self) -> np.ndarray:
        """Returns the coordinate grid."""
        return np.stack(
            np.meshgrid(*[self.vals(dim) for dim in range(self.ndim)], indexing="ij"),
            axis=-1,
        )

    @property
    def flatgrid(self) -> np.ndarray:
        """Returns the flattened coordinate grid."""
        return self.grid.reshape(-1, self.ndim)

    def __getitem__(self, key) -> NDImage:
        """Slicing the image."""
        new_array = self.array[key]
        new_extent = compute_extent_after_slicing(self.shape, self.extent, key)
        return NDImage(new_array, new_extent, metadata=self.metadata)

    def __setitem__(self, key, value) -> None:
        self.array[key] = value

    # ==========================================================================
    # Functions
    # ==========================================================================
    def save(self, path, cmap="gray", vmin=None, vmax=None) -> NDImage:
        """Save image to HDF5 file."""
        path = Path(path)
        if path.suffix in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
            matplotlib.image.imsave(path, self.array.T, cmap=cmap, vmin=vmin, vmax=vmax)
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
    def load(cls, path, indices=slice(None)) -> NDImage:
        """Load image from HDF5 file."""
        path = Path(path)
        assert path.suffix == ".hdf5", "File must be HDF5 format."
        return load_hdf5_image(path, indices=indices)

    def with_array(self, array) -> NDImage:
        return NDImage(array, extent=self.extent, metadata=self.metadata)

    def with_extent(self, extent) -> NDImage:
        return NDImage(self.array, extent=extent, metadata=self.metadata)

    def map_range(self, new_min, new_max, old_min=None, old_max=None) -> NDImage:
        """Map the image values to a new range [new_min, new_max]."""
        if old_min is None:
            old_min = np.min(self.array)
        if old_max is None:
            old_max = np.max(self.array)
        scaled = (self.array - old_min) / (old_max - old_min)
        mapped = scaled * (new_max - new_min) + new_min
        return NDImage(mapped, self.extent, metadata=self.metadata)

    def to_pixels(self) -> NDImage:
        """Convert the image to pixel values in the range [0, 1]."""
        return self.map_range(0, 1)

    def clip(self, min=None, max=None) -> NDImage:
        """Clip the image values to the given range."""
        return NDImage(
            np.clip(self.array, min, max), self.extent, metadata=self.metadata
        )

    def resample(self, shape, extent=None, method="linear", fill_value=0) -> NDImage:
        """Resample image to a new shape."""

        if extent is None:
            extent = self.extent
        else:
            extent = Extent(extent)

        all_vals = [self.vals(dim) for dim in range(self.ndim)]

        interpolator = RegularGridInterpolator(
            all_vals,
            self.array,
            bounds_error=False,
            fill_value=fill_value,
            method=method,
        )
        new_all_vals = [
            np.linspace(extent.start(dim), extent.end(dim), shape[dim])
            for dim in range(len(shape))
        ]

        new_data = interpolator(np.meshgrid(*new_all_vals, indexing="ij")).reshape(
            shape
        )

        return NDImage(
            new_data,
            extent=extent,
            metadata=self.metadata,
        )

    def transpose(self, axes=None) -> NDImage:
        """Transpose the image."""
        if axes is None:
            axes = list(reversed(range(self.ndim)))
        new_array = np.transpose(self.array, axes)
        new_extent_initializer = []
        for axis in axes:
            new_extent_initializer.append(self.extent.start(axis))
            new_extent_initializer.append(self.extent.end(axis))
        new_extent = Extent(new_extent_initializer)
        return NDImage(new_array, extent=new_extent, metadata=self.metadata)

    def square_pixels(self) -> NDImage:
        nonzero_pixel_sizes = [
            size for size in filter(lambda size: size > 0, self.pixel_sizes)
        ]
        new_pixel_size = min(nonzero_pixel_sizes)

        new_extent_initializer = []
        new_shape = []
        for dim in range(self.ndim):
            new_n_pixels = int(self.extent.dim_size(dim) / new_pixel_size) + 1
            new_shape.append(new_n_pixels)

            new_extent_initializer.append(self.extent.start(dim))
            new_extent_initializer.append(
                self.extent.start(dim) + (new_n_pixels - 1) * new_pixel_size
            )
        new_extent = Extent(new_extent_initializer)
        return self.resample(shape=new_shape, extent=new_extent, method="nearest")

    def resample_scale(self, factor, axes=None) -> NDImage:
        """Scale the image by a given factor."""
        factors = [
            factor if (axes is None or dim in axes) else 1 for dim in range(self.ndim)
        ]
        new_shape = [
            int(dim_size * factor_for_dim)
            for dim_size, factor_for_dim in zip(self.shape, factors)
        ]
        return self.resample(shape=new_shape, extent=self.extent, method="linear")

    def get_window(self, extent: Extent) -> NDImage:
        """Returns a new image that contains only the pixels in the window.

        Parameters
        ----------
        extent : Extent
            The extent of the window to extract (in the image units, not pixel indices).
        """
        extent = Extent(extent)

        assert extent.ndim == self.ndim, (
            "The extent must have the same number of dimensions as the image."
            f"Got {extent.ndim} and {self.ndim}."
        )

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
                limit = np.clip(index, 0, self.shape[dim])
                limits.append(limit)
            slices.append(slice(*limits))

        return self[tuple(slices)]

    def match_histogram(self, other) -> NDImage:
        """Match the histogram of the image to another image."""

        array = match_histograms(self.array, other.array)
        return NDImage(array, extent=self.extent, metadata=self.metadata)

    def apply_dynamic_range_curve(self, curve: np.ndarray) -> NDImage:
        """Apply a dynamic range curve to the image data."""
        data = apply_dynamic_range_curve(curve, self.array)
        return NDImage(data, extent=self.extent, metadata=self.metadata)

    def log_compress(self) -> NDImage:
        """Log-compress image data with 20*log10(image)."""

        data = np.where(self.array > 0, self.array, 1e-12)
        data = 20 * np.log10(data)

        return NDImage(data, extent=self.extent, metadata=self.metadata)

    def log_expand(self) -> NDImage:
        """Log-expand image data."""

        array = np.power(10, self.array / 20)
        array = np.where(self.array <= -240, 0, array)

        return self.with_array(array)

    def symlog_compress(self, threshold=1.0) -> NDImage:
        """Symmetric log-compress image data."""

        data = np.where(
            np.abs(self.array) > threshold,
            np.sign(self.array)
            * (threshold + np.log10(np.abs(self.array) / threshold)),
            self.array,
        )

        return NDImage(data, extent=self.extent, metadata=self.metadata)

    def symlog_expand(self, threshold=1.0) -> NDImage:
        """Symmetric log-expand image data."""

        array = np.where(
            np.abs(self.array) > threshold,
            np.sign(self.array)
            * threshold
            * np.power(10, (np.abs(self.array) - threshold) / threshold),
            self.array,
        )

        return self.with_array(array)

    def abs(self) -> NDImage:
        """Returns the absolute value of the image."""
        return self.with_array(np.abs(self.array))

    def normalize(self, normval=None) -> NDImage:
        """Normalize image data by dividing by the max or normval."""

        if normval is None:
            normval = self.array.max()

        if normval == 0.0:
            warnings.warn("Warning: normval is 0. Returning original image.")
            return self

        return self / normval

    def normalize_db(self, normval=None) -> NDImage:
        """Normalize image data by adding the max or normval."""
        if normval is None:
            normval = self.array.max()

        return self - np.array(normval)

    def normalize_percentile(self, percentile=99) -> NDImage:
        """Normalize image data to the given percentile value."""
        normval = np.percentile(self.array, percentile)
        return self.normalize(normval)

    def copy(self) -> NDImage:
        """Returns a copy of the image."""
        return NDImage(self.array.copy(), self.extent, metadata=self.metadata.copy())

    def max(self, **kwargs) -> float:
        """Returns the maximum value of the image."""
        return np.max(self.array, **kwargs)

    def min(self, **kwargs) -> float:
        """Returns the minimum value of the image."""
        return np.min(self.array, **kwargs)

    def mean(self, **kwargs) -> float:
        """Returns the mean value of the image."""
        return np.mean(self.array, **kwargs)

    def xflip(self) -> NDImage:
        """Returns a copy of the image flipped in the x direction."""
        return self[::-1, :]

    def yflip(self) -> NDImage:
        """Returns a copy of the image flipped in the y direction."""
        return self[:, ::-1]

    def fft(self, axes=None) -> NDImage:
        """Returns the FFT of the image.

        The spectrum is shifted so that the zero frequency component is in the center
        of the spectrum.

        The extent is updated to reflect the spatial frequency range."""
        data = np.fft.fftn(self.array, axes=axes)

        new_extent = list(self.extent)

        for axis in axes or range(self.ndim):
            data = np.fft.fftshift(data, axes=axis)
            spatial_sampling_interval = self.pixel_size(axis)
            spatial_freqs = np.fft.fftfreq(
                self.shape[axis], d=spatial_sampling_interval
            )

            new_extent[2 * axis] = np.min(spatial_freqs)
            new_extent[2 * axis + 1] = np.max(spatial_freqs)

        return NDImage(data, extent=new_extent, metadata=self.metadata)

    def moving_average(self, ax, window_size) -> NDImage:
        """Apply a moving average filter along the given axis."""
        smoothed = uniform_filter1d(
            self.array, size=window_size, axis=ax, mode="constant", cval=0
        )
        return self.with_array(smoothed)

    def normalize_moving_average(self, ax, window_size, eps=1e-6) -> NDImage:
        """Computes the moving average along the given axis and normalizes the image by dividing by the moving average."""

        moving_avg = np.abs(self.moving_average(ax, window_size).array) + eps

        all_axes = tuple(set(range(self.ndim)) - set([ax % self.ndim]))
        moving_avg = np.mean(moving_avg, axis=all_axes)
        dummy_dim_tuple = [None] * self.ndim
        dummy_dim_tuple[ax] = slice(None)
        moving_avg = moving_avg[tuple(dummy_dim_tuple)]
        normalized = np.where(moving_avg > 0, self.array / moving_avg, 0)
        return self.with_array(normalized)

    def sample(self, positions) -> np.ndarray:
        """Sample image values at the given spatial positions without interpolation.

        Parameters
        ----------
        positions : np.ndarray of shape (N, D)
            Spatial coordinates to sample. D must match the image dimensionality.

        Returns
        -------
        np.ndarray of shape (N,)
            Image values at the nearest pixel for each position.
        """
        indices = self.coordinates_to_indices(positions)
        return self.array[tuple(indices[:, dim] for dim in range(self.ndim))]

    def coordinates_to_indices(self, coordinates) -> np.ndarray:
        """Convert coordinates to pixel indices."""
        assert coordinates.ndim == 2
        assert coordinates.shape[1] == self.ndim
        indices_total = []
        for dim in range(self.ndim):
            indices = (coordinates[:, dim] - self.extent.start(dim)) / self.pixel_size(
                dim
            )
            indices_rounded = np.round(indices).astype(int)
            indices_rounded = np.clip(indices_rounded, 0, self.shape[dim] - 1)
            indices_total.append(indices_rounded)
        return np.stack(indices_total, axis=-1)

    def clahe(
        self,
        clip_limit: float = 0.01,
        tile_grid_size: tuple = (8, 8),
        axes: tuple = (0, 1),
    ) -> NDImage:
        """Apply CLAHE to the image."""
        if self.ndim < 2:
            raise ValueError("CLAHE requires at least 2D images.")
        data = clahe(self.array, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
        return NDImage(data, extent=self.extent, metadata=self.metadata)

    # ==========================================================================
    # Dunder methods
    # ==========================================================================

    def __add__(self, other) -> NDImage:
        """Add two images together."""
        if isinstance(other, (int, float, np.number)):
            data = self.array + other
            return NDImage(data, extent=self.extent, metadata=self.metadata)

        if isinstance(other, NDImage):
            assert all([e1 == e2 for e1, e2 in zip(self.extent, other.extent)])
            data = self.array + other.array
            return NDImage(data, extent=self.extent, metadata=self.metadata)

        other = np.array(other)
        data = self.array + other
        return NDImage(data, extent=self.extent, metadata=self.metadata)

    def __mul__(self, other) -> NDImage:
        """Multiply image."""
        if isinstance(other, NDImage):
            other = other.array

        data = self.array * other
        return NDImage(data, extent=self.extent, metadata=self.metadata)

    def __rmul__(self, other) -> NDImage:
        """Multiply image."""
        return self.__mul__(other)

    def __truediv__(self, other):
        """Divide image."""
        if isinstance(other, NDImage):
            other = other.array

        data = self.array / other
        return NDImage(data, extent=self.extent, metadata=self.metadata)

    def __sub__(self, other) -> NDImage:
        """Subtract two images."""
        return self + (other * -1)

    def __eq__(self, other) -> bool:
        """Check if two images are equal."""
        if not isinstance(other, NDImage):
            return False

        if self.extent != other.extent:
            return False

        return np.allclose(self.array, other.array)

    @classmethod
    def test_image(cls) -> NDImage:
        """Returns a test image."""

        extent = Extent((-10, 0, 0, 20))
        x_vals = np.linspace(extent.start(0), extent.end(0), 128)
        y_vals = np.linspace(extent.start(1), extent.end(1), 256)
        x_grid, y_grid = np.meshgrid(x_vals, y_vals, indexing="ij")
        array = (
            np.exp(-((x_grid + 5) ** 2 + (y_grid - 10) ** 2) / 20)
            * np.sin(2 * np.pi * x_grid)
            * np.cos(2 * np.pi * y_grid * 2)
        )
        return NDImage(array, extent=extent)

    @classmethod
    def from_png(cls, path, extent=None) -> NDImage:
        """Load image from PNG file."""
        array = np.mean(matplotlib.image.imread(path), axis=2).T  # convert to grayscale
        if extent is None:
            extent = Extent((0, array.shape[0] - 1, 0, array.shape[1] - 1))
        return NDImage(array, extent=extent)


# ==============================================================================
# Helper functions
# ==============================================================================
def _check_ndimage_initializers(array, extent: Extent):
    assert array.ndim == extent.ndim, (
        "The array and extent must have the same number of dimensions. "
        f"Got {array.ndim} and {extent.ndim}."
    )
    for dim in range(array.ndim):
        if array.shape[dim] == 1 and not extent.dim_size(dim) == 0.0:
            raise ValueError(
                "Dimensions with one element must have zero size in the extent"
            )


def correct_extent_for_imshow(extent: Extent, shape):
    new_initializer = []
    for dim in range(extent.ndim):
        pixel_size = extent.dim_size(dim) / (shape[dim] - 1) if shape[dim] > 1 else 0.0
        new_initializer.append(extent.start(dim) - pixel_size / 2)
        new_initializer.append(extent.end(dim) + pixel_size / 2)
    return Extent(new_initializer)


def compute_pixel_sizes(extent: Extent, shape):
    pixel_sizes = []
    for dim in range(extent.ndim):
        pixel_size = extent.dim_size(dim) / (shape[dim] - 1) if shape[dim] > 1 else 0.0
        pixel_sizes.append(pixel_size)
    return pixel_sizes
