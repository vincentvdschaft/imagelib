"""Container for image data."""

import logging
from copy import deepcopy
from pathlib import Path

import h5py
import matplotlib
import matplotlib.image
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from imagelib.extent import Extent

SCALE_LINEAR = 0
SCALE_DB = 1


class Image:
    """Container for image data. Contains a 2D numpy array and metadata."""

    # Required for rmul with numpy arrays
    __array_priority__ = 1000

    def __init__(self, data, extent, scale=SCALE_LINEAR, metadata=None):
        """Initialilze Image object.

        Parameters
        ----------
        data : array_like
            2D numpy array containing image data (n_x, n_y).
        extent : array_like
            4-element array containing the extent of the image.
        scale : int
            The scale of the image data (SCALE_LINEAR or SCALE_DB).
        metadata : dict
            Additional metadata for the image.
        """
        self.extent = Extent(extent)
        self.data = data
        self.scale = scale

        self._metadata = {}
        if metadata is not None:
            self.update_metadata(metadata)

    def imshow(self, ax, *args, **kwargs):
        """Display image using matplotlib imshow."""
        extent = self.extent_imshow
        return ax.imshow(self.data.T, extent=extent, origin="lower", *args, **kwargs)

    @property
    def shape(self):
        """Return shape of image data."""
        return self.data.shape

    @property
    def data(self):
        """Return image data."""
        return np.copy(self._data)

    @data.setter
    def data(self, value):
        """Set image data."""
        if np.iscomplexobj(value):
            logging.warning("Image data is complex. Taking magnitude.")
            value = np.abs(value)
        data = np.array(value, dtype=np.float32)

        self._check_data(data)

        self._data = data

    def _check_data(self, data):
        if data.shape[0] < 1 or data.shape[1] < 1:
            raise ValueError("Data must be at least size 1 in both dimensions.")

        if data.shape[0] == 1 and self.extent.width != 0:
            raise ValueError("Extent width must be 0.")

        if data.shape[1] == 1 and self.extent.height != 0:
            raise ValueError("Extent height must be 0.")

        if data.ndim != 2:
            raise ValueError("Data must be 2D.")

    @property
    def extent(self):
        """Return extent of image."""
        return self._extent

    @extent.setter
    def extent(self, value):
        """Set extent of image."""
        self._extent = Extent(value).sort()

    @property
    def pixel_w(self):
        """The width of a pixel in the image. Returns 0 if the image has size one in the x-direction."""
        if self.shape[0] == 1:
            return 0.0
        return self.extent.width / (self.shape[0] - 1)

    @property
    def pixel_h(self):
        """The height of a pixel in the image. Returns 0 if the image has size one in the y-direction."""
        if self.shape[1] == 1:
            return 0.0
        return self.extent.height / (self.shape[1] - 1)

    @property
    def pixel_size(self):
        """The width and height of a pixel in the image."""
        return self.pixel_w, self.pixel_h

    @property
    def n_pixels(self):
        """The number of pixels in the image."""
        return self.shape[0] * self.shape[1]

    @property
    def scale(self):
        """Return whether image data is in dB or linear."""
        return self._scale

    @scale.setter
    def scale(self, value):
        """Set whether image data is in dB or linear."""
        self._scale = Image._parse_scale(value)

    def set_scale(self, value):
        """Returns a copy of the image with a new scale."""
        return Image(self.data, self.extent, scale=value, metadata=self.metadata)

    def set_data(self, data):
        """Returns a copy of the image with new data."""
        return Image(data, self.extent, scale=self.scale, metadata=self.metadata)

    def set_extent(self, extent):
        """Returns a copy of the image with new extent."""
        return Image(self.data, extent, scale=self.scale, metadata=self.metadata)

    def set_metadata(self, metadata):
        """Returns a copy of the image with new metadata."""
        return Image(self.data, self.extent, scale=self.scale, metadata=metadata)

    def __getitem__(self, idx):
        if isinstance(idx, (int, slice)):
            idx = (idx, slice(None, None, None))
        elif isinstance(idx, (tuple, list)):
            idx = tuple(idx)

        assert isinstance(idx, tuple) and len(idx) == 2
        # Index with integers
        if isinstance(idx[0], int) and isinstance(idx[1], int):
            return self.data[idx]

        slices = []
        for dim, slice_input in enumerate(idx):
            if isinstance(slice_input, int):
                new_slice = slice(slice_input, slice_input + 1)
            else:
                new_slice = slice(
                    slice_input.start if slice_input.start is not None else 0,
                    (
                        slice_input.stop
                        if slice_input.stop is not None
                        else self.shape[dim]
                    ),
                    slice_input.step,
                )
            slices.append(new_slice)

        # Index with slices
        data = self.data[tuple(slices)]
        extent = [
            self.extent[0] + slices[0].start * self.pixel_w,
            self.extent[0] + (slices[0].stop - 1) * self.pixel_w,
            self.extent[2] + slices[1].start * self.pixel_h,
            self.extent[2] + (slices[1].stop - 1) * self.pixel_h,
        ]
        return Image(data, extent=extent, scale=self.scale, metadata=self.metadata)

    @property
    def in_db(self):
        """Return whether image data is log-compressed."""
        return self.scale == SCALE_DB

    def save(self, path, cmap="gray"):
        """Save image to HDF5 file."""
        path = Path(path)
        if path.suffix in [".png", ".jpg", ".jpeg", ".bmp"]:
            matplotlib.image.imsave(path, self.data.T, cmap=cmap)
            return self
        assert path.suffix == ".hdf5", "File must be HDF5 format."

        save_hdf5_image(
            path=path,
            image=self.data,
            extent=self.extent,
            scale=self.scale,
            metadata=self.metadata,
        )
        return self

    @classmethod
    def load(cls, path):
        """Load image from HDF5 file."""
        return load_hdf5_image(path)

    def log_compress(self):
        """Log-compress image data."""
        if self.scale == SCALE_DB:
            logging.warning("Image data is already log-compressed. Skipping.")
            return self

        # Prevent taking the log of 0
        data = np.where(self.data > 0, self.data, 1e-12)
        data = 20 * np.log10(data)
        scale = SCALE_DB

        return Image(data, extent=self.extent, scale=scale, metadata=self.metadata)

    def log_expand(self):
        """Log-expand image data."""
        if self.scale == SCALE_LINEAR:
            logging.warning("Image data is already linear. Skipping.")
            return self

        data = np.power(10, self.data / 20)
        data = np.where(self.data <= -240, 0, data)
        scale = SCALE_LINEAR

        return Image(data, extent=self.extent, scale=scale, metadata=self.metadata)

    def normalize(self, normval=None):
        """Normalize image data to max 1 when not log-compressed or 0 when log-compressed."""

        if normval is None:
            normval = self.data.max()

        data = self.data
        if self.scale == SCALE_DB:
            data -= normval
        else:
            data /= normval

        return Image(data, extent=self.extent, scale=self.scale, metadata=self.metadata)

    def normalize_percentile(self, percentile=99):
        """Normalize image data to the given percentile value."""
        normval = np.percentile(self.data, percentile)
        return self.normalize(normval)

    def __repr__(self):
        """Return string representation of Image object."""
        shape = self.shape
        log_compressed_str = ", in dB" if self.in_db else ""
        return (
            f"Image(({shape[0], shape[1]}), extent={self.extent}{log_compressed_str})"
        )

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

    @property
    def x_vals(self):
        """Return x values of image."""
        return np.linspace(self.extent[0], self.extent[1], self.shape[0])

    @property
    def y_vals(self):
        """Return y values of image."""
        return np.linspace(self.extent[2], self.extent[3], self.shape[1])

    @property
    def grid(self):
        """Return grid of image of shape (n_x, n_y, 2)."""

        x_grid, y_grid = np.meshgrid(self.x_vals, self.y_vals, indexing="ij")
        return np.stack([x_grid, y_grid], axis=-1)

    @property
    def flatgrid(self):
        """Return flat grid of image."""
        return self.grid.reshape(-1, 2)

    @property
    def aspect(self):
        """Return aspect ratio of image."""
        return self.extent.aspect

    def match_histogram(self, other):
        """Match the histogram of the image to another image."""

        data = _match_histograms(self.data, other.data)
        return Image(data, extent=self.extent, scale=self.scale, metadata=self.metadata)

    @staticmethod
    def _parse_scale(val):
        """Parse scale value."""
        if val == SCALE_DB or val == SCALE_LINEAR:
            return val

        if isinstance(val, str):
            val = val.lower()
            if "lin" in val:
                return SCALE_LINEAR
            else:
                return SCALE_DB

        val = bool(val)
        return SCALE_DB if val else SCALE_LINEAR

    def max(self):
        """Find the maximum value in the image data."""
        return np.max(self.data)

    def min(self):
        """Find the minimum value in the image data."""
        return np.min(self.data)

    def clip(self, minval=None, maxval=None):
        """Clip the image data to a range."""
        data = np.clip(self.data, minval, maxval)
        return Image(data, extent=self.extent, scale=self.scale, metadata=self.metadata)

    def apply_fn(self, fn):
        """Apply a function to the image data."""
        self.data = fn(self.data)
        return self

    def map_range(self, minval, maxval, old_min=None, old_max=None):
        """Map the image data to a new range."""
        if old_min is None:
            old_min = self.min()
        if old_max is None:
            old_max = self.max()

        data = (self.data - old_min) / (old_max - old_min) * (maxval - minval) + minval
        return Image(data, extent=self.extent, scale=self.scale, metadata=self.metadata)

    def to_pixels(self):
        """Converts the image to linear pixel values [0, 1]."""
        return self.map_range(0.0, 1.0).set_scale(SCALE_LINEAR)

    def resample(self, shape, extent=None, method="linear"):
        """Resample image to a new shape."""

        if extent is None:
            extent = self.extent
        else:
            extent = Extent(extent)

        interpolator = RegularGridInterpolator(
            (self.x_vals, self.y_vals),
            self.data,
            bounds_error=False,
            fill_value=0 if self.scale == SCALE_LINEAR else -240,
            method=method,
        )
        new_xvals = np.linspace(extent[0], extent[1], shape[0])
        new_yvals = np.linspace(extent[2], extent[3], shape[1])

        x_grid, y_grid = np.meshgrid(new_xvals, new_yvals, indexing="ij")
        new_data = interpolator((x_grid, y_grid))

        return Image(
            new_data,
            extent=extent,
            scale=self.scale,
            metadata=self.metadata,
        )

    def resize(self, factor, method="linear"):
        """Resamples the image to a different resolution retaining the aspect ratio."""
        new_shape = [dim * factor for dim in self.shape]
        return self.resample(new_shape, extent=self.extent, method=method)

    def square_pixels(self):
        """Ensures that the pixels are square by changing the extent and resampling
        the image if necessary."""
        if self.pixel_w == self.pixel_h:
            return self

        if self.pixel_w < self.pixel_h:
            n_x_new = self.shape[0]
            n_y_new = int(self.extent.height / self.pixel_w)
            extent_new = self.extent.set_y1(
                self.extent.y0 + (n_y_new - 1) * self.pixel_w
            )
        else:
            n_x_new = int(self.extent.width / self.pixel_h)
            n_y_new = self.shape[1]
            extent_new = self.extent.set_x1(
                self.extent.x0 + (n_x_new - 1) * self.pixel_h
            )

        return self.resample((n_x_new, n_y_new), extent=extent_new)

    def transpose(self):
        """Transpose image data."""
        data = self.data.T
        extent = [self.extent[2], self.extent[3], self.extent[0], self.extent[1]]
        return Image(data, extent=extent, scale=self.scale, metadata=self.metadata)

    def xflip(self):
        """Flip image data along x-axis.

        Note
        ----
        The extent is NOT flipped. This is because the extent is defined as always
        being sorted.
        """
        data = np.flip(self.data, axis=0)
        return Image(data, extent=self.extent, scale=self.scale, metadata=self.metadata)

    def yflip(self):
        """Flip image data along y-axis.

        Note
        ----
        The extent is NOT flipped. This is because the extent is defined as always
        being sorted.
        """
        data = np.flip(self.data, axis=1)
        return Image(data, extent=self.extent, scale=self.scale, metadata=self.metadata)

    @property
    def extent_imshow(self):
        """Returns an extent corrected for how imshow works. It ensures that the
        gridpoints represent the center of the pixels."""
        return correct_imshow_extent(self.extent, self.shape)

    def copy(self):
        """Return a copy of the image."""
        return Image(
            self.data,
            extent=self.extent,
            scale=self.scale,
            metadata=deepcopy(self.metadata),
        )

    @classmethod
    def test_image(cls):
        """Create a test image."""
        n_x, n_y = 129, 129
        extent = (-30, 30, 0, 40)
        x, y = np.meshgrid(
            np.linspace(-1, 1, n_x), np.linspace(-1, 1, n_y), indexing="ij"
        )
        data = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) * (x**2)
        return cls(data, extent=extent)

    def __add__(self, other):
        """Add two images together."""
        if isinstance(other, (int, float, np.number)):
            data = self.data + other
            return Image(
                data, extent=self.extent, scale=self.scale, metadata=self.metadata
            )

        if isinstance(other, Image):
            assert all([e1 == e2 for e1, e2 in zip(self.extent, other.extent)])
            assert self.scale == other.scale
            data = self.data + other.data
            return Image(
                data, extent=self.extent, scale=self.scale, metadata=self.metadata
            )

        other = np.array(other)
        data = self.data + other
        return Image(data, extent=self.extent, scale=self.scale, metadata=self.metadata)

    def __mul__(self, other):
        """Multiply image."""
        if isinstance(other, Image):
            other = other.data

        data = self.data * other
        return Image(data, extent=self.extent, scale=self.scale, metadata=self.metadata)

    def __rmul__(self, other):
        """Multiply image."""
        return self.__mul__(other)

    def __truediv__(self, other):
        """Divide image."""
        if isinstance(other, Image):
            other = other.data

        data = self.data / other
        return Image(data, extent=self.extent, scale=self.scale, metadata=self.metadata)

    def __sub__(self, other):
        """Subtract two images."""
        return self + (other * -1)

    def __eq__(self, other):
        """Check if two images are equal."""
        if not isinstance(other, Image):
            return False

        if self.scale != other.scale:
            return False

        if self.extent != other.extent:
            return False

        return np.allclose(self.data, other.data)


def correct_imshow_extent(extent, shape):
    """Corrects the extent of an image to ensure the min and max
    extent are in the centers of the outer pixels.

    Parameters
    ----------
    extent :  Extent
        The extent of the image (x0, x1, y0, y1).
    shape : tuple of int
        The shape of the image (width, height).

    Returns
    -------
    extent : tuple of float
        The corrected extent of the image.
    """
    extent = Extent(extent).sort()
    width, height = extent.size
    pixel_w = width / (shape[0] - 1)
    pixel_h = height / (shape[1] - 1)

    offset = (
        -pixel_w / 2,
        pixel_w / 2,
        -pixel_h / 2,
        pixel_h / 2,
    )
    return Extent([ext + off for ext, off in zip(extent, offset)])


def save_hdf5_image(path, image, extent, scale=SCALE_LINEAR, metadata=None):
    """
    Saves an image to an hdf5 file.

    Parameters
    ----------
    path : str
        The path to the hdf5 file.
    image : np.ndarray
        The image to save.
    extent : list
        The extent of the image (x0, x1, z0, z1).
    scale : int
        The scale of the image (SCALE_LINEAR or SCALE_DB).
    metadata : dict
        Additional metadata to save.
    """

    extent = Extent(extent).sort()

    path = Path(path)

    if path.exists():
        logging.warning("Overwriting existing file %s", path)
        path.unlink()
    if not path.parent.exists():
        path.parent.mkdir(parents=True)

    if image.ndim > 2:
        image = np.squeeze(image)
        if image.ndim > 2:
            raise ValueError(
                f"Image must be 2D, but has shape {image.shape}. "
                f"Try using np.squeeze to remove extra dimensions."
            )

    with h5py.File(path, "w") as dataset:
        dataset.create_dataset("image", data=image)
        dataset["image"].attrs["extent"] = extent
        dataset["image"].attrs["scale"] = "linear" if scale == SCALE_LINEAR else "dB"
        if metadata is not None:
            save_dict_to_hdf5(dataset, metadata)


def load_hdf5_image(path):
    """
    Loads an image from an hdf5 file.

    Parameters
    ----------
    path : str
        The path to the hdf5 file.

    Returns
    -------
    image : np.ndarray
        The image.
    extent : np.ndarray
        The extent of the image (x0, x1, z0, z1).
    scale : int
        The scale of the image (SCALE_LINEAR or SCALE_DB).
    """

    with h5py.File(path, "r") as dataset:
        image = dataset["image"][()]
        extent = dataset["image"].attrs["extent"]
        scale = dataset["image"].attrs["scale"]
        metadata = load_hdf5_to_dict(dataset)
        metadata.pop("image", None)

    return Image(data=image, extent=extent, scale=scale, metadata=metadata)


def save_dict_to_hdf5(hdf5_file, data_dict, parent_group="/"):
    """
    Recursively saves a nested dictionary to an HDF5 file.

    Parameters
    ----------
    hdf5_file : h5py.File
        Opened h5py.File object.
    data_dict : dict
        (Nested) dictionary to save.
    parent_group : h5py.Group
        Current group path in HDF5 file (default is root "/").
    """
    data_dict = deepcopy(data_dict)
    data_dict = _lists_to_numbered_dict(data_dict)
    for key, value in data_dict.items():
        group_path = f"{parent_group}/{key}"
        if isinstance(value, dict):
            # Create a new group for nested dictionary
            hdf5_file.require_group(group_path)
            save_dict_to_hdf5(hdf5_file, value, parent_group=group_path)
        else:
            # Convert leaf items into datasets
            hdf5_file[group_path] = value


def _lists_to_numbered_dict(data_dict):
    """Transforms all lists in a dictionary to dictionaries with numbered keys."""
    for key, value in data_dict.items():
        if isinstance(value, list):
            data_dict[key] = {str(i).zfill(3): v for i, v in enumerate(value)}
        elif isinstance(value, dict):
            data_dict[key] = _lists_to_numbered_dict(value)
    return data_dict


def _is_numbered_dict(data_dict):
    keys = data_dict.keys()
    try:
        keys = [int(k) for k in keys]
    except ValueError:
        return False
    return set(keys) == set(range(len(keys)))


def _numbered_dicts_to_list(data_dict):
    """Transforms all dictionaries with numbered keys to lists."""
    for key, value in data_dict.items():
        if isinstance(value, dict):
            if _is_numbered_dict(value):
                data_dict[key] = [value[k] for k in sorted(value.keys(), key=int)]
            else:
                data_dict[key] = _numbered_dicts_to_list(value)
    return data_dict


def load_hdf5_to_dict(hdf5_file, parent_group="/"):
    """
    Recursively reads an HDF5 file into a nested dictionary.

    Parameters
    ----------
    hdf5_file : h5py.File
        Opened h5py.File object.
    parent_group : str
        Current group path in HDF5 file (default is root "/").

    Returns
    -------
        Nested dictionary representing the HDF5 file structure.
    """
    data_dict = {}
    for key in hdf5_file[parent_group]:
        item_path = f"{parent_group}/{key}"
        if isinstance(hdf5_file[item_path], h5py.Group):
            data_dict[key] = load_hdf5_to_dict(hdf5_file, parent_group=item_path)
        else:
            item = hdf5_file[item_path][()]
            if isinstance(item, bytes):
                item = item.decode("utf-8")
            # Convert scalar numpy arrays to Python scalars
            elif np.isscalar(item):
                item = item.item()

            data_dict[key] = item

    return _numbered_dicts_to_list(data_dict)


def _is_number(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def _match_histograms(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image.

    Parameters
    ----------
    source : np.ndarray
        Image to transform; the histogram is computed over the flattened array.
    template : np.ndarray
        Template image; can have different dimensions to source.
    Returns
    -------
    matched : np.ndarray
        The transformed output image.
    """
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # Get the set of unique pixel values and their corresponding indices and counts
    _, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # Calculate the empirical cumulative distribution functions (CDF) for the source and template images
    s_quantiles = np.cumsum(s_counts).astype(np.float64) / source.size
    t_quantiles = np.cumsum(t_counts).astype(np.float64) / template.size

    # Interpolate to find the pixel values in the template image that correspond to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)
