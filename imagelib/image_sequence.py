from pathlib import Path

import numpy as np

from imagelib.extent import Extent
from imagelib.image import Image


class ImageSequence:
    """Container class to hold a sequence of images."""

    def __init__(self, images):
        self.images = images

    @property
    def images(self):
        """Get the list of images."""
        return self._images

    @images.setter
    def images(self, value):
        """Set the list of images."""
        if isinstance(value, Image):
            value = [value]
        if not all(isinstance(obj, Image) for obj in value):
            raise TypeError("All objects must be of type Image.")
        if not all([im.extent == value[0].extent for im in value]):
            raise ValueError("All images must have the same extent.")
        self._images = list(value)

    @property
    def data(self):
        """Get the data cube of all images."""
        return np.stack([im.data for im in self.images], axis=0)

    @property
    def extent(self):
        """Get the extent."""
        return self.images[0].extent

    def __getitem__(self, idx):
        """Get image from list."""
        if isinstance(idx, slice):
            return ImageSequence(self.images[idx])

        return self.images[idx]

    def append(self, im):
        """Append image to list."""
        assert isinstance(im, Image)
        self.images.append(im)

    def save(self, directory, name):
        """Save image sequence to disk."""
        # Remove file extension if it exists
        suffix = Path(name).suffix
        if suffix == "":
            suffix = ".hdf5"
        name = str(Path(name).with_suffix(""))
        for n, im in enumerate(self.images):
            save_name = f"{name}_{str(n).zfill(5)}{suffix}"
            im.save(Path(directory) / save_name)

        return self

    @staticmethod
    def load(paths):
        """Load image sequence from disk.

        Parameters
        ----------
        paths : Path or [Path]
            The path of a directory to load all images from or a list of specific paths.

        """
        if isinstance(paths, (Path, str)):
            paths = Path(paths)
            paths = list(paths.glob("*.hdf5"))
            paths.sort()
            paths = [Path(p) for p in paths]

        assert isinstance(paths, list)

        images = []
        for path in paths:
            images.append(Image.load(path))

        return ImageSequence(images)

    @staticmethod
    def from_numpy(data, extent):
        assert data.ndim == 3
        n_frames = data.shape[0]
        images = []
        for n in range(n_frames):
            images.append(Image(data[n], extent=extent))

        return ImageSequence(images)

    def to_numpy(self):
        """Convert image sequence to numpy array."""
        return np.stack([im.data for im in self.images], axis=0)

    def log_compress(self):
        return ImageSequence([im.log_compress() for im in self.images])

    def normalize(self, normval=None):
        """Normalize the image values.

        Parameters
        ----------
        normval : float, optional
            The value to normalize the images to. If None, the maximum value across all
            images is used.
        """
        if normval is None:
            normval = self.max()
        return ImageSequence([im.normalize(normval) for im in self.images])

    def normalize_percentile(self, percentile=99):
        data = self.to_numpy()
        normval = np.percentile(data, percentile)
        return ImageSequence(
            [im.normalize(normval) for im in self.images]
        )
    
    def to_pixels(self):
        """Convert the extent of all images to pixel coordinates."""
        return ImageSequence([im.to_pixels() for im in self.images])

    def match_histogram(self, match_image):
        """Match the histogram of the images to another image."""
        assert isinstance(match_image, Image), "match_image must be an Image."
        return ImageSequence([im.match_histogram(match_image) for im in self.images])

    def clip(self, minval=None, maxval=None):
        """Clip the image values."""
        return ImageSequence([im.clip(minval, maxval) for im in self.images])

    def max(self):
        """Get the maximum pixel value across all images."""
        return max([im.max() for im in self.images])

    def min(self):
        """Get the minimum pixel value across all images."""
        return min([im.min() for im in self.images])

    def transpose(self):
        """Transpose all images."""
        return ImageSequence([im.transpose() for im in self.images])

    def xflip(self):
        """Flip all images in the x direction."""
        return ImageSequence([im.xflip() for im in self.images])

    def yflip(self):
        """Flip all images in the y direction."""
        return ImageSequence([im.yflip() for im in self.images])

    def max_image(self):
        """Construct an image where every pixel is the maximum across all images."""
        data = self.to_numpy()
        data = np.max(data, axis=0)
        return Image(data, extent=self.extent)

    def min_image(self):
        """Construct an image where every pixel is the minimum across all images."""
        data = self.to_numpy()
        data = np.min(data, axis=0)
        return Image(data, extent=self.extent)

    def mean_image(self):
        """Construct an image where every pixel is the mean across all images."""
        data = self.to_numpy()
        data = np.mean(data, axis=0)
        return Image(data, extent=self.extent)

    def __iter__(self):
        return iter(self.images)

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return f"ImageSequence({len(self.images)} images)"

    def __add__(self, value):
        """Add value to images."""
        if isinstance(value, (int, float, np.number, np.ndarray)):
            return ImageSequence([im + value for im in self.images])
        raise ValueError("Invalid type for addition.")

    def __sub__(self, value):
        """Subtract value from images."""
        if isinstance(value, (int, float, np.number, np.ndarray)):
            return ImageSequence([im - value for im in self.images])
        raise ValueError("Invalid type for subtraction.")

    def map(self, func):
        """Apply a function to each image in the sequence."""
        list(map(lambda im: Image.apply_fn(im, func), self.images))
        return self
