from pathlib import Path

import numpy as np

from imagelib.extent import Extent
from imagelib.image import SCALE_DB, SCALE_LINEAR, Image


class ImageSequence:
    """Container class to hold a sequence of images."""

    def __init__(self, images):
        self.images = images

    @property
    def images(self):
        """Get the list of images."""
        return self._images

    @property
    def data(self):
        """Get the data cube of all images."""
        return np.stack([im.data for im in self.images], axis=0)

    @images.setter
    def images(self, value):
        assert all(isinstance(obj, Image) for obj in value)
        self._images = list(value)

    @property
    def extent(self):
        """Get the extent."""
        return self.images[0].extent

    @property
    def scale(self):
        """Get the scale (SCALE_LINEAR or SCALE_DB)."""
        return self.images[0].scale

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
        # Remove file extension if it exists
        name = str(Path(name).with_suffix(""))
        for n, im in enumerate(self.images):
            im.save(Path(directory) / f"{name}_{str(n).zfill(5)}.hdf5")

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
    def from_numpy(data, extent, scale=SCALE_LINEAR):
        assert data.ndim == 3
        n_frames = data.shape[0]
        images = []
        for n in range(n_frames):
            images.append(Image(data[n], extent=extent, scale=scale))

        return ImageSequence(images)

    def log_compress(self):
        return ImageSequence([im.log_compress() for im in self.images])

    def normalize(self, normval=None):
        if normval is None:
            normval = self.max()
        return ImageSequence([im.normalize(normval) for im in self.images])

    def normalize_percentile(self, percentile=99):
        return ImageSequence(
            [im.normalize_percentile(percentile) for im in self.images]
        )

    def match_histogram(self, other):
        return ImageSequence([im.match_histogram(other) for im in self.images])

    def clip(self, minval=None, maxval=None):
        return ImageSequence([im.clip(minval, maxval) for im in self.images])

    def max(self):
        return max([im.max() for im in self.images])

    def min(self):
        return min([im.min() for im in self.images])

    def transpose(self):
        return ImageSequence([im.transpose() for im in self.images])

    def xflip(self):
        return ImageSequence([im.xflip() for im in self.images])

    def yflip(self):
        return ImageSequence([im.yflip() for im in self.images])

    def __iter__(self):
        return iter(self.images)

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return f"ImageSequence({len(self.images)} images)"

    def map(self, func):
        """Apply a function to each image in the sequence."""
        list(map(lambda im: Image.apply_fn(im, func), self.images))
        return self
