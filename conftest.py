import numpy as np
from pytest import fixture

from imagelib import Extent, Image, ImageSequence


@fixture
def fixture_image_data():
    """Produces a random 2D array."""
    return np.random.rand(100, 100)


@fixture
def fixture_extent():
    """Produces an extent object."""
    return Extent(-1, 1, 0, 3)


@fixture
def fixture_metadata():
    """Produces a metadata object."""
    return {
        "name": "test",
        "date": "2020-01-01",
        "n_samples": 3,
        "names": ["a", "b", "c"],
    }


@fixture
def fixture_extent_tuple():
    """Produces a size 4 tuple."""
    return (-1, 1, 0, 3)


@fixture
def fixture_image():
    """Produces an Image object."""
    return Image(data=np.random.rand(100, 100), extent=(-1, 1, 0, 3))


@fixture
def fixture_image_with_metadata(fixture_image_data, fixture_extent, fixture_metadata):
    """Produces an Image object with metadata."""
    return Image(
        data=fixture_image_data, extent=fixture_extent, metadata=fixture_metadata
    )


@fixture
def fixture_list_of_images(fixture_image_data, fixture_extent):
    """Produces a list of Image objects."""
    list = []
    for n in range(5):
        fixture_image_data += np.ones(fixture_image_data.shape)
        list.append(Image(data=fixture_image_data, extent=fixture_extent))

    return list


@fixture
def fixture_image_sequence(fixture_list_of_images):
    """Produces an ImageSequence object."""
    return ImageSequence(fixture_list_of_images)
