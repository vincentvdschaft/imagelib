from pathlib import Path

import numpy as np
import pytest

from imagelib import SCALE_DB, SCALE_LINEAR, Extent, Image


def _dict_equal(dict1, dict2):
    for key, value1 in dict1.items():
        value2 = dict2[key]
        if isinstance(value1, dict):
            if not _dict_equal(value1, value2):
                return False
        elif isinstance(value1, np.ndarray):
            if not np.allclose(value1, value2):
                return False
        elif value1 != value2:
            return False
    return True


def test_initialize_image(fixture_image_data, fixture_extent):
    """Initializes an image object."""
    image = Image(data=fixture_image_data, extent=fixture_extent)
    assert np.allclose(image.data, fixture_image_data)
    assert image.extent == fixture_extent
    assert image.shape == fixture_image_data.shape
    assert image.scale == SCALE_LINEAR


def test_data_immutable(fixture_image):
    """Ensures that the data attribute is immutable."""
    fixture_image.data[5, 5] = np.pi

    assert fixture_image.data[5, 5] != np.pi


def test_mul_scalar(fixture_image):
    """Tests multiplication of an image by a scalar."""
    image = fixture_image * 2
    assert np.allclose(image.data, fixture_image.data * 2)


def test_mul_ndarray(fixture_image):
    """Tests multiplication of an image by an ndarray."""
    multiplier = np.arange(fixture_image.n_pixels).reshape(fixture_image.shape)
    image = fixture_image.data * multiplier
    assert np.allclose(fixture_image.data * multiplier, image.data), "Data not equal."


def test_mul_image(fixture_image):
    """Tests multiplication of an image by an ndarray."""
    multiplier = np.arange(fixture_image.n_pixels).reshape(fixture_image.shape)
    image = fixture_image * multiplier
    assert isinstance(image, Image), "Not an Image object."
    assert np.allclose(fixture_image.data * multiplier, image.data), "Data not equal."


def test_rmul_scalar(fixture_image):
    """Tests multiplication of a scalar by an image."""
    image = 2 * fixture_image
    assert np.allclose(image.data, fixture_image.data * 2)


def test_rmul_ndarray(fixture_image):
    """Tests multiplication of an ndarray by an image."""
    multiplier = np.arange(fixture_image.n_pixels).reshape(fixture_image.shape)
    image = multiplier * fixture_image
    assert np.allclose(fixture_image.data * multiplier, image.data), "Data not equal."


def test_rmul_image(fixture_image):
    """Tests multiplication of an image by an image."""
    image = fixture_image * fixture_image
    assert isinstance(image, Image), "Not an Image object."
    assert np.allclose(fixture_image.data * fixture_image.data, image.data), (
        "Data not equal."
    )


def test_add(fixture_image):
    """Tests addition of an image by a scalar."""
    image = fixture_image + 2
    assert np.allclose(image.data, fixture_image.data + 2)


def test_sub(fixture_image):
    """Tests subtraction of an image by a scalar."""
    image = fixture_image - 2
    assert np.allclose(image.data, fixture_image.data - 2)


def test_resample(fixture_image):
    """Tests resampling of an image."""
    new_shape = (20, 50)
    new_extent = Extent(0, 1, 0, 1)
    image = fixture_image.resample(shape=new_shape, extent=new_extent, method="nearest")
    assert image.shape == new_shape
    assert image.extent == new_extent


def test_square_pixels(fixture_image):
    """Tests the square_pixels method."""
    assert fixture_image.pixel_w != fixture_image.pixel_h
    image = fixture_image.square_pixels()
    assert image.pixel_w == image.pixel_h


def test_raise_error_0_width(fixture_extent):
    """Tests the raise_error method."""
    with pytest.raises(ValueError):
        Image(data=np.random.rand(100, 0), extent=fixture_extent)

    with pytest.raises(ValueError):
        Image(data=np.random.rand(0, 100), extent=fixture_extent)


@pytest.mark.parametrize("shape", [(2, 100, 100), (100, 100, 3), (100, 100, 3, 3)])
def test_wrong_n_dim(shape, fixture_extent):
    """Tests the raise_error method."""
    with pytest.raises(ValueError):
        Image(data=np.random.rand(*shape), extent=fixture_extent)


@pytest.mark.parametrize("shape", [(1, 100), (100, 1)])
def test_size_one_nonzero_width(shape, fixture_extent):
    """Ensures that an error is raised when a dimension has size 1 but non-zero width."""

    with pytest.raises(ValueError):
        Image(data=np.ones(shape), extent=fixture_extent)


@pytest.mark.parametrize("slice_x, slice_y", [(0, slice(None, None, None))])
def test_getitem_scalar(fixture_image, slice_x, slice_y):
    """Tests slicing of an image."""
    image = fixture_image[slice_x, slice_y]
    assert np.allclose(image.data, fixture_image.data[slice_x, slice_y])
    if isinstance(slice_x, int):
        assert image.extent.width == 0
        assert image.shape[0] == 1
    if isinstance(slice_y, int):
        assert image.extent.height == 0
        assert image.shape[1] == 1


@pytest.mark.parametrize(
    "slice_x, slice_y",
    [
        (0, slice(5, 10, None)),
        (slice(5, 10, None), 0),
        (slice(55, 80, None), slice(5, 10, None)),
        (slice(-10, -1, None), slice(-10, -1, None)),
    ],
)
def test_getitem_slice(fixture_image, slice_x, slice_y):
    """Tests slicing of an image."""

    image = fixture_image[slice_x, slice_y]
    data_sliced_by_numpy = fixture_image.data[slice_x, slice_y].reshape(image.shape)
    assert np.allclose(image.data, data_sliced_by_numpy)
    assert image.extent != fixture_image.extent


def test_extent_after_slicing():
    """Tests the extent of an image after slicing."""
    data = np.random.rand(101, 101)
    extent = Extent(-5, 5, 0, 2)
    image = Image(data=data, extent=extent)
    image_sliced = image[0:51, 0:51]
    print(image_sliced.extent)
    assert image_sliced.extent == Extent(-5, 0, 0, 1)


def test_save_hdf5(fixture_image_with_metadata, tmp_path):
    """Tests saving an image to an HDF5 file."""
    fixture_image_with_metadata.save(tmp_path / "test.hdf5")
    image = Image.load(tmp_path / "test.hdf5")
    assert np.allclose(image.data, fixture_image_with_metadata.data)
    assert image.extent == fixture_image_with_metadata.extent
    assert _dict_equal(image.metadata, fixture_image_with_metadata.metadata)


@pytest.mark.parametrize("suffix", [".png", ".jpg", ".jpeg", ".bmp"])
def test_save_image_format(fixture_image, tmp_path, suffix):
    """Tests saving an image to an image file."""
    save_path = tmp_path / f"test{suffix}"
    fixture_image.save(save_path)
    assert Path(save_path).exists()


@pytest.mark.parametrize("transform", [np.square, np.log, np.exp, np.sqrt])
def test_match_histograms(fixture_image, transform):
    """Tests matching histograms of an image by transforming it with a monotonic
    function and then matching back to the original image."""
    image_transformed = fixture_image.apply_fn(transform)
    data_matched = image_transformed.match_histogram(fixture_image)
    assert np.allclose(data_matched.data, fixture_image.data)


@pytest.mark.parametrize("key, value", [("name", "test"), ("number", 3)])
def test_add_metadata(fixture_image, key, value):
    """Tests appending metadata to an image."""
    image = fixture_image.add_metadata(key=key, value=value)
    assert key in image.metadata
    assert image.metadata[key] == value


def test_append_metadata(fixture_image_with_metadata):
    """Tests appending metadata to an image."""
    fixture_image_with_metadata.append_metadata("new_key", "new_value")
    assert "new_key" in fixture_image_with_metadata.metadata
    assert isinstance(fixture_image_with_metadata.metadata["new_key"], list)
    assert fixture_image_with_metadata.metadata["new_key"][0] == "new_value"

    fixture_image_with_metadata.append_metadata("new_key", "second_value")
    assert "new_key" in fixture_image_with_metadata.metadata
    assert len(fixture_image_with_metadata.metadata["new_key"]) == 2


def test_grid(fixture_image):
    """Tests the grid method."""
    grid = fixture_image.grid
    assert grid.shape == (fixture_image.shape[0], fixture_image.shape[1], 2)
    assert np.min(grid[:, :, 0]) == fixture_image.extent.x0
    assert np.max(grid[:, :, 0]) == fixture_image.extent.x1
    assert np.min(grid[:, :, 1]) == fixture_image.extent.y0
    assert np.max(grid[:, :, 1]) == fixture_image.extent.y1


def test_flatgrid(fixture_image):
    """Tests the flatgrid method."""
    flatgrid = fixture_image.flatgrid
    assert flatgrid.shape == (fixture_image.n_pixels, 2)
    assert np.min(flatgrid[:, 0]) == fixture_image.extent.x0
    assert np.max(flatgrid[:, 0]) == fixture_image.extent.x1
    assert np.min(flatgrid[:, 1]) == fixture_image.extent.y0
    assert np.max(flatgrid[:, 1]) == fixture_image.extent.y1


def test_equal(fixture_image):
    """Tests the __eq__ method."""
    assert fixture_image == fixture_image
    assert fixture_image == fixture_image.copy()
    assert fixture_image != fixture_image + 1


def test_transpose(fixture_image):
    """Tests the transpose method."""
    image = fixture_image.transpose()
    assert image.shape == (fixture_image.shape[1], fixture_image.shape[0])
    assert fixture_image.transpose().transpose() == fixture_image


def test_xflip(fixture_image):
    """Tests the xflip method."""
    image = fixture_image.xflip()
    assert np.allclose(image.data, fixture_image.data[::-1])
    # Flipping should not change the extent
    assert image.extent == fixture_image.extent
    # Flipping twice should return the original image
    assert fixture_image.xflip().xflip() == fixture_image


def test_yflip(fixture_image):
    """Tests the yflip method."""
    image = fixture_image.yflip()
    assert np.allclose(image.data, fixture_image.data[:, ::-1])
    # Flipping should not change the extent
    assert image.extent == fixture_image.extent
    # Flipping twice should return the original image
    assert fixture_image.yflip().yflip() == fixture_image


def test_to_pixels(fixture_image):
    """Tests the to_pixels method."""
    image = fixture_image.to_pixels()
    assert image.max() == 1
    assert image.min() == 0
    assert image.scale == SCALE_LINEAR


def test_log_compress(fixture_image):
    """Tests the log_compress method."""
    image = fixture_image.normalize().log_compress()
    assert image.scale == SCALE_DB
    assert image.max() == 0
    assert fixture_image.log_compress().log_expand() == fixture_image


def test_window(fixture_image):
    """Tests the get_window method."""
    window = fixture_image.get_window(Extent(0, 1, 0, 1))
