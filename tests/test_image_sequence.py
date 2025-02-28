import pytest
import numpy as np
from imagelib.image_sequence import ImageSequence


def test_initialize_image_sequence(fixture_list_of_images):
    """Test that the ImageSequence class can be initialized."""
    image_sequence = ImageSequence(fixture_list_of_images)
    assert len(image_sequence.images) == 5


def test_initialize_image_sequence_differing_extents(fixture_list_of_images):
    """Test that the ImageSequence class can be initialized with differing extents."""
    fixture_list_of_images[0].extent = (-1, 1, 0, 3)
    fixture_list_of_images[1].extent = (-2, 2, 1, 4)
    with pytest.raises(ValueError):
        ImageSequence(fixture_list_of_images)


def test_initialize_image_sequence_differing_scales(fixture_list_of_images):
    """Test that the ImageSequence class can be initialized with differing scales."""
    fixture_list_of_images[0].scale = "SCALE_LINEAR"
    fixture_list_of_images[1].scale = "SCALE_DB"
    with pytest.raises(ValueError):
        ImageSequence(fixture_list_of_images)


def test_initialize_image_sequence_single_image(fixture_image):
    """Test that the ImageSequence class can be initialized with a single image."""
    image_sequence = ImageSequence(fixture_image)
    assert len(image_sequence.images) == 1
    assert image_sequence.images[0] == fixture_image


def test_initialize_image_sequence_3d_array(fixture_extent):
    """Test that the ImageSequence class can be initialized with a 3D array."""
    data = np.random.rand(5, 10, 10)
    image_sequence = ImageSequence.from_numpy(data=data, extent=fixture_extent)
    assert len(image_sequence.images) == 5
    assert image_sequence.images[0].data.shape == (10, 10)


def test_save(tmpdir, fixture_image_sequence):
    """Test that the ImageSequence class can save images."""
    fixture_image_sequence.save(directory=tmpdir, name="test")
    assert (tmpdir / "test_00000.hdf5").exists()


def test_save_png(tmpdir, fixture_image_sequence):
    """Test that the ImageSequence class can save images as PNG."""
    fixture_image_sequence.save(directory=tmpdir, name="test.png")
    assert (tmpdir / "test_00000.png").exists()


def test_add(fixture_image_sequence):
    """Test that the ImageSequence class can add images."""
    subtracted = fixture_image_sequence - 5
    im0 = fixture_image_sequence.images[0]
    im1 = subtracted.images[0]
    assert np.allclose(im0.data - 5, im1.data)
