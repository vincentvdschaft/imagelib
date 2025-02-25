import numpy as np
import pytest

from imagelib import Extent, Image


def test_extent_initialize_from_tuple(fixture_extent_tuple):
    """Initializes an extent object from a tuple."""
    extent = Extent(*fixture_extent_tuple)
    assert extent == fixture_extent_tuple


def test_extent_initialize_from_kwargs(fixture_extent_tuple):
    """Initializes an extent object from kwargs."""
    extent = Extent(
        x0=fixture_extent_tuple[0],
        x1=fixture_extent_tuple[1],
        y0=fixture_extent_tuple[2],
        y1=fixture_extent_tuple[3],
    )
    assert extent == fixture_extent_tuple


@pytest.mark.parametrize(
    "input_tuple, expected",
    [
        ((-1, 1, 0, 2), (-1, 1, 0, 2)),
        ((1, -1, 2, 0), (-1, 1, 0, 2)),
        ((1, 1, 2, 2), (1, 1, 2, 2)),
        ((1, -1, 0, 2), (-1, 1, 0, 2)),
        ((1, -1, 2, 0), (-1, 1, 0, 2)),
    ],
)
def test_extent_sort(input_tuple, expected):
    """Tests the sort method of the extent object."""
    extent = Extent(*input_tuple)
    assert extent.sort() == expected


def test_mul():
    """Tests the multiplication operator of the extent object."""
    extent = Extent(-1, 1, 0, 2)
    extent = extent * 2
    assert extent == (-2, 2, 0, 4)


def test_rmul():
    """Tests the reverse multiplication operator of the extent object."""
    extent = Extent(-1, 1, 0, 2)
    extent = 2 * extent
    assert extent == (-2, 2, 0, 4)


def test_div():
    """Tests the division operator of the extent object."""
    extent = Extent(-1, 1, 0, 2)
    extent = extent / 2
    assert extent == (-0.5, 0.5, 0, 1)


def test_floordiv():
    """Tests the floor division operator of the extent object."""
    extent = Extent(-1, 1, 0, 2)
    extent = extent // 2
    assert extent == (-1, 0, 0, 1)


def test_add_scalar():
    """Tests the subtraction operator of the extent object."""
    extent = Extent(-1, 1, 0, 2)
    extent = extent + 1
    assert extent == (0, 2, 1, 3)


def test_sub_scalar():
    """Tests the subtraction operator of the extent object."""
    extent = Extent(-1, 1, 0, 2)
    extent = extent - 1
    assert extent == (-2, 0, -1, 1)


def test_add_tuple():
    """Tests the addition operator of the extent object."""
    extent = Extent(-1, 1, 0, 2)
    extent = extent + (1, 1)
    assert extent == (0, 2, 1, 3)


def test_sub_tuple():
    """Tests the subtraction operator of the extent object."""
    extent = Extent(-1, 1, 0, 2)
    extent = extent - (1, 1)
    assert extent == (-2, 0, -1, 1)


def test_set_x0(fixture_extent):
    """Tests the set_x0 method of the extent object."""
    assert fixture_extent.set_x0(4)[0] == 4


def test_set_x1(fixture_extent):
    """Tests the set_x1 method of the extent object."""
    assert fixture_extent.set_x1(4)[1] == 4


def test_set_y0(fixture_extent):
    """Tests the set_y0 method of the extent object."""
    assert fixture_extent.set_y0(4)[2] == 4


def test_set_y1(fixture_extent):
    """Tests the set_y1 method of the extent object."""
    assert fixture_extent.set_y1(4)[3] == 4


def test_aspect(fixture_extent):
    """Tests the aspect property of the extent object."""
    assert fixture_extent.aspect == fixture_extent.height / fixture_extent.width


def test_xlims(fixture_extent):
    """Tests the xlims property of the extent object."""
    assert fixture_extent.xlims == (fixture_extent.x0, fixture_extent.x1)


def test_xlims_flipped(fixture_extent):
    """Tests the xlims_flipped property of the extent object."""
    assert fixture_extent.xlims_flipped == (fixture_extent.x1, fixture_extent.x0)


def test_ylims(fixture_extent):
    """Tests the ylims property of the extent object."""
    assert fixture_extent.ylims == (fixture_extent.y0, fixture_extent.y1)


def test_ylims_flipped(fixture_extent):
    """Tests the ylims_flipped property of the extent object."""
    assert fixture_extent.ylims_flipped == (fixture_extent.y1, fixture_extent.y0)
