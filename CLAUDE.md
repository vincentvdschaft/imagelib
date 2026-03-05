# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all tests
pytest

# Run a single test file
pytest tests/test_image.py

# Run a single test
pytest tests/test_image.py::test_initialize_image

# Install the package in editable mode (from .venv)
.venv/bin/pip install -e .
```

The virtual environment is at `.venv/` (Python 3.12). Activate with `source .venv/bin/activate` or prefix commands with `.venv/bin/`.

## Architecture

`imagelib` is a Python library providing an `NDImage` class — a numpy array bundled with physical spatial extent metadata, so dimensions (x, y, z ranges) stay synchronized through transformations.

### Core classes

**`Extent`** (`imagelib/extent.py`) — immutable tuple subclass storing physical coordinates as `(x0, x1, y0, y1, ...)`. Indexed by dimension: `extent.start(dim)`, `extent.end(dim)`. Supports arithmetic operators that broadcast across all elements.

**`NDImage`** (`imagelib/ndimage.py`) — the main class, exposed as `Image` from the package. Wraps a numpy array with an `Extent`. Key design principles:
- All operations return a new `NDImage` (immutable-style)
- Implements `__array_ufunc__` and `__array_function__` so numpy ufuncs (e.g. `np.sin(image)`) work transparently, preserving extent
- Slicing via `__getitem__` recomputes extent from the physical coordinate grid, so `image[:65]` correctly updates the extent
- `extent_imshow` adjusts extent by half a pixel for use with `matplotlib.imshow` (which treats extent as pixel edges, not centers)

**`saving.py`** — HDF5 serialization via h5py. Supports nested dict metadata with list-to-numbered-dict round-trip encoding.

### Public API (from `imagelib import *`)

- `Image` — the main class (`NDImage`)
- `Extent` — spatial extent class
- `save_hdf5_image`, `load_hdf5_image`, `check_hdf5_image_hash` — standalone HDF5 I/O

### Coordinate convention

Extent stores coordinates as `(dim0_start, dim0_end, dim1_start, dim1_end, ...)`. Array axis 0 = x, axis 1 = y. When plotting with `imshow`, use `image.array.T` and `extent=image.extent_imshow` (or `image.extent` for `[x0, x1, y0, y1]` order expected by matplotlib's `extent` parameter when `origin="lower"`).
