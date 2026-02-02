from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np

from .extent import Extent


def save_hdf5_image(path, array, extent: Extent, metadata=None):
    """
    Saves an image to an hdf5 file.

    Parameters
    ----------
    path : str
        The path to the hdf5 file.
    array : np.ndarray
        The image to save.
    extent : list
        The extent of the image (x0, x1, z0, z1).
    metadata : dict
        Additional metadata to save.
    """

    extent = Extent(extent).sort()

    path = Path(path)

    if path.exists():
        path.unlink()
    if not path.parent.exists():
        path.parent.mkdir(parents=True)

    with h5py.File(path, "w") as dataset:
        dataset.create_dataset("image", data=array)
        dataset["image"].attrs["extent"] = extent
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
    """

    with h5py.File(path, "r") as dataset:
        array = dataset["image"][()]
        extent = dataset["image"].attrs["extent"]
        metadata = load_hdf5_to_dict(dataset)
        metadata.pop("image", None)
    from .ndimage import NDImage

    return NDImage(array=array, extent=extent, metadata=metadata)


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
            if value is None:
                continue
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


def check_hdf5_image_hash(path, hashable):
    """
    Checks the hash of an image in an hdf5 file.

    Parameters
    ----------
    path : str
        The path to the hdf5 file.
    hashable : any
        The data to check the hash against.

    Returns
    -------
    bool
        True if the hash matches, False otherwise.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File {path} does not exist.")

    with h5py.File(path, "r") as dataset:
        stored_hash = dataset["image"].attrs.get("hash", None)

    return stored_hash == hash(hashable)
