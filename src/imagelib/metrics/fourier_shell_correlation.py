from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

from imagelib import Image


@dataclass
class FSCResult:
    frequencies: np.ndarray
    correlations: np.ndarray
    num_voxels_in_shell: np.ndarray

    def save(self, path: str | Path):
        """Saves the FSC result to a file."""
        np.savez(
            path,
            frequencies=self.frequencies,
            correlations=self.correlations,
            num_voxels_in_shell=self.num_voxels_in_shell,
        )

    @classmethod
    def load(cls, path: str | Path) -> FSCResult:
        """Loads the FSC result from a file."""
        data = np.load(path)
        return cls(
            frequencies=data["frequencies"],
            correlations=data["correlations"],
            num_voxels_in_shell=data["num_voxels_in_shell"],
        )


def fourier_shell_correlation(
    image1: Image, image2: Image, num_shells: int
) -> FSCResult:
    """Computes the Fourier Shell Correlation (FSC) between two 3D images.

    Args:
        image1: First ND image
        image2: Second ND image
        num_shells: Number of shells to compute the FSC over

    Returns:
        fsc_result: A dataclass containing frequencies, correlations, and number of
            voxels in each shell
    """
    _check_input_fourier_shell_correlation(image1, image2, num_shells)

    image1_ft, image2_ft = image1.fft(), image2.fft()

    grid = image1_ft.grid
    radii = np.linalg.norm(grid, axis=-1)
    pixel_size = image1.pixel_size(0)
    final_frequency = _compute_final_frequency(image1.shape, pixel_size)
    print(f"Final frequency: {final_frequency:.4f} 1/μm")
    num_voxels_in_shell = np.zeros(num_shells, dtype=np.int64)

    shell_width = final_frequency / num_shells
    correlations = np.zeros(num_shells)
    for n in range(num_shells):
        r_min = n * shell_width
        r_max = (n + 1) * shell_width

        shell_mask = (radii >= r_min) & (radii < r_max)

        num_voxels_in_shell[n] = np.sum(shell_mask)
        if num_voxels_in_shell[n] == 0:
            correlations[n] = 0.0
            continue

        num = np.sum(image1_ft.array[shell_mask] * np.conj(image2_ft.array[shell_mask]))
        denom1 = np.sum(np.abs(image1_ft.array[shell_mask]) ** 2)
        denom2 = np.sum(np.abs(image2_ft.array[shell_mask]) ** 2)

        if denom1 == 0 or denom2 == 0:
            correlations[n] = 0.0
        else:
            correlations[n] = np.abs(num) / np.sqrt(denom1 * denom2)

    fsc_frequencies = (np.arange(num_shells) + 0.5) * shell_width
    return FSCResult(
        frequencies=fsc_frequencies,
        correlations=correlations,
        num_voxels_in_shell=num_voxels_in_shell,
    )


def _compute_final_frequency(shape: tuple, pixel_size: float) -> float:
    """Computes the smallest maximum frequency in the Fourier grid."""
    smallest_dim = min(shape)
    return np.max(np.fft.fftfreq(smallest_dim, d=pixel_size))


def _check_input_fourier_shell_correlation(
    image1: Image, image2: Image, num_shells: int
) -> None:
    """Checks the input parameters for the fourier_shell_correlation function."""
    if not isinstance(image1, Image):
        raise TypeError("image1 must be an instance of Image")
    if not isinstance(image2, Image):
        raise TypeError("image2 must be an instance of Image")
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same shape")
    if image1.limits != image2.limits:
        raise ValueError("Images must have the same limits")
    if not isinstance(num_shells, int) or num_shells <= 0:
        raise ValueError("num_shells must be a positive integer")
    if not np.diff(image1.pixel_sizes).max() < 1e-6:
        raise ValueError("images must have isotropic pixel sizes")


def threshold_2sigma(num_voxels: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return 2 / np.sqrt(num_voxels)


def threshold_half_bit(num_voxels: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return (0.2071 + 1.9102 / np.sqrt(num_voxels)) / (
            1.2071 + 0.9102 / np.sqrt(num_voxels)
        )


def compute_resolution_from_fsc(
    fsc_result: FSCResult,
    threshold_func: Callable = threshold_half_bit,
    smoothing_size: int = 10,
) -> float:
    """Computes the resolution from the FSC result using a given threshold function.

    Parameters
    ----------
        fsc_result : FSCResult
            The result of the Fourier Shell Correlation computation.
        threshold_func : function
            A function that takes the number of voxels in a shell and returns the corresponding FSC threshold.
        smoothing_size : int
            The size of the uniform filter for smoothing the FSC values.

    Returns
    -------
        float
            The resolution corresponding to the first shell where the FSC drops below the threshold.
    """
    smoothed = uniform_filter1d(
        fsc_result.correlations, size=smoothing_size, mode="nearest"
    )
    frequency = _find_first_below_threshold(
        smoothed,
        fsc_result.frequencies,
        threshold_func(fsc_result.num_voxels_in_shell),
    )
    return 1 / frequency if frequency > 0 else float("inf")


def _find_first_below_threshold(fsc_values, fsc_frequencies, threshold_per_shell):
    for n in range(len(fsc_values)):
        if fsc_values[n] < threshold_per_shell[n]:
            return fsc_frequencies[n]
    return fsc_frequencies[-1]
