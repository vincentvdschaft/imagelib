import numpy as np

from imagelib import Image


def fourier_shell_correlation_cutoff_resolution(
    image1: Image, image2: Image, threshold: float = 0.5, num_shells: int = 100
) -> float:
    """Computes the Fourier Shell Correlation (FSC) cutoff frequency between two 3D images.

    Parameters
    ----------
        image1 : Image
            First ND image
        image2 : Image
            Second ND image
        threshold : float
            FSC threshold to determine the cutoff frequency
        num_shells : int
            Number of shells to compute the FSC over

    Returns
    -------
        cutoff_frequency : float
            The cutoff frequency where the FSC drops below the threshold
    """
    fsc_frequencies, fsc_values = fourier_shell_correlation(image1, image2, num_shells)

    # Find the cutoff frequency
    cutoff_frequency = _find_first_below_threshold(
        fsc_values, fsc_frequencies, threshold
    )

    resolution = 1 / cutoff_frequency if cutoff_frequency != 0 else 0.0
    return resolution


def _find_first_below_threshold(fsc_values, fsc_frequencies, threshold):
    for n in range(len(fsc_values)):
        if fsc_values[n] < threshold:
            return fsc_frequencies[n]
    raise ValueError("FSC values never drop below the threshold.")


def fourier_shell_correlation(image1: Image, image2: Image, num_shells: int) -> list:
    """Computes the Fourier Shell Correlation (FSC) between two 3D images.

    Parameters
    ----------
        image1 : Image
            First ND image
        image2 : Image
            Second ND image
        num_shells : int
            Number of shells to compute the FSC over

    Returns
    -------
        fsc_frequencies : np.ndarray
            Array of frequencies corresponding to each shell
        fsc_values : np.ndarray
            Array of FSC values for each shell
    """
    _check_input_fourier_shell_correlation(image1, image2, num_shells)

    image1_ft, image2_ft = image1.fft(), image2.fft()

    grid = image1_ft.grid
    radii = np.linalg.norm(grid, axis=-1)
    largest_max = _compute_largest_maximum(grid)

    shell_width = largest_max / num_shells
    fsc_values = np.zeros(num_shells)
    for n in range(num_shells):
        r_min = n * shell_width
        r_max = (n + 1) * shell_width

        shell_mask = (radii >= r_min) & (radii < r_max)

        num = np.sum(image1_ft.array[shell_mask] * np.conj(image2_ft.array[shell_mask]))
        denom1 = np.sum(np.abs(image1_ft.array[shell_mask]) ** 2)
        denom2 = np.sum(np.abs(image2_ft.array[shell_mask]) ** 2)

        if denom1 == 0 or denom2 == 0:
            fsc_values[n] = 0.0
        else:
            fsc_values[n] = np.abs(num) / np.sqrt(denom1 * denom2)

    fsc_frequencies = (np.arange(num_shells) + 0.5) * shell_width
    return fsc_frequencies, fsc_values


def _compute_largest_maximum(grid):
    flatgrid = grid.reshape(-1, grid.shape[-1])
    max_per_axis = np.max(np.abs(flatgrid), axis=0)
    largest_max = np.max(max_per_axis)
    return largest_max


def _check_input_fourier_shell_correlation(image1, image2, num_shells):
    """Checks the input parameters for the fourier_shell_correlation function."""
    assert isinstance(image1, Image), "image1 must be an instance of Image"
    assert isinstance(image2, Image), "image2 must be an instance of Image"
    assert image1.shape == image2.shape, "Images must have the same shape"
    assert image1.extent == image2.extent, "Images must have the same extent"
    assert isinstance(num_shells, int) and num_shells > 0, (
        "num_shells must be a positive integer"
    )
