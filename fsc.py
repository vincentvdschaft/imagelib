import matplotlib.pyplot as plt
import numpy as np

from imagelib import Extent, Image
from imagelib.metrics import (
    fourier_shell_correlation,
    fourier_shell_correlation_cutoff_resolution,
)

n_files = 100
image1 = Image.load(
    f"/home/vincent/1-projects/pala_process/data/images/rfulm_image_preprocessed_{n_files:03d}_000.hdf5"
)
image2 = Image.load(
    f"/home/vincent/1-projects/pala_process/data/images/rfulm_image_preprocessed_{n_files:03d}_{n_files:03d}.hdf5"
)

window = Extent.from_bbox(-4e-3, 3e-3, 3e-3, 3e-3)

noise = np.random.randn(*image1.shape) * 0.04

image1 = image1 + noise
image2 = image2 + noise

# image1 = image1.get_window(window)
# image2 = image2.get_window(window)

fsc_freqs, fsc_vals = fourier_shell_correlation(image1, image2, num_shells=256)

resolution = fourier_shell_correlation_cutoff_resolution(
    image1, image2, num_shells=256, threshold=0.5
)

fc = 15.8e6
c = 1540
wavelength = c / fc
print(f"Estimated resolution: {resolution * 1e6:.0f} um")
print(f"this is: {resolution / wavelength:.2f} wavelengths")

fig, axes = plt.subplots(2, 1)
axes[0].imshow(image1.T, extent=image1.extent_imshow * 1e6)
axes[1].plot(fsc_freqs, fsc_vals)
axes[1].axvline(1 / resolution, color="red", linestyle="--")
plt.tight_layout()
plt.show()
