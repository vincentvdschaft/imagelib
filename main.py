import matplotlib.pyplot as plt
import numpy as np

from imagelib import *

image = Image.test_image()

# Extent is now (-10.0, 0.0, 0.0, 20.0)
image_sliced = image[:65]

# Add metadata and save
image_sliced.add_metadata(key="date", value="1980-10-10").save("image.hdf5")

# Load image
image_loaded = Image.load("image.hdf5")

# Resample and window image
image_resampled = image_loaded.resample(
    shape=(8, 8), extent=(-10, -2, 0, 10), method="nearest"
)

image_kspace = np.abs(image.fft())

n_x, n_y = image_resampled.shape
image_extent_in_pixels = image_resampled.with_extent(Extent((0, n_x - 1, 0, n_y - 1)))

# Plot the images
fig, axes = plt.subplots(1, 6)
axes[0].imshow(image.array, extent=image.extent_imshow)
axes[1].imshow(image_sliced.array, extent=image_sliced.extent_imshow)
axes[2].imshow(image_loaded.array, extent=image_loaded.extent_imshow)
axes[3].imshow(image_resampled.array, extent=image_resampled.extent_imshow)
axes[4].imshow(image_kspace.array, extent=image_kspace.extent_imshow)
axes[5].imshow(
    image_extent_in_pixels.array, extent=image_extent_in_pixels.extent_imshow
)
plt.tight_layout()
plt.show()
