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
    shape=(16, 16), extent=(-10, -2, 0, 10), method="nearest"
)

# Plot the images
fig, axes = plt.subplots(1, 4)
axes[0].imshow(image.data.T, extent=image.extent)
axes[1].imshow(image_sliced.data.T, extent=image_sliced.extent)
axes[2].imshow(image_loaded.data.T, extent=image_loaded.extent)
axes[3].imshow(image_resampled.data.T, extent=image_resampled.extent)
plt.tight_layout()
plt.show()
