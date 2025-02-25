import matplotlib.pyplot as plt
import numpy as np

from imagelib import *

image = Image(data=np.random.randn(129, 129), extent=(-10, 10, 0, 15))

# Extent is now (-10.0, 0.0, 0.0, 20.0)
image = image[:65]

# Add metadata and save
image.add_metadata(key="date", value="1980-10-10").save("image.hdf5")

# Load image
image_loaded = Image.load("image.hdf5")

# Resample and window image
image_resampled = image_loaded.resample(
    shape=(32, 32), extent=(-10, -2, 0, 10), method="nearest"
)

# Plot the images
fig, axes = plt.subplots(1, 3)
axes[0].imshow(image.data.T, extent=image_loaded.extent)
axes[1].imshow(image_loaded.data.T, extent=image_loaded.extent)
axes[2].imshow(image_resampled.data.T, extent=image_loaded.extent)
plt.tight_layout()
plt.show()
