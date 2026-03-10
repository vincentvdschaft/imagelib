import matplotlib.pyplot as plt
import numpy as np

from imagelib import Image

size = 32
array = np.random.rand(size, size)

image = (
    Image(array).moving_average(ax=0, window_size=5).moving_average(ax=1, window_size=5)
)

# image2 = image.resample((5, 5), extent=(5, 15, 5, 15))
plt.imshow(image.array.T, extent=image.extent_imshow, origin="lower")
# plt.imshow(image2.array.T, extent=image2.extent_imshow, origin="lower")
plt.show()
