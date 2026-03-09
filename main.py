import matplotlib.pyplot as plt
import numpy as np

from imagelib import *

image = Image(np.arange(100).reshape(10, 10), extent=(0, 10, 0, 10))

image2 = image.resample((5, 5), extent=(5, 15, 5, 15))
# plt.imshow(image.array.T, extent=image.extent_imshow, origin="lower")
plt.imshow(image2.array.T, extent=image2.extent_imshow, origin="lower")
plt.show()
