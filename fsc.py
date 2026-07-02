import matplotlib.pyplot as plt
import numpy as np

from imagelib import Image

im = Image(
    np.arange(4 * 4).reshape(4, 4),
    labels=("x", "y"),
    units=("mm", "mm"),
    limits=((0, 3), (0, 3)),
).T

plt.imshow(im.array, extent=im.extent_imshow, origin="lower", cmap="viridis")
plt.xlabel(f"{im.labels[0]} [{im.units[0]}]")
plt.ylabel(f"{im.labels[1]} [{im.units[1]}]")

plt.show()
