import matplotlib.pyplot as plt
import numpy as np

from imagelib import Image

im = Image(
    np.arange(4 * 4).reshape(4, 4),
    labels=("y", "x"),
    units=("mm", "mm"),
    limits=((0, 6), (0, 3)),
)

fig, axes = plt.subplots(1, 2)

im2 = im.T

for ax, im in zip(axes, [im, im2]):
    ax.imshow(im.array, extent=im.extent_imshow, origin="lower", cmap="viridis")
    ax.set_xlabel(f"{im.labels[-1]} [{im.units[-1]}]")
    ax.set_ylabel(f"{im.labels[-2]} [{im.units[-2]}]")


plt.show()
