import matplotlib.pyplot as plt
import numpy as np

from imagelib.ndextent import Extent
from imagelib.ndimage import NDImage

image = NDImage(np.random.randn(4, 5, 5), extent=(0, 3, -1, 1, 0, 1))
image + np.array(5)
print(image.log_expand())
image_normalized = image.normalize()

# print(image.clip(0, 1).array)


image_square = image.square_pixels()
print(image_square)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(
    image[0].log_compress().normalize_db().clip(-60, 0).T,
    origin="lower",
    extent=image.extent_imshow,
    aspect="equal",
)
axes[1].imshow(
    image_square.T, origin="lower", extent=image_square.extent_imshow, aspect="equal"
)

plt.show()
print(image.square_pixels())
print(image.get_window(extent=(-0.8, 0.8, -0.8, 0.8)))
print(np.square(image).metadata)

exit()
image = NDImage(np.ones((25, 25)), extent=(-1, 1, -1, 1))
grid = image.grid
image = NDImage(
    np.sin(np.pi * grid[..., 0]) * np.cos(np.pi * grid[..., 1]),
    extent=(-1, 1, -1, 1),
)

resamples = image.resample(shape=(100, 100))
print(resamples)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image, origin="lower", extent=image.extent_imshow)
axes[1].imshow(resamples, origin="lower", extent=resamples.extent_imshow)
plt.show()

print(image.resample(shape=(10, 10)))
image[:3].save("test.hdf5")


image_loaded = NDImage.load("test.hdf5")
print(image_loaded)
print(image[:3, 2:])
