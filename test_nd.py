import numpy as np

from imagelib.ndextent import Extent
from imagelib.ndimage import NDImage

image = NDImage(np.ones((5, 5)), extent=(-1, 1, -1, 1), metadata={"author": "test"})
print(image.get_window(extent=(-0.8, 0.8, -0.8, 0.8)))
print(np.square(image).metadata)


image[:3].save("test.hdf5")


image_loaded = NDImage.load("test.hdf5")
print(image_loaded)
print(image[:3, 2:])
