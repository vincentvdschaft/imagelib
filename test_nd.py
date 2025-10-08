import numpy as np

from imagelib.ndimage import NDImage

image = NDImage(np.ones((5, 5)), extent=(-1, 1, -1, 1), metadata={"author": "test"})
print(image.metadata)
print(np.square(image).metadata)


print(image[:3, 2:])
