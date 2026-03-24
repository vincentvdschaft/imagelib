import matplotlib.pyplot as plt

from imagelib import Image

image = Image.load("c.hdf5").normalize().log_compress().clip(-90, 0).to_pixels()
image_clahe = image.clahe(tile_grid_size=(16, 16))

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image.array.T, cmap="gray")
axes[0].set_title("Original Image")
axes[0].axis("off")
axes[1].imshow(image_clahe.array.T, cmap="gray")
axes[1].set_title("CLAHE Image")
axes[1].axis("off")
plt.tight_layout()
plt.show()
