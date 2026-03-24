import matplotlib.pyplot as plt
import numpy as np

from imagelib import Image
from imagelib.metrics import (
    fourier_shell_correlation,
    threshold_2sigma,
    threshold_half_bit,
)

image1, image2 = (
    Image.load("RF_002.hdf5").normalize(),
    Image.load("RF_003.hdf5").normalize(),
)
print(np.sum(image1.array - image2.array))

frc = fourier_shell_correlation(image1, image2, num_shells=600)
resolutions = 1 / frc.frequencies
plt.plot(frc.frequencies, frc.correlations)
plt.plot(frc.frequencies, threshold_2sigma(frc.num_voxels_in_shell), label="2-sigma")
plt.plot(frc.frequencies, threshold_half_bit(frc.num_voxels_in_shell), label="Half-bit")
xticks = plt.xticks()
plt.xticks(xticks[0], [f"{1 / f:.2f}" if f > 0 else "Inf" for f in xticks[0]])
plt.xlabel("Frequency")
plt.ylabel("FSC")
plt.title("Fourier Shell Correlation")
plt.legend()
plt.grid()
plt.show()
