import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.morphology import medial_axis
from scipy.ndimage import zoom

# Load actual crack mask
mask = imread('/mnt/e/camerer_ml/datasets/METU_Concrete_Crack_Seg/BW_visible/236.png', as_gray=True) > 0.5

# Measure real widths via medial axis
medial, dist = medial_axis(mask, return_distance=True)
w_real = np.zeros(mask.shape[1])
for x in range(mask.shape[1]):
    ys = np.where(medial[:, x])[0]
    if ys.size:
        w_real[x] = (dist[ys, x] * 2).mean()
    else:
        w_real[x] = np.nan

# Generate synthetic aperture via pySimFrac
from pysimfrac import SimFrac
from pysimfrac.src.general.simFrac import SimFrac

# Example values — adjust as needed for your test!
sf = SimFrac(
    h=0.01,      # grid spacing (mm)
    lx=3.0,      # domain length in x (mm)
    ly=1.0,      # domain length in y (mm)
    nx=300,      # grid points x
    ny=100,      # grid points y
    method="spectral",  # or "box", "gaussian"
    units="mm"
)
sf.create_fracture()  # Don't forget to generate the surface!
ap = sf.aperture


# Take center row profile
w_synth = ap[ap.shape[0]//2]

# Resize widths for plotting
L = len(w_synth)
w_resized = zoom(np.nan_to_num(w_real), L / len(w_real), order=1)

# Plot comparison
plt.figure(figsize=(8,4))
plt.plot(w_synth, label='pySimFrac synthetic (aperture)')
plt.plot(w_resized, label='Real mask via medial axis')
plt.legend()
plt.xlabel('Position along crack')
plt.ylabel('Width (pixels or mm)')
plt.title('Synthetic vs Measured crack width')
plt.show()
