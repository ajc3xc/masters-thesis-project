import numpy as np
import matplotlib.pyplot as plt

# ---- 1. Generate synthetic aperture (width) field via pySimFrac ----
from pysimfrac.src.general.simFrac import SimFrac

myfrac = SimFrac(h=0.01, lx=3.0, ly=1.0, nx=300, ny=100, method="spectral", units="mm")
myfrac.params["mean-aperture"]["value"] = 1.0
myfrac.params["roughness"]["value"] = 0.01
myfrac.params["H"]["value"] = 0.5
myfrac.create_fracture()
aperture = myfrac.aperture  # 2D array: each pixel is local width in mm

# ---- 2. Plot the synthetic aperture field ----
plt.figure(figsize=(6,4))
plt.imshow(aperture, cmap="viridis")
plt.colorbar(label="Aperture (mm)")
plt.title("pySimFrac Aperture Field")
plt.axis("off")
plt.tight_layout()
plt.savefig("pysimfrac_aperture_field.png", dpi=300)
plt.close()

# ---- 3. Extract width profile (middle row) ----
profile_synth = aperture[aperture.shape[0]//2, :]
x_synth = np.arange(len(profile_synth))

plt.figure(figsize=(8,4))
plt.plot(x_synth, profile_synth, label='pySimFrac (mid-row)')
plt.xlabel('Position along fracture (x)')
plt.ylabel('Aperture / Width (mm)')
plt.title("Aperture profile: pySimFrac vs. Real Mask")
plt.grid()

# ---- 4. OPTIONAL: Overlay real mask width profile ----
# To compare to a real mask, set BW_PATH or RGB_PATH:
BW_PATH = '/mnt/e/camerer_ml/datasets/METU_Concrete_Crack_Seg/BW_visible/236.png'   # <-- update!
try:
    from skimage.io import imread
    from skimage.morphology import medial_axis
    from scipy.ndimage import zoom

    mask = imread(BW_PATH, as_gray=True)
    mask_bin = mask > 0.5
    # Medial axis for width at each column
    medial, dist = medial_axis(mask_bin, return_distance=True)
    widths_real = np.zeros(mask_bin.shape[1])
    for x in range(mask_bin.shape[1]):
        pts = np.where(medial[:, x])[0]
        if len(pts):
            widths_real[x] = np.mean(dist[pts, x]) * 2
        else:
            widths_real[x] = np.nan

    # Resize real profile to match synthetic profile's length
    target_length = len(profile_synth)  # 300
    if len(widths_real) != target_length:
        widths_no_nan = np.nan_to_num(widths_real, nan=0)
        factor = target_length / len(widths_real)
        widths_resized = zoom(widths_no_nan, factor, order=1)
    else:
        widths_resized = widths_real

    # CALIBRATION: convert to mm if you know pixel size!
    PIXEL_SIZE_MM = 0.02   # <-- Set this to your true pixel size in mm (example: 0.02 mm/pixel)
    widths_resized_mm = widths_resized * PIXEL_SIZE_MM

    plt.plot(x_synth, widths_resized_mm,
             label='Real Mask (col avg, resized, mm)', color='red', alpha=0.6)
except Exception as e:
    print(f"Could not compare to real mask: {e}")

plt.legend()
plt.tight_layout()
plt.savefig("pysimfrac_vs_real_profile.png", dpi=300)
plt.show()
