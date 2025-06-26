from pysimfrac.src.general.simFrac import SimFrac
import numpy as np
import matplotlib.pyplot as plt

# ---- 1. Generate synthetic aperture field ----
myfrac = SimFrac(
    h=0.01,         # grid spacing in mm
    lx=3.0,         # domain length x (mm)
    ly=1.0,         # domain length y (mm)
    nx=300,         # number of x nodes
    ny=100,         # number of y nodes
    method="spectral",  # or 'box', 'gaussian'
    units="mm"
)
myfrac.params["mean-aperture"]["value"] = 1.0    # mean width (mm)
myfrac.params["roughness"]["value"] = 0.01       # RMS roughness (mm)
myfrac.params["H"]["value"] = 0.5                # Hurst exponent
myfrac.create_fracture()
aperture = myfrac.aperture  # 2D aperture/width map, units: mm

# ---- 2. Plot the aperture field ----
plt.figure(figsize=(6,4))
plt.imshow(aperture, cmap="viridis", aspect='auto')
plt.colorbar(label="Aperture / Width (mm)")
plt.title("pySimFrac Synthetic Aperture Map")
plt.axis("off")
plt.tight_layout()
plt.savefig("pysimfrac_aperture_field.png", dpi=300)
plt.close()

# ---- 3. Plot a centerline width profile ----
profile = aperture[aperture.shape[0] // 2, :]
x_profile = np.linspace(0, myfrac.lx, len(profile))

plt.figure(figsize=(8,4))
plt.plot(x_profile, profile, label="pySimFrac Centerline Profile")
plt.xlabel("Position along fracture (mm)")
plt.ylabel("Aperture / Width (mm)")
plt.title("pySimFrac: Centerline Crack Width Profile")
plt.legend()
plt.tight_layout()
plt.savefig("pysimfrac_centerline_profile.png", dpi=300)
plt.show()

# ---- 4. Print basic stats (optional) ----
print("Aperture field: shape", aperture.shape)
print("Mean width (mm):", np.mean(aperture))
print("Stddev width (mm):", np.std(aperture))
print("Min width (mm):", np.min(aperture), "Max width (mm):", np.max(aperture))
