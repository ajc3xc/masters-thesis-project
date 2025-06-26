# pysimfrac_crack_width.py
#from pysimfrac.src.general import SimFrac
from src.pysimfrac.src.general import SimFrac
import matplotlib.pyplot as plt

# 1. Create a synthetic fracture surface
myfrac = SimFrac(
    h      = 0.01,     # grid spacing (mm)
    lx     = 3.0,      # domain length x (mm)
    ly     = 1.0,      # domain length y (mm)
    nx     = 300,      # nodes in x
    ny     = 100,      # nodes in y
    method = "spectral",  # generation method
    units  = "mm"
)                    # :contentReference[oaicite:7]{index=7}

# 2. Set spectral parameters
myfrac.params["mean-aperture"]["value"] = 0.5   # mean width 0.5 mm :contentReference[oaicite:8]{index=8}
myfrac.params["roughness"]["value"]     = 0.05  # σ = 0.05 mm :contentReference[oaicite:9]{index=9}
myfrac.params["H"]["value"]             = 0.5   # Hurst exponent :contentReference[oaicite:10]{index=10}
myfrac.params["aniso"]["value"]         = 0.5   # anisotropy ratio :contentReference[oaicite:11]{index=11}
myfrac.create_fracture()                     # generate aperture field :contentReference[oaicite:12]{index=12}

# 3. Retrieve the full-field aperture map
aperture = myfrac.aperture_field  # 2D numpy array of local widths

# 4. Estimate effective apertures
models = ["mean", "gmean", "hmean", "numerical"]
for m in models:
    myfrac.get_effective_aperture(m)    # :contentReference[oaicite:13]{index=13}

print("Effective apertures:", myfrac.effective_aperture)

# 5. Visualize the aperture field
plt.imshow(aperture, cmap='viridis')
plt.colorbar(label='Aperture (mm)')
plt.title('Synthetic Fracture Aperture Field')
plt.tight_layout()
plt.savefig('aperture_field.png', dpi=300)
