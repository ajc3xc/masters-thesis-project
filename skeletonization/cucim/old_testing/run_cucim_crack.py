import time
import cupy as cp
import numpy as np
from cucim.skimage.morphology import medial_axis as gpu_medial_axis
from skimage.morphology import medial_axis as cpu_medial_axis, skeletonize as cpu_skeletonize
import imageio.v3 as iio
from pathlib import Path
import matplotlib.pyplot as plt

# Load binary image as NumPy
input_path = Path("/mnt/d/camerer_ml/skeletonization/skeletonide/test/images/horse.png")
img_np = iio.imread(input_path) < 128

# GPU: cuCIM medial_axis (skeleton and width)
img_gpu = cp.array(img_np)
_ = gpu_medial_axis(img_gpu, return_distance=True)
cp.cuda.Stream.null.synchronize()

t0 = time.time()
skel_gpu, dist_gpu = gpu_medial_axis(img_gpu, return_distance=True)
cp.cuda.Stream.null.synchronize()
t_gpu = time.time() - t0
print(f"[GPU] cucim.medial_axis: {t_gpu:.4f} s")

# Convert results to numpy
skel_gpu_np = cp.asnumpy(skel_gpu)
dist_gpu_np = cp.asnumpy(dist_gpu)
widthmap_gpu = np.zeros_like(dist_gpu_np)
widthmap_gpu[skel_gpu_np] = 2 * dist_gpu_np[skel_gpu_np]

# CPU: skimage medial_axis
t0 = time.time()
skel_cpu, dist_cpu = cpu_medial_axis(img_np, return_distance=True)
t_cpu = time.time() - t0
print(f"[CPU] skimage.medial_axis: {t_cpu:.4f} s")

widthmap_cpu = np.zeros_like(dist_cpu)
widthmap_cpu[skel_cpu] = 2 * dist_cpu[skel_cpu]

# CPU: skimage skeletonize (for reference, Zhang-Suen)
skel_zs = cpu_skeletonize(img_np)

# ----- Plot overlays -----
fig, axs = plt.subplots(2, 2, figsize=(12, 12))

# GPU overlay
axs[0, 0].imshow(img_np, cmap="gray")
masked_width_gpu = np.ma.masked_where(widthmap_gpu == 0, widthmap_gpu)
im0 = axs[0, 0].imshow(masked_width_gpu, cmap="jet", alpha=0.7)
axs[0, 0].set_title("cuCIM medial_axis width (over original)")
axs[0, 0].axis('off')
fig.colorbar(im0, ax=axs[0, 0])

# CPU overlay
axs[0, 1].imshow(img_np, cmap="gray")
masked_width_cpu = np.ma.masked_where(widthmap_cpu == 0, widthmap_cpu)
im1 = axs[0, 1].imshow(masked_width_cpu, cmap="jet", alpha=0.7)
axs[0, 1].set_title("skimage medial_axis width (over original)")
axs[0, 1].axis('off')
fig.colorbar(im1, ax=axs[0, 1])

# GPU skeleton
axs[1, 0].imshow(skel_gpu_np, cmap="gray")
axs[1, 0].set_title("cuCIM medial_axis skeleton")
axs[1, 0].axis('off')

# CPU skeletonize (Zhang-Suen) as reference
axs[1, 1].imshow(skel_zs, cmap="gray")
axs[1, 1].set_title("skimage skeletonize (Zhang-Suen)")
axs[1, 1].axis('off')

plt.tight_layout()
plt.show()