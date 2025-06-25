import cupy as cp
from cucim.skimage.morphology import medial_axis
from dsepruning.dsepruning import skel_pruning_DSE
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.filters import gabor
from skimage.morphology import remove_small_objects, binary_opening, disk

# --- USER SETTINGS ---
IMG_PATH = '/mnt/e/camerer_ml/datasets/METU_Concrete_Crack_Seg/BW/236.JPG'
THRESHOLD = 0.25
MIN_AREA_PX = 1000

# --- LOAD & BINARIZE ---
gray = imread(IMG_PATH, as_gray=True)
bw = (gray > THRESHOLD)

from skimage.transform import rescale

scale_factor = 0.5  # 4x speedup for 4K → 1K
gray = rescale(gray, scale_factor, channel_axis=None, anti_aliasing=True)
bw = rescale(bw.astype(float), scale_factor, channel_axis=None) > 0.5


# --- GABOR ENHANCEMENT (STACK MAX OVER MULTIPLE ORIENTATIONS) ---
thetas = np.linspace(0, np.pi, 4, endpoint=False)
filt_stack = []
for theta in thetas:
    filt_real, _ = gabor(gray, frequency=0.08, theta=theta)
    filt_stack.append(filt_real)
filt_max = np.max(np.stack(filt_stack), axis=0)
# Normalize and threshold to keep only strong Gabor responses
filt_max = (filt_max - filt_max.min()) / (filt_max.ptp() + 1e-8)
gabor_mask = filt_max > np.percentile(filt_max, 80)  # Try 80–90%

# --- COMBINE GABOR WITH MASK ---
# Only keep regions present in both the binary mask and the Gabor-enhanced mask
cleaned = bw & gabor_mask
# After combining gabor_mask and bw_small:
cleaned = remove_small_objects(cleaned, min_size=200)           # Try higher min_size
cleaned = binary_opening(cleaned, disk(3))                      # Remove noise and tiny bridges

# --- MEDIAL AXIS + DSE (GPU) ---
def width_medial_dse(bw):
    print("Using GPU for medial axis...")
    sk_gpu, dist_gpu = medial_axis(cp.array(bw), return_distance=True)
    sk, dist = sk_gpu.get(), dist_gpu.get()
    pruned = skel_pruning_DSE(sk, dist, min_area_px=MIN_AREA_PX, return_graph=False)
    w = np.zeros_like(bw, float)
    w[pruned] = dist[pruned] * 2
    return w, pruned

w_medial, pruned = width_medial_dse(bw)
w_medial_gabor, pruned_gabor = width_medial_dse(cleaned)

# --- PLOT ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, title, wmap, skel in [
    (axes[0], 'Medial+DSE (default)', w_medial, pruned),
    (axes[1], 'Medial+DSE (Gabor preproc)', w_medial_gabor, pruned_gabor)
]:
    ys, xs = np.nonzero(skel)
    ax.imshow(bw, cmap='gray')
    sc = ax.scatter(xs, ys, c=wmap[ys, xs], cmap='plasma', s=10)
    ax.set_title(title)
    ax.axis('off')
    plt.colorbar(sc, ax=ax, label='Width (px)')
plt.tight_layout()
plt.show()
