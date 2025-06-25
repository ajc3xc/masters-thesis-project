import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.morphology import skeletonize, remove_small_objects
from skimage.segmentation import find_boundaries
from scipy.spatial import cKDTree
from cucim.skimage.morphology import medial_axis
from dsepruning.dsepruning import skel_pruning_DSE
import cupy as cp

# ---------------- User Settings ----------------
IMG_PATH    = '/mnt/e/camerer_ml/datasets/METU_Concrete_Crack_Seg/BW/236.JPG'
THRESHOLD   = 0.25
MIN_AREA_PX = 1000
GPU_AVAILABLE = True
OUT_DIR = 'plot_outputs'
os.makedirs(OUT_DIR, exist_ok=True)

# --- Load and binarize ---
gray = imread(IMG_PATH, as_gray=True)
bw = (gray > THRESHOLD)
bw = remove_small_objects(bw, min_size=MIN_AREA_PX)

# --- (A) Medial+DSE width (cucim GPU) ---
def width_medial_dse(bw, min_area_px=MIN_AREA_PX, gpu_available=GPU_AVAILABLE):
    if gpu_available:
        sk_gpu, dist_gpu = medial_axis(cp.array(bw), return_distance=True)
        sk, dist = sk_gpu.get(), dist_gpu.get()
    else:
        from skimage.morphology import medial_axis as cpu_medial_axis
        sk, dist = cpu_medial_axis(bw, return_distance=True)
    pruned = skel_pruning_DSE(sk, dist, min_area_px=min_area_px, return_graph=False)
    w = np.zeros_like(bw, float)
    w[pruned] = dist[pruned] * 2
    return w, pruned

w_medial, pruned = width_medial_dse(bw)

# --- (B) Faithful skeleton-normal width (walk to edge, color-mapped) ---
#skel = skeletonize(bw)
Y_skel, X_skel = np.nonzero(pruned)
boundaries = find_boundaries(bw, mode='outer')
profile_half_length = max(bw.shape)  # large enough to always reach the edge

width_map = np.zeros_like(bw, float)
width_lines = []

for (y, x) in zip(Y_skel, X_skel):
    # Local PCA for tangent/normal
    y0, y1 = max(0, y-5), min(pruned.shape[0], y+6)
    x0, x1 = max(0, x-5), min(pruned.shape[1], x+6)
    local_points = np.column_stack(np.nonzero(pruned[y0:y1, x0:x1]))
    if len(local_points) < 3:
        continue
    local_points = local_points + [y0, x0]
    mean = local_points.mean(axis=0)
    cov = np.cov(local_points.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    tangent = eigvecs[:, np.argmax(eigvals)]
    normal = np.array([-tangent[1], tangent[0]])
    tangent /= np.linalg.norm(tangent)
    normal /= np.linalg.norm(normal)

    found_edges = []
    for sign in [-1, 1]:
        for i in range(1, profile_half_length):
            p = np.array([y, x]) + normal * i * sign
            pi = np.round(p).astype(int)
            if not (0 <= pi[0] < bw.shape[0] and 0 <= pi[1] < bw.shape[1]):
                break
            if not bw[pi[0], pi[1]]:
                edge = np.array([y, x]) + normal * (i-1) * sign
                found_edges.append(np.round(edge).astype(int))
                break
    if len(found_edges) == 2:
        width = np.linalg.norm(found_edges[0] - found_edges[1])
        width_map[y, x] = width
        width_lines.append([found_edges[0], [y, x], found_edges[1], width])

# --- Plot and save ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Medial+DSE
# Medial+DSE (unchanged)
ys, xs = np.nonzero(pruned)
sc1 = axes[0].scatter(xs, ys, c=w_medial[ys, xs], cmap='plasma', s=10)
axes[0].imshow(bw, cmap='gray', alpha=0.5)
axes[0].set_title("Medial+DSE (cucim)")
axes[0].axis('off')
plt.colorbar(sc1, ax=axes[0], label='Width (px)')

# Skeleton-normal, color-mapped
ys2, xs2 = np.nonzero(width_map)
sc2 = axes[1].scatter(xs2, ys2, c=width_map[ys2, xs2], cmap='plasma', s=10)
axes[1].imshow(bw, cmap='gray', alpha=0.5)
axes[1].set_title("Faithful Profile-Normal (2023, color-mapped)")
axes[1].axis('off')
plt.colorbar(sc2, ax=axes[1], label='Width (px)')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "width_comparison_colormap.png"), dpi=600)
plt.show()

plt.figure(figsize=(7,7))
plt.imshow(bw, cmap='gray', alpha=0.5)
for pt1, center, pt2, w in width_lines:
    plt.plot([pt1[1], pt2[1]], [pt1[0], pt2[0]], color='white', alpha=0.5, linewidth=1)
plt.scatter(xs2, ys2, c=width_map[ys2, xs2], cmap='plasma', s=5)
plt.title("Width Lines and Colormapped Skeleton (Illustration)")
plt.axis('off')
plt.colorbar(label='Width (px)')
plt.savefig(os.path.join(OUT_DIR, "widthlines_supplementary.png"), dpi=400)
plt.show()