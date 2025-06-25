import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.morphology import skeletonize, remove_small_objects
from skimage.segmentation import find_boundaries
from scipy.ndimage import distance_transform_edt
from sklearn.decomposition import PCA

from cucim.skimage.morphology import medial_axis
import cupy as cp
from dsepruning.dsepruning import skel_pruning_DSE

# --- User Settings ---
IMG_PATH = '/mnt/e/camerer_ml/datasets/METU_Concrete_Crack_Seg/BW/236.JPG'
THRESHOLD = 0.25
MIN_AREA_PX = 1250
OUT_DIR = 'plot_outputs'
os.makedirs(OUT_DIR, exist_ok=True)

# --- Load, binarize, remove small objects ---
gray = imread(IMG_PATH, as_gray=True)
bw = (gray > THRESHOLD)
bw = remove_small_objects(bw, min_size=MIN_AREA_PX)

# --- (A) Medial Axis + DSE Pruning (GPU) ---
def width_medial_dse(bw, min_area_px=MIN_AREA_PX):
    sk_gpu, dist_gpu = medial_axis(cp.array(bw), return_distance=True)
    sk, dist = sk_gpu.get(), dist_gpu.get()
    pruned = skel_pruning_DSE(sk, dist, min_area_px=min_area_px, return_graph=False)
    w = np.zeros_like(bw, float)
    w[pruned] = dist[pruned] * 2
    return w, pruned

w_medial, pruned = width_medial_dse(bw)

# --- (B) Profile-Normal (Faithful 2023) ---
skel = skeletonize(bw)
Y_skel, X_skel = np.nonzero(skel)
profile_half_length = max(bw.shape)
width_map = np.zeros_like(bw, float)

def get_local_tangent_normal(y, x, skel, window=5):
    y0, y1 = max(0, y-window), min(skel.shape[0], y+window+1)
    x0, x1 = max(0, x-window), min(skel.shape[1], x+window+1)
    local_points = np.column_stack(np.nonzero(skel[y0:y1, x0:x1]))
    if len(local_points) < 3:
        return np.array([1.0, 0.0]), np.array([0.0, 1.0])
    local_points = local_points + [y0, x0]
    mean = local_points.mean(axis=0)
    cov = np.cov(local_points.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    tangent = eigvecs[:, np.argmax(eigvals)]
    normal = np.array([-tangent[1], tangent[0]])
    tangent /= np.linalg.norm(tangent)
    normal /= np.linalg.norm(normal)
    return tangent, normal

for (y, x) in zip(Y_skel, X_skel):
    tangent, normal = get_local_tangent_normal(y, x, skel, window=5)
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

# --- (C) PCA-Local Width ---
def width_pca_local(bw, skel, dist_map, patch_scale=1.5, min_points=4):
    width_map = np.zeros_like(skel, float)
    Y, X = np.nonzero(skel)
    for y, x in zip(Y, X):
        r = int(max(4, dist_map[y, x] * patch_scale))
        y0, y1 = max(0, y - r), min(bw.shape[0], y + r + 1)
        x0, x1 = max(0, x - r), min(bw.shape[1], x + r + 1)
        patch = bw[y0:y1, x0:x1]
        # Get coordinates of edge points (bw==1 and 4-neighbor to 0)
        edges = np.argwhere((patch) & (
            (np.pad(patch, 1)[1:-1, :-2] == 0) |
            (np.pad(patch, 1)[1:-1, 2:] == 0) |
            (np.pad(patch, 1)[:-2, 1:-1] == 0) |
            (np.pad(patch, 1)[2:, 1:-1] == 0)
        ))
        if len(edges) < min_points:
            continue
        edges += [y0, x0]
        pca = PCA(n_components=2)
        pca.fit(edges)
        minor_axis = pca.components_[1]
        proj = (edges - np.array([y, x])) @ minor_axis
        width_map[y, x] = proj.max() - proj.min()
    return width_map

dist_map = distance_transform_edt(bw)
w_pca = width_pca_local(bw, skel, dist_map, patch_scale=1.5)

# --- (D) Adaptive PCA-Local (patch size grows with width) ---
def width_adaptive_pca(bw, skel, dist_map, base_patch=8, min_points=4):
    width_map = np.zeros_like(skel, float)
    Y, X = np.nonzero(skel)
    for y, x in zip(Y, X):
        r = int(max(base_patch, dist_map[y, x] * 2))
        y0, y1 = max(0, y - r), min(bw.shape[0], y + r + 1)
        x0, x1 = max(0, x - r), min(bw.shape[1], x + r + 1)
        patch = bw[y0:y1, x0:x1]
        edges = np.argwhere((patch) & (
            (np.pad(patch, 1)[1:-1, :-2] == 0) |
            (np.pad(patch, 1)[1:-1, 2:] == 0) |
            (np.pad(patch, 1)[:-2, 1:-1] == 0) |
            (np.pad(patch, 1)[2:, 1:-1] == 0)
        ))
        if len(edges) < min_points:
            continue
        edges += [y0, x0]
        pca = PCA(n_components=2)
        pca.fit(edges)
        minor_axis = pca.components_[1]
        proj = (edges - np.array([y, x])) @ minor_axis
        width_map[y, x] = proj.max() - proj.min()
    return width_map

w_adap_pca = width_adaptive_pca(bw, skel, dist_map, base_patch=8)

# --- (Optional) Distance Ridge width map ---
ridge_map = distance_transform_edt(bw) * 2  # "Raw" width map

# --- Plot all methods ---
labels = [
    ("Medial+DSE (cucim)", w_medial, pruned),
    ("Profile-Normal (2023)", width_map, skel),
    ("PCA-Local", w_pca, skel),
    ("Adaptive PCA-Local", w_adap_pca, skel),
    ("Distance Ridge", ridge_map, bw)
]

fig, axes = plt.subplots(1, len(labels), figsize=(5*len(labels), 6))
for ax, (label, wmap, skel_mask) in zip(axes, labels):
    ys, xs = np.nonzero(skel_mask)
    sc = ax.scatter(xs, ys, c=wmap[ys, xs], cmap='plasma', s=10)
    ax.imshow(bw, cmap='gray', alpha=0.5)
    ax.set_title(label)
    ax.axis('off')
    plt.colorbar(sc, ax=ax, label='Width (px)')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "width_comparison_allmethods.png"), dpi=600)
plt.show()
