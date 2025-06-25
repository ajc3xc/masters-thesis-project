import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label
from cucim.skimage.morphology import medial_axis
import cupy as cp
from dsepruning.dsepruning import skel_pruning_DSE

# --- User Settings ---
IMG_PATH = '/mnt/e/camerer_ml/datasets/METU_Concrete_Crack_Seg/BW/236.JPG'
THRESHOLD = 0.25
MIN_AREA_PX = 1250

gray = imread(IMG_PATH, as_gray=True)
bw = (gray > THRESHOLD)
# bw = remove_small_objects(bw, min_size=MIN_AREA_PX)

# Medial axis + DSE pruning
sk_gpu, dist_gpu = medial_axis(cp.array(bw), return_distance=True)
sk, dist = sk_gpu.get(), dist_gpu.get()
pruned = skel_pruning_DSE(sk, dist, min_area_px=MIN_AREA_PX, return_graph=False)
width_medial = np.zeros_like(bw, float)
width_medial[pruned] = dist[pruned] * 2

# Helper: get local tangent/normal as before
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

def plot_width_ribbon(ax, mask, ys, xs, widths, vmin, vmax, title, outname):
    # Sort by width (shortest first, longest last)
    lines = []
    for y, x in zip(ys, xs):
        w = widths[y, x]
        if not np.isfinite(w) or w <= 0: continue
        _, normal = get_local_tangent_normal(y, x, mask, window=5)
        pt1 = [x - 0.5 * w * normal[1], y - 0.5 * w * normal[0]]
        pt2 = [x + 0.5 * w * normal[1], y + 0.5 * w * normal[0]]
        lines.append((w, pt1, pt2))
    lines.sort(key=lambda t: t[0])  # Plot thickest last
    ax.imshow(np.zeros_like(mask), cmap='gray', vmin=0, vmax=1)
    ax.imshow(mask, cmap='gray', vmin=0, vmax=1, alpha=1.0)
    for w, pt1, pt2 in lines:
        color = plt.cm.plasma((w - vmin) / (vmax - vmin))
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=color, linewidth=2)
    ax.set_title(title)
    ax.axis('off')
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin, vmax))
    plt.colorbar(sm, ax=ax, label='Width (px)')
    plt.tight_layout()
    plt.savefig(outname, dpi=400)
    print(f"Saved: {outname}")

# --- Method 1: Main trunk only ---
labeled, num = label(pruned, return_num=True)
sizes = np.bincount(labeled.ravel())
sizes[0] = 0  # ignore background
main_label = np.argmax(sizes)
main_skel = labeled == main_label
ys_main, xs_main = np.nonzero(main_skel)

fig1, ax1 = plt.subplots(figsize=(8,8))
vmin, vmax = np.nanmin(width_medial[pruned]), np.nanmax(width_medial[pruned])
plot_width_ribbon(ax1, bw, ys_main, xs_main, width_medial, vmin, vmax,
    "Width-colored normal lines (main trunk)", "ribbon_maintrunk.png")

# --- Method 2: All pruned skeleton ---
ys_all, xs_all = np.nonzero(pruned)
fig2, ax2 = plt.subplots(figsize=(8,8))
plot_width_ribbon(ax2, bw, ys_all, xs_all, width_medial, vmin, vmax,
    "Width-colored normal lines (all pruned)", "ribbon_allpruned.png")

plt.show()
