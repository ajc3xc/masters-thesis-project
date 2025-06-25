import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries
from scipy.ndimage import binary_dilation
from cucim.skimage.morphology import medial_axis
import cupy as cp
from dsepruning.dsepruning import skel_pruning_DSE

# --- User Settings ---
IMG_PATH = '/mnt/e/camerer_ml/datasets/METU_Concrete_Crack_Seg/BW/236.JPG'
THRESHOLD = 0.25
MIN_AREA_PX = 1250
EDGE_PROXIMITY = 10  # pixels for filtering near edge

gray = imread(IMG_PATH, as_gray=True)
bw = (gray > THRESHOLD)
#bw = remove_small_objects(bw, min_size=MIN_AREA_PX)

# Medial axis + DSE pruning
sk_gpu, dist_gpu = medial_axis(cp.array(bw), return_distance=True)
sk, dist = sk_gpu.get(), dist_gpu.get()
pruned = skel_pruning_DSE(sk, dist, min_area_px=MIN_AREA_PX, return_graph=False)
width_medial = np.zeros_like(bw, float)
width_medial[pruned] = dist[pruned] * 2

# --- Filtering skeletons by proximity to image border or crack edge ---
h, w = bw.shape
labeled, num = label(pruned, return_num=True)
main_skel = np.zeros_like(pruned, dtype=bool)

crack_edge = find_boundaries(bw, mode='outer')
dilated_edge = binary_dilation(crack_edge, iterations=EDGE_PROXIMITY)

for r in regionprops(labeled):
    coords = r.coords
    # Within N pixels of image border?
    close_to_border = np.any(
        (coords[:, 0] < EDGE_PROXIMITY) | (coords[:, 0] >= h-EDGE_PROXIMITY) |
        (coords[:, 1] < EDGE_PROXIMITY) | (coords[:, 1] >= w-EDGE_PROXIMITY)
    )
    # Within N pixels of crack edge?
    close_to_crack = np.any(dilated_edge[coords[:, 0], coords[:, 1]])
    if close_to_border or close_to_crack:
        main_skel[coords[:, 0], coords[:, 1]] = True

# --- Plotting ---
def get_local_tangent_normal(y, x, skel, window=5):
    y0, y1 = max(0, y-window), min(skel.shape[0], y+window+1)
    x0, x1 = max(0, x-window), min(skel.shape[1], x+window+1)
    local_points = np.column_stack(np.nonzero(skel[y0:y1, x0:x1]))
    if len(local_points) < 3:
        return np.array([1.0, 0.0]), np.array([0.0, 1.0])
    local_points = local_points + [y0, x0]
    cov = np.cov(local_points.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    tangent = eigvecs[:, np.argmax(eigvals)]
    normal = np.array([-tangent[1], tangent[0]])
    tangent /= np.linalg.norm(tangent)
    normal /= np.linalg.norm(normal)
    return tangent, normal

# --- Show DSE-pruned skeleton before and after filtering ---
plt.figure(figsize=(7,7))
plt.imshow(bw, cmap='gray', alpha=0.5)
plt.imshow(pruned, cmap='hot', alpha=0.8)
plt.title('DSE-Pruned Skeleton (All)')
plt.axis('off')
plt.tight_layout()
plt.savefig('dse_pruned_all.png', dpi=300)

plt.figure(figsize=(7,7))
plt.imshow(bw, cmap='gray', alpha=0.5)
plt.imshow(main_skel, cmap='hot', alpha=0.8)
plt.title('Edge/Border/Near-Crack-Connected Skeleton')
plt.axis('off')
plt.tight_layout()
plt.savefig('dse_pruned_edgeproximity.png', dpi=300)

# --- Plot width-colored normal lines ribbon (filtered skeleton only) ---
fig, ax = plt.subplots(figsize=(8,8))
ax.imshow(np.zeros_like(bw), cmap='gray', vmin=0, vmax=1)  # Black background
ax.imshow(bw, cmap='gray', vmin=0, vmax=1, alpha=1.0)      # White crack

ys, xs = np.nonzero(main_skel)
vmin, vmax = np.nanmin(width_medial[main_skel]), np.nanmax(width_medial[main_skel])

# Plot lines in order of increasing width (thickest last)
lines = []
for y, x in zip(ys, xs):
    w = width_medial[y, x]
    if not np.isfinite(w) or w <= 0: continue
    _, normal = get_local_tangent_normal(y, x, main_skel, window=5)
    pt1 = [x - 0.5 * w * normal[1], y - 0.5 * w * normal[0]]
    pt2 = [x + 0.5 * w * normal[1], y + 0.5 * w * normal[0]]
    lines.append((w, pt1, pt2))
lines.sort(key=lambda t: t[0])
for w, pt1, pt2 in lines:
    color = plt.cm.plasma((w - vmin) / (vmax - vmin))
    ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=color, linewidth=2)
ax.set_title(f"Width-colored normal lines, N={EDGE_PROXIMITY}px border/edge filter")
ax.axis('off')
sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin, vmax))
fig.colorbar(sm, ax=ax, label='Width (px)')
plt.tight_layout()
plt.savefig('width_ribbon_edgeproximity.png', dpi=400)
plt.show()
