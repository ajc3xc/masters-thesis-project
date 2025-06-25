import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.morphology import skeletonize, remove_small_objects
from scipy.ndimage import distance_transform_edt
from cucim.skimage.morphology import medial_axis
import cupy as cp
from dsepruning.dsepruning import skel_pruning_DSE

# --- User Settings ---
IMG_PATH = '/mnt/e/camerer_ml/datasets/METU_Concrete_Crack_Seg/BW/236.JPG'
THRESHOLD = 0.25
MIN_AREA_PX = 1000

gray = imread(IMG_PATH, as_gray=True)
bw = (gray > THRESHOLD)
#bw = remove_small_objects(bw, min_size=MIN_AREA_PX)

# Medial axis + DSE pruning
sk_gpu, dist_gpu = medial_axis(cp.array(bw), return_distance=True)
sk, dist = sk_gpu.get(), dist_gpu.get()
pruned = skel_pruning_DSE(sk, dist, min_area_px=MIN_AREA_PX, return_graph=False)
# Width: 2 * distance transform at pruned skeleton points
width_medial = np.zeros_like(bw, float)
width_medial[pruned] = dist[pruned] * 2

# Plot
import numpy as np
import matplotlib.pyplot as plt

# Assuming you have:
# bw         -- binary mask
# pruned     -- skeleton points (mask, bool)
# w_medial   -- width map at skeleton points (width in px)

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

plt.figure(figsize=(8,8))
plt.imshow(bw, cmap='gray', alpha=0.2)
ys, xs = np.nonzero(pruned)
vmin, vmax = np.nanmin(width_medial[pruned]), np.nanmax(width_medial[pruned])
for y, x in zip(ys, xs):
    w = width_medial[y, x]
    if not np.isfinite(w) or w <= 0: continue
    _, normal = get_local_tangent_normal(y, x, pruned, window=5)
    pt1 = [x - 0.5 * w * normal[1], y - 0.5 * w * normal[0]]
    pt2 = [x + 0.5 * w * normal[1], y + 0.5 * w * normal[0]]
    color = plt.cm.plasma((w - vmin) / (vmax - vmin))
    plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=color, linewidth=2)
plt.title("Skeleton colored and widened by measured width")
plt.axis('off')
plt.tight_layout()
plt.show()

