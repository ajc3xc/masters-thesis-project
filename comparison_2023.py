import numpy as np
from skimage.io import imread
from skimage.morphology import skeletonize, remove_small_objects
from skimage.segmentation import find_boundaries
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

# --- Load and binarize ---
IMG_PATH = '/mnt/e/camerer_ml/datasets/METU_Concrete_Crack_Seg/BW/236.JPG'
THRESHOLD = 0.25
MIN_SIZE = 1000

gray = imread(IMG_PATH, as_gray=True)
bw = (gray > THRESHOLD)
bw = remove_small_objects(bw, min_size=MIN_SIZE)

# (1) Original mask
plt.figure(figsize=(16,3))
plt.subplot(1,5,1)
plt.imshow(bw, cmap='gray')
plt.title("Original Mask")
plt.axis('off')

# (2) Skeleton
skel = skeletonize(bw)
plt.subplot(1,5,2)
plt.imshow(bw, cmap='gray')
plt.imshow(skel, cmap='Blues', alpha=0.7)
plt.title("Skeleton")
plt.axis('off')

# (3) Boundaries (edges)
boundaries = find_boundaries(bw, mode='outer')
plt.subplot(1,5,3)
plt.imshow(bw, cmap='gray')
plt.imshow(boundaries, cmap='Reds', alpha=0.7)
plt.title("Mask Boundaries")
plt.axis('off')

# (4) Skeleton with normal vectors
Y_skel, X_skel = np.nonzero(skel)
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

plt.subplot(1,5,4)
plt.imshow(bw, cmap='gray')
plt.imshow(skel, cmap='Blues', alpha=0.7)
# Plot normal vectors at every Nth skeleton point
Nskip = max(1, len(Y_skel)//200)
for y, x in zip(Y_skel[::Nskip], X_skel[::Nskip]):
    _, normal = get_local_tangent_normal(y, x, skel, window=5)
    plt.arrow(x, y, 8*normal[1], 8*normal[0], color='lime', width=0.5, head_width=2)
plt.title("Normals at Skeleton")
plt.axis('off')

# (5) Width lines (feature points, as in paper)
Y_edge, X_edge = np.nonzero(boundaries)
edge_coords = np.column_stack([Y_edge, X_edge])
tree = cKDTree(edge_coords)
profile_half_length = max(bw.shape)  # just go until you exit the mask!
width_lines = []
width_values = []
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
                # last in-mask point is the true edge
                edge = np.array([y, x]) + normal * (i-1) * sign
                found_edges.append(np.round(edge).astype(int))
                break
    if len(found_edges) == 2:
        width_lines.append([found_edges[0], [y, x], found_edges[1]])
        width_values.append(np.linalg.norm(found_edges[0] - found_edges[1]))
    if len(found_edges) == 2:
        width_lines.append([found_edges[0], [y, x], found_edges[1]])
        width_values.append(np.linalg.norm(found_edges[0] - found_edges[1]))

plt.subplot(1,5,5)
plt.imshow(bw, cmap='gray')
for (pt1, center, pt2) in width_lines:
    plt.plot([pt1[1], pt2[1]], [pt1[0], pt2[0]], color='red', linewidth=1)
    plt.plot(center[1], center[0], 'bo', markersize=1)
plt.title("Final Width Lines")
plt.axis('off')

plt.tight_layout()
plt.show()
