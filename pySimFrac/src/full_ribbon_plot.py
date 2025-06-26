import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.morphology import medial_axis, skeletonize

BW_PATH = '/mnt/e/camerer_ml/datasets/METU_Concrete_Crack_Seg/BW_visible/236.png'
mask = imread(BW_PATH, as_gray=True) > 0.5

medial, _ = skeletonize(mask, return_distance=True)
ys, xs = np.nonzero(medial)

# How many pixels away from border to ignore (to avoid artifacts)
EDGE_FILTER = 10

# Store: [x, y, width, normal]
points = []

from sklearn.decomposition import PCA

for y, x in zip(ys, xs):
    # skip points close to border
    if (x < EDGE_FILTER or x >= mask.shape[1]-EDGE_FILTER or
        y < EDGE_FILTER or y >= mask.shape[0]-EDGE_FILTER):
        continue

    # Local window for PCA
    win = 7
    y0, y1 = y - win, y + win + 1
    x0, x1 = x - win, x + win + 1
    # Get local medial axis points
    local = np.column_stack(np.nonzero(medial[y0:y1, x0:x1]))
    if len(local) < 3:
        continue
    local = local + [y0, x0]
    pca = PCA(n_components=2).fit(local)
    normal = pca.components_[1]
    normal = normal / np.linalg.norm(normal)

    # Shoot out along normal until you hit the mask border in both directions
    d1, d2 = 0, 0
    for sign in [+1, -1]:
        for d in range(1, 200):  # max search distance
            yy = int(round(y + normal[0]*d*sign))
            xx = int(round(x + normal[1]*d*sign))
            if (0 <= yy < mask.shape[0]) and (0 <= xx < mask.shape[1]):
                if not mask[yy, xx]:
                    if sign > 0:
                        d1 = d
                    else:
                        d2 = d
                    break
            else:
                break
    width = d1 + d2
    points.append((x, y, width, normal.copy()))

# --- Plotting ---
fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(mask, cmap='gray', alpha=0.7)
widths = np.array([p[2] for p in points])
vmin, vmax = np.percentile(widths, [1, 99])
norm = plt.Normalize(vmin, vmax)

for x, y, width, normal in points:
    pt1 = [x - 0.5*width*normal[1], y - 0.5*width*normal[0]]
    pt2 = [x + 0.5*width*normal[1], y + 0.5*width*normal[0]]
    color = plt.cm.plasma(norm(width))
    ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=color, linewidth=2)

sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
fig.colorbar(sm, ax=ax, label='Width (px)')
ax.set_title('Width-colored normal lines, edge-filtered')
plt.axis('off')
plt.tight_layout()
plt.savefig('width_ribbon_edgeproximity_python.png', dpi=300)
plt.show()
