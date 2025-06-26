import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 1. Sort skeleton points in order from tip (for single crack)
#   (Already done if your extraction is tip-to-tail)
from skimage.io import imread
from skimage.morphology import skeletonize

# --- Example: Extract skeleton from binary mask ---
# gray = imread(IMG_PATH, as_gray=True)
# bw = (gray > THRESHOLD)
# skel = skeletonize(bw)
IMG_PATH = '/mnt/e/camerer_ml/datasets/METU_Concrete_Crack_Seg/BW_visible/236.png'  # binary mask image


# 1. Get (y, x) coordinates of all skeleton pixels
ys, xs = np.nonzero(skel)
skeleton_coords = list(zip(ys, xs))

# 2. Build a map of neighbors for each skeleton pixel
from collections import defaultdict, deque

neighbor_offsets = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
neighbors = defaultdict(list)
coord_set = set(skeleton_coords)
for y, x in skeleton_coords:
    for dy, dx in neighbor_offsets:
        nb = (y + dy, x + dx)
        if nb in coord_set:
            neighbors[(y, x)].append(nb)

# 3. Find endpoints (pixels with only one neighbor)
endpoints = [pt for pt, nbs in neighbors.items() if len(nbs) == 1]
print(f"Found {len(endpoints)} skeleton endpoints.")

# 4. BFS to order the skeleton points from one endpoint (the 'tip')
#   (If more than one crack, you may get multiple endpoints—pick one)
if endpoints:
    start = endpoints[0]
else:
    # fallback: pick an arbitrary point (not ideal, but prevents crash)
    start = skeleton_coords[0]

visited = set()
ordered_skeleton = []
q = deque([start])
while q:
    pt = q.popleft()
    if pt in visited:
        continue
    ordered_skeleton.append(pt)
    visited.add(pt)
    for nb in neighbors[pt]:
        if nb not in visited:
            q.append(nb)

# Convert ordered list of (y, x) to numpy array as (N, 2)
skeleton_points = np.array([(x, y) for y, x in ordered_skeleton])
# skeleton_points[:, 0] = x, skeleton_points[:, 1] = y

print(f"Ordered skeleton length: {len(skeleton_points)}")

# 2. Compute distance along skeleton (r)
#   (Assume skeleton_points is in order)
distances = [0]
for i in range(1, len(skeleton_points)):
    dx = skeleton_points[i][0] - skeleton_points[i-1][0]
    dy = skeleton_points[i][1] - skeleton_points[i-1][1]
    distances.append(distances[-1] + np.hypot(dx, dy))
r = np.array(distances)

# Optionally convert r and widths to mm
# r = r * PIXEL_SIZE
# widths_mm = widths * PIXEL_SIZE

# 3. Fit FM model: w(r) = C * sqrt(r)
def fm_model(r, C):
    return C * np.sqrt(r)

# Only fit near the tip (first N points), e.g. N=50
fit_mask = (r > 0) & (r < np.percentile(r, 30)) & (widths > 0) & np.isfinite(widths)
r_fit = r[fit_mask]
w_fit = widths[fit_mask]

if len(r_fit) > 2:
    popt, _ = curve_fit(fm_model, r_fit, w_fit)
    C_fit = popt[0]
else:
    print("Not enough points to fit FM model.")
    C_fit = np.nan

# 4. Predict FM-based width along the skeleton
w_fm = fm_model(r, C_fit)

# 5. Plot both: measured and FM-predicted width
plt.figure(figsize=(12,6))
plt.plot(r, widths, '.', ms=2, label='Measured (normal-based)')
plt.plot(r, w_fm, '-', lw=2, label='FM fit: $C\sqrt{r}$')
plt.xlabel("Distance from tip (px)")
plt.ylabel("Crack width (px)")
plt.title("Crack width: Measured vs. Fracture Mechanics Fit")
plt.legend()
plt.tight_layout()
plt.savefig("crack_width_fm_fit.png", dpi=300)
plt.close()
print("Saved FM fit plot as crack_width_fm_fit.png")
