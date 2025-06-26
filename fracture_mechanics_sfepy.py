import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.morphology import skeletonize
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit
import pandas as pd
from collections import defaultdict, deque

# ========== CONFIG ==========
IMG_PATH = '/mnt/e/camerer_ml/datasets/METU_Concrete_Crack_Seg/BW_visible/236.png'  # <- UPDATE THIS
PIXEL_SIZE = 1.0  # mm per pixel (set to 1 if unknown, update if you know)
THRESHOLD = 0.25  # For grayscale thresholding. Ignore if binary image.
OUTPUT_DIR = "physics_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== LOAD IMAGE ==========
img = imread(IMG_PATH)
if img.ndim == 3:
    # Convert to grayscale if RGB
    img_gray = np.mean(img, axis=2) if img.shape[2] == 3 else img[...,0]
else:
    img_gray = img

# ========== BINARIZE (BW mask) ==========
if img_gray.max() > 1.0:  # Convert if 8bit
    img_gray = img_gray / 255.
bw = (img_gray > THRESHOLD)
print(f"Mask shape: {bw.shape}, Crack area: {np.sum(bw)} px")

plt.imsave(os.path.join(OUTPUT_DIR, 'input_mask.png'), bw, cmap='gray')

import cupy as cp
from cucim.skimage.morphology import medial_axis
from dsepruning.dsepruning import skel_pruning_DSE

MIN_AREA_PX = 1100  # Set as desired (try 1000, or lower for short cracks)
GPU_AVAILABLE = True  # Set to False to use CPU

def width_medial_dse(bw, min_area_px=MIN_AREA_PX, gpu=GPU_AVAILABLE):
    """Medial-axis + EDT + DSE pruning for robust crack centerline/width."""
    if gpu:
        print("Using GPU for medial axis...")
        sk_gpu, dist_gpu = medial_axis(cp.array(bw), return_distance=True)
        sk, dist = sk_gpu.get(), dist_gpu.get()
    else:
        sk, dist = medial_axis(bw, return_distance=True)
    pruned = skel_pruning_DSE(sk, dist, min_area_px=min_area_px, return_graph=False)
    widths = np.zeros_like(sk, dtype=float)
    widths[pruned] = dist[pruned] * 2
    ys, xs = np.nonzero(pruned)
    skeleton_points = np.column_stack((xs, ys))
    widths = widths[ys, xs]
    return sk, skeleton_points, widths

# --- Use in main pipeline:
#skel, skeleton_points, widths = width_medial_dse(bw)

# After:
skel, skeleton_points, widths = width_medial_dse(bw)
# widths is already 2*dist at each skeleton/medial axis point!
widths_mm = widths * PIXEL_SIZE

print(f"Extracted {len(skeleton_points)} skeleton points with DSE-pruned medial axis.")


plt.figure(figsize=(10, 10))
plt.imshow(bw, cmap='gray', alpha=0.8)
plt.imshow(skel, cmap='hot', alpha=0.5)
plt.title("Medial Axis (Before Pruning)")
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "medial_axis_raw.png"), dpi=300)
plt.close()
print("Saved: medial_axis_raw.png")

# ========== MEASURE WIDTHS ALONG NORMALS ==========
def get_normal(y, x, window=7):
    y0, y1 = max(0, y-window), min(skel.shape[0], y+window+1)
    x0, x1 = max(0, x-window), min(skel.shape[1], x+window+1)
    pts = np.column_stack(np.nonzero(skel[y0:y1, x0:x1]))
    if len(pts) < 3:
        return np.array([0,1])
    pts = pts + [y0, x0]
    pca = PCA(n_components=2).fit(pts)
    v = pca.components_[1]
    v = v / np.linalg.norm(v)
    return v

widths = []
normal_ends = []
for (x, y) in skeleton_points:
    n = get_normal(y, x)
    dists = []
    ends = []
    for direction in [+1, -1]:
        for i in range(1, 100): # max width search (pixels)
            py = int(round(y + direction * n[0] * i))
            px = int(round(x + direction * n[1] * i))
            if 0 <= py < bw.shape[0] and 0 <= px < bw.shape[1]:
                if not bw[py, px]:
                    dists.append(i)
                    ends.append((px, py))
                    break
            else:
                dists.append(i)
                ends.append((px, py))
                break
    if len(dists)==2:
        widths.append(sum(dists))
        normal_ends.append((ends[0], ends[1]))
    else:
        widths.append(np.nan)
        normal_ends.append(((x, y), (x, y)))

widths = np.array(widths)
widths_mm = widths * PIXEL_SIZE

# ========== DISTANCE ALONG SKELETON ==========
distances = [0]
for i in range(1, len(skeleton_points)):
    dx = skeleton_points[i][0] - skeleton_points[i-1][0]
    dy = skeleton_points[i][1] - skeleton_points[i-1][1]
    distances.append(distances[-1] + np.hypot(dx, dy))
distances = np.array(distances) * PIXEL_SIZE

# ========== FRACTURE MECHANICS FIT ==========
def fm_model(r, C):
    return C * np.sqrt(r)
fit_mask = (distances > 0) & np.isfinite(widths_mm) & (widths_mm > 0)
r_fit = distances[fit_mask]
w_fit = widths_mm[fit_mask]
if len(r_fit) > 5:
    popt, _ = curve_fit(fm_model, r_fit, w_fit)
    C_fit = popt[0]
else:
    print("Not enough points for FM fit.")
    C_fit = np.nan
w_fm = fm_model(distances, C_fit)

# ========== SAVE CSV ==========
df = pd.DataFrame({
    'x': skeleton_points[:,0],
    'y': skeleton_points[:,1],
    'distance_px': distances / PIXEL_SIZE,
    'distance_mm': distances,
    'width_px': widths,
    'width_mm': widths_mm,
    'fm_width_mm': w_fm
})
csv_path = os.path.join(OUTPUT_DIR, "crack_widths_fm.csv")
df.to_csv(csv_path, index=False)
print(f"Saved table: {csv_path}")

# ========== PLOT WIDTH VS DISTANCE ==========
plt.figure(figsize=(12,6))
plt.plot(distances, widths_mm, '.', ms=3, label='Measured')
plt.plot(distances, w_fm, '-', lw=2, label=f'FM fit: C={C_fit:.2f}')
plt.xlabel("Distance from tip (mm)")
plt.ylabel("Crack width (mm)")
plt.title("Crack width along skeleton: Measured vs. Fracture Mechanics Fit")
plt.legend()
plt.tight_layout()
plot1_path = os.path.join(OUTPUT_DIR, "crack_width_vs_distance.png")
plt.savefig(plot1_path, dpi=300)
plt.close()
print(f"Saved plot: {plot1_path}")

# ========== PLOT NORMAL RIBBONS ==========
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(bw, cmap='gray', alpha=0.8)
norm = plt.Normalize(np.nanmin(widths_mm), np.nanmax(widths_mm))

for i, ((x, y), w) in enumerate(zip(skeleton_points, widths_mm)):
    if np.isnan(w) or w <= 0:
        continue
    # Tangent/normal from local PCA
    y0, y1 = max(0, y-5), min(bw.shape[0], y+6)
    x0, x1 = max(0, x-5), min(bw.shape[1], x+6)
    pts = np.column_stack(np.nonzero(skel[y0:y1, x0:x1]))
    if len(pts) < 2:
        continue
    pts = pts + [y0, x0]
    pca = PCA(n_components=2).fit(pts)
    n = pca.components_[1]  # normal direction
    n = n / np.linalg.norm(n)
    pt1 = [x - 0.5 * w * n[1], y - 0.5 * w * n[0]]
    pt2 = [x + 0.5 * w * n[1], y + 0.5 * w * n[0]]
    color = plt.cm.plasma(norm(w))
    ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=color, linewidth=2)
ax.set_title('Crack width (measured) as colored ribbons')
sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
fig.colorbar(sm, ax=ax, label='Measured Width (mm)')
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ribbons_measured.png"), dpi=400)
plt.close()

# ========== PLOT NORMAL RIBBONS (FM PREDICTED) ==========
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(bw, cmap='gray', alpha=0.8)
norm_fm = plt.Normalize(np.nanmin(w_fm), np.nanmax(w_fm))

for i, ((x, y), wf) in enumerate(zip(skeleton_points, w_fm)):
    if np.isnan(wf) or wf <= 0:
        continue
    # Tangent/normal from local PCA
    y0, y1 = max(0, y-5), min(bw.shape[0], y+6)
    x0, x1 = max(0, x-5), min(bw.shape[1], x+6)
    pts = np.column_stack(np.nonzero(skel[y0:y1, x0:x1]))
    if len(pts) < 2:
        continue
    pts = pts + [y0, x0]
    pca = PCA(n_components=2).fit(pts)
    n = pca.components_[1]  # normal direction
    n = n / np.linalg.norm(n)
    pt1 = [x - 0.5 * wf * n[1], y - 0.5 * wf * n[0]]
    pt2 = [x + 0.5 * wf * n[1], y + 0.5 * wf * n[0]]
    color = plt.cm.inferno(norm_fm(wf))
    ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=color, linewidth=2)
ax.set_title('Crack width (FM predicted) as colored ribbons')
sm = plt.cm.ScalarMappable(cmap='inferno', norm=norm_fm)
fig.colorbar(sm, ax=ax, label='FM Predicted Width (mm)')
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ribbons_fm.png"), dpi=400)
plt.close()

print("\nAll physics-informed outputs complete. Check the 'physics_outputs' folder!")
