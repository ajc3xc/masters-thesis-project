#!/usr/bin/env python3
"""
clean_hybrid_crack_width.py

– Binarize input mask
– Full‐field medial‐axis + EDT width map
– Prune skeleton (DSE) on GPU
– Extract longest centerline (drop side branches)
– Profile‐normal widths along that centerline
– LEFM √r fit
– Hybrid width = max(medial, profile, LEFM)
– Plots: medial, profile, LEFM, hybrid all on same centerline with unified cmap
– CSV export of all three widths + distance

Requires: numpy, scipy, scikit‐image, cupy+cucim, dsepruning, sklearn, matplotlib, pandas
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage.io import imread
from scipy.ndimage import distance_transform_edt
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from collections import defaultdict, deque

# GPU medial + DSE pruning
import cupy as cp
from cucim.skimage.morphology import medial_axis
from dsepruning.dsepruning import skel_pruning_DSE

# ──────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ──────────────────────────────────────────────────────────
IMG_PATH    = '/mnt/e/camerer_ml/datasets/METU_Concrete_Crack_Seg/BW_visible/236.png'
PIXEL_SIZE  = 1.0     # mm per pixel
THRESHOLD   = 0.5     # grayscale → binary
MIN_AREA_PX = 1000    # DSE pruning area
PCA_WINDOW  = 11      # window for PCA‐based normal estimation (odd)

OUT_DIR = "physics_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# unified colormap for all skeleton plots
CMAP = plt.cm.plasma


# ──────────────────────────────────────────────────────────
# 2. LOAD + BINARIZE
# ──────────────────────────────────────────────────────────
img = imread(IMG_PATH)
if img.ndim == 3:
    gray = img.mean(axis=2)
else:
    gray = img.astype(float)
if gray.max() > 1.0:
    gray /= 255.0
bw = gray > THRESHOLD
print(f"[1] mask: {bw.shape}, crack area={bw.sum()} px")
plt.imsave(os.path.join(OUT_DIR, 'input_mask.png'), bw, cmap='gray')


# ──────────────────────────────────────────────────────────
# 3. FULL-FIELD MEDIAL-AXIS + EDT
# ──────────────────────────────────────────────────────────
dt = distance_transform_edt(bw)      # Euclidean distance to background
width_medial_map = dt * 2.0          # crack opening in px
np.save(os.path.join(OUT_DIR, 'width_medial.npy'), width_medial_map)
plt.imsave(os.path.join(OUT_DIR, 'width_medial.png'),
           width_medial_map, cmap=CMAP)
print("[2] full-field medial+EDT saved")


# ──────────────────────────────────────────────────────────
# 4. PRUNE SKELETON WITH DSE (GPU)
# ──────────────────────────────────────────────────────────
print("[3] medial_axis + DSE pruning...")
sk_gpu, dist_gpu = medial_axis(cp.array(bw), return_distance=True)
sk, dist = sk_gpu.get(), dist_gpu.get()
pruned = skel_pruning_DSE(sk, dist,
                          min_area_px=MIN_AREA_PX,
                          return_graph=False)
ys, xs = np.nonzero(pruned)
skeleton_pts = list(zip(ys, xs))
print(f"    pruned skeleton pts: {len(skeleton_pts)}")


# ──────────────────────────────────────────────────────────
# 5. EXTRACT LONGEST CENTERLINE PATH
# ──────────────────────────────────────────────────────────
nbrs = defaultdict(list)
dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
sset = set(skeleton_pts)
for y,x in skeleton_pts:
    for dy,dx in dirs:
        if (y+dy, x+dx) in sset:
            nbrs[(y,x)].append((y+dy, x+dx))

# endpoints = degree‐1 nodes
ends = [pt for pt,n in nbrs.items() if len(n)==1]
if len(ends) < 2:
    print("⚠️ too few endpoints, using entire skeleton pts as path")
    path = skeleton_pts
else:
    # BFS from one endpoint
    src = ends[0]
    pred = {src: None}
    q = deque([src])
    while q:
        u = q.popleft()
        for v in nbrs[u]:
            if v not in pred:
                pred[v] = u
                q.append(v)
    # keep only reachable endpoints
    reachable = [e for e in ends if e in pred]
    if len(reachable) < 2:
        print("⚠️ fewer than 2 reachable endpoints → full skeleton")
        path = skeleton_pts
    else:
        # farthest endpoint by squared‐distance
        far = max(reachable,
                  key=lambda e: (e[0]-src[0])**2 + (e[1]-src[1])**2)
        # backtrack to build centerline
        path = []
        cur = far
        while cur is not None:
            path.append(cur)
            cur = pred[cur]
        path.reverse()
print(f"[4] main centerline path: {len(path)} pts")


# ──────────────────────────────────────────────────────────
# 6. PROFILE-NORMAL WIDTH MEASUREMENT
# ──────────────────────────────────────────────────────────
def compute_normals(path, all_pts, window=PCA_WINDOW):
    arr = np.array(all_pts)
    normals = []
    for (y,x) in path:
        # select local points
        dy = np.abs(arr[:,0]-y) <= window//2
        dx = np.abs(arr[:,1]-x) <= window//2
        sel = arr[dy & dx]
        if len(sel) < 3:
            normals.append(np.array([0,1], float))
        else:
            pca = PCA(2).fit(sel)
            tang = pca.components_[0]
            n = np.array([-tang[1], tang[0]])
            n /= np.linalg.norm(n)
            normals.append(n)
    return normals

print("[5] shooting profile-normals along centerline…")
normals = compute_normals(path, skeleton_pts)
width_prof = []
ends_prof  = []
for (y,x), n in zip(path, normals):
    dists = []
    pts   = []
    for sgn in (+1, -1):
        for i in range(1, max(bw.shape)//2):
            yy = int(round(y + sgn*n[0]*i))
            xx = int(round(x + sgn*n[1]*i))
            if not (0<=yy<bw.shape[0] and 0<=xx<bw.shape[1]) or not bw[yy,xx]:
                dists.append(i-1)
                pts.append((yy,xx))
                break
    if len(dists)==2:
        width_prof.append(sum(dists))
        ends_prof.append(tuple(pts))
    else:
        width_prof.append(np.nan)
        ends_prof.append(((y,x),(y,x)))
width_prof = np.array(width_prof)
print(f"    profile widths: {np.count_nonzero(~np.isnan(width_prof))}/{len(width_prof)} valid")


# ──────────────────────────────────────────────────────────
# 7. DISTANCE ALONG CENTERLINE
# ──────────────────────────────────────────────────────────
dists = [0.0]
for (y0,x0),(y1,x1) in zip(path[:-1], path[1:]):
    d = np.hypot(x1-x0, y1-y0)*PIXEL_SIZE
    dists.append(dists[-1] + d)
dists = np.array(dists)


# ──────────────────────────────────────────────────────────
# 8. LEFM √r FIT
# ──────────────────────────────────────────────────────────
def LEFM(r, C): return C * np.sqrt(r)
mask = (dists>0) & np.isfinite(width_prof) & (width_prof>0)
if mask.sum() > 5:
    C_fit, _ = curve_fit(LEFM, dists[mask], width_prof[mask])
    C = C_fit[0]
else:
    C = np.nan
width_fm = LEFM(dists, C)
print(f"[6] LEFM fit: C = {C:.3g}")


# ──────────────────────────────────────────────────────────
# 9. CSV EXPORT
# ──────────────────────────────────────────────────────────
width_med_skel = np.array([width_medial_map[y,x] for y,x in path])
df = pd.DataFrame({
    'y'            : [p[0] for p in path],
    'x'            : [p[1] for p in path],
    'dist_mm'      : dists,
    'width_medial' : width_med_skel,
    'width_profile': width_prof,
    'width_fm'     : width_fm
})
df.to_csv(os.path.join(OUT_DIR, 'crack_widths_hybrid.csv'), index=False)
print("[7] CSV saved")


# ──────────────────────────────────────────────────────────
# 10. HYBRID WIDTH & PLOTTING
# ──────────────────────────────────────────────────────────
width_hybrid = np.nanmax(
    np.vstack([width_med_skel, width_prof, width_fm]), axis=0
)

def plot_on_centerline(vals, cmap, title, fname):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(bw, cmap='gray', alpha=0.5)
    norm = plt.Normalize(np.nanmin(vals), np.nanmax(vals))
    for (y,x), v in zip(path, vals):
        if np.isfinite(v) and v>0:
            ax.scatter(x, y, c=[cmap(norm(v))], s=4)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Width (px)')
    ax.set_title(title)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=300)
    plt.close(fig)

# all four with the same colormap
plot_on_centerline(width_med_skel, CMAP, "Medial-Axis EDT Width",    "skeleton_medial.png")
plot_on_centerline(width_prof,    CMAP, "Profile-Normal Width",     "skeleton_profile.png")
plot_on_centerline(width_fm,      CMAP, "LEFM √r Predicted Width",   "skeleton_fm.png")
plot_on_centerline(width_hybrid,  CMAP, "Hybrid Width (max of all)", "skeleton_hybrid.png")

# lastly: scatter + LEFM overlay
plt.figure(figsize=(8,4))
plt.scatter(dists, width_prof, s=4, alpha=0.5, label='Profile-Measured')
plt.plot(dists, width_fm, 'r-', lw=2, label=f'LEFM fit: C={C:.2f}')
plt.xlabel("Distance from tip (mm)")
plt.ylabel("Width (px)")
plt.title("LEFM √r Crack Opening Fit")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "lefm_fit.png"), dpi=300)
plt.close()

print("[8] All plots saved under", OUT_DIR)
