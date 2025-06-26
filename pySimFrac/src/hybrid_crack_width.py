#!/usr/bin/env python3
"""
hybrid_crack_width.py

Comprehensive crack-width measurement:
  1) Full-field medial-axis + EDT widths
  2) Pruned skeleton centerline (DSE)
  3) Profile-normal widths along longest path (robust PCA window)
  4) LEFM fit w = C * sqrt(r)
  5) Hybrid width = max(profile, LEFM)
  6) Plots and CSV outputs.

Requires:
  scikit-image, dsepruning, cupy+cucim (for GPU medial_axis), scipy, sklearn, matplotlib, pandas
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.io import imread
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from collections import defaultdict, deque

# GPU medial + DSE
import cupy as cp
from cucim.skimage.morphology import medial_axis
from dsepruning.dsepruning import skel_pruning_DSE

# ────────────────────────────────
# 1. CONFIG
# ────────────────────────────────
IMG_PATH     = '/mnt/e/camerer_ml/datasets/METU_Concrete_Crack_Seg/BW_visible/236.png'
PIXEL_SIZE   = 1.0        # mm per pixel
THRESHOLD    = 0.5        # for binarization
MIN_AREA_PX  = 1000       # DSE pruning
PCA_WINDOW   = 11         # must be odd, larger => smoother normals

OUT_DIR = "physics_outputs"
os.makedirs(OUT_DIR, exist_ok=True)


# ────────────────────────────────
# 2. LOAD + BINARIZE
# ────────────────────────────────
img = imread(IMG_PATH)
if img.ndim==3:
    gray = img.mean(axis=2)
else:
    gray = img.astype(float)
if gray.max()>1.0:
    gray = gray/255.0
bw = gray>THRESHOLD
print(f"[1] mask: {bw.shape}, crack area={bw.sum()} px")
plt.imsave(os.path.join(OUT_DIR,'input_mask.png'), bw, cmap='gray')


# ────────────────────────────────
# 3. FULL-FIELD MEDIAL-AXIS + EDT
# ────────────────────────────────
# NOTE: this is your classic "width map" for ML labels.
dt = distance_transform_edt(bw)  # CPU EDT
width_map = dt*2  # in px
np.save(os.path.join(OUT_DIR,'width_medial.npy'), width_map)
plt.imsave(os.path.join(OUT_DIR,'width_medial.png'),
           width_map, cmap='magma')
print("[2] full-field medial+EDT saved")


# ────────────────────────────────
# 4. PRUNE SKELETON WITH DSE (gpu)
# ────────────────────────────────
print("[3] medial_axis + DSE pruning...")
sk_gpu, dist_gpu = medial_axis(cp.array(bw), return_distance=True)
sk, dist = sk_gpu.get(), dist_gpu.get()
pruned = skel_pruning_DSE(sk, dist, min_area_px=MIN_AREA_PX,
                          return_graph=False)
ys, xs = np.nonzero(pruned)
skeleton_pts = list(zip(ys, xs))
print(f"    skeleton pts: {len(skeleton_pts)}")
plt.figure(figsize=(6,6))
plt.imshow(bw, cmap='gray', alpha=0.6)
plt.scatter(xs, ys, s=1, c='cyan')
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,'skeleton_pruned.png'), dpi=300)
plt.close()


# ────────────────────────────────
# 5. EXTRACT LONGEST CENTERLINE PATH
# ────────────────────────────────
# Build 8-con neighbor graph
nbrs = defaultdict(list)
dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
sset = set(skeleton_pts)
for y,x in skeleton_pts:
    for dy,dx in dirs:
        if (y+dy,x+dx) in sset:
            nbrs[(y,x)].append((y+dy,x+dx))

# Find endpoints
ends = [pt for pt, ne in nbrs.items() if len(ne) == 1]
if len(ends) < 2:
    print("⚠️ too few endpoints, using full skeleton order")
    path = skeleton_pts
else:
    # BFS from one endpoint, track predecessors
    src = ends[0]
    pred = {src: None}
    q = deque([src])
    while q:
        u = q.popleft()
        for v in nbrs[u]:
            if v not in pred:
                pred[v] = u
                q.append(v)

    # only keep those endpoints we actually reached
    reachable_ends = [e for e in ends if e in pred]
    if len(reachable_ends) < 2:
        print("⚠️ fewer than 2 reachable endpoints, falling back to full skeleton")
        path = skeleton_pts
    else:
        # pick the farthest among those
        far = max(
            reachable_ends,
            key=lambda e: (e[0]-src[0])**2 + (e[1]-src[1])**2
        )
        # backtrack
        path = []
        cur = far
        while cur is not None:
            path.append(cur)
            cur = pred[cur]
        path = path[::-1]

print(f"[4] main path: {len(path)} pts")

# ────────────────────────────────
# 6. PROFILE-NORMAL WIDTHS
# ────────────────────────────────
def compute_normals(path, skeleton_pts, window=PCA_WINDOW):
    sk_arr = np.array(skeleton_pts)
    normals = []
    for (y,x) in path:
        # gather neighbors in window
        dy = np.abs(sk_arr[:,0]-y) <= window//2
        dx = np.abs(sk_arr[:,1]-x) <= window//2
        sel = sk_arr[dy&dx]
        if len(sel)<3:
            normals.append(np.array([0,1],float))
        else:
            pca = PCA(2).fit(sel)
            tang = pca.components_[0]
            # perp:
            n = np.array([-tang[1], tang[0]])
            n /= np.linalg.norm(n)
            normals.append(n)
    return normals

print("[5] shooting profile normals...")
norms = compute_normals(path, skeleton_pts)
w_prof = []
ends_prof = []
for (y,x), n in zip(path, norms):
    dists = []
    pts   = []
    for sgn in [+1, -1]:
        for i in range(1,  max(bw.shape)//2):
            yy = int(round(y + sgn*n[0]*i))
            xx = int(round(x + sgn*n[1]*i))
            if not (0<=yy<bw.shape[0] and 0<=xx<bw.shape[1]) or not bw[yy,xx]:
                dists.append(i-1)
                pts.append((yy,xx))
                break
    if len(dists)==2:
        w_prof.append(np.sum(dists))
        ends_prof.append(tuple(pts))
    else:
        w_prof.append(np.nan)
        ends_prof.append(((y,x),(y,x)))
w_prof = np.array(w_prof)
print(f"    prof widths: {np.count_nonzero(np.isfinite(w_prof))}/{len(w_prof)} valid")


# ────────────────────────────────
# 7. BUILD DISTANCE‐ALONG‐PATH
# ────────────────────────────────
dists = [0.0]
for (y0,x0),(y1,x1) in zip(path[:-1], path[1:]):
    d = np.hypot(x1-x0, y1-y0)*PIXEL_SIZE
    dists.append(dists[-1]+d)
dists = np.array(dists)


# ────────────────────────────────
# 8. LEFM FIT
# ────────────────────────────────
def LEFM(r, C): return C * np.sqrt(r)
mask = (dists>0)&np.isfinite(w_prof)&(w_prof>0)
if mask.sum()>5:
    C_fit,_ = curve_fit(LEFM, dists[mask], w_prof[mask])
    C = C_fit[0]
else:
    C = np.nan
w_fm = LEFM(dists, C)
print(f"[6] LEFM fit C={C:.3g}")

# Save CSV
df = pd.DataFrame({
    'y':[p[0] for p in path],
    'x':[p[1] for p in path],
    'dist_mm':dists,
    'width_medial_px': [width_map[y,x] for y,x in path],
    'width_prof_px': w_prof,
    'width_fm_px':   w_fm
})
df.to_csv(os.path.join(OUT_DIR,'crack_widths_hybrid.csv'), index=False)
print("[7] CSV saved")


# ────────────────────────────────
# 9. HYBRID WIDTH
# ────────────────────────────────
w_med_skel = np.array([width_map[y,x] for y,x in path])
w_hybrid = np.nanmax(np.vstack([w_prof, w_fm, w_med_skel]), axis=0)


# ────────────────────────────────
# 10. PLOTS
# ────────────────────────────────

def plot_skeleton_colorfield(vals, cmap, title, fn):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(bw, cmap='gray', alpha=0.5)
    norm = plt.Normalize(np.nanmin(vals), np.nanmax(vals))
    for (y,x), v in zip(path, vals):
        if np.isfinite(v) and v>0:
            ax.scatter(x, y, c=[cmap(norm(v))], s=4)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Width (px)')
    ax.set_title(title); ax.axis('off')
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, fn), dpi=300)
    plt.close(fig)

# (a) medial‐axis width on skeleton
plot_skeleton_colorfield(w_med_skel, plt.cm.plasma,
    "Medial‐Axis EDT Width on Skeleton", "skeleton_medial.png")

# (b) profile‐normal width
plot_skeleton_colorfield(w_prof, plt.cm.viridis,
    "Profile-Normal Width on Skeleton", "skeleton_profile.png")

# (c) LEFM predicted
plot_skeleton_colorfield(w_fm, plt.cm.inferno,
    "LEFM √r Predicted Width on Skeleton", "skeleton_fm.png")

# (d) hybrid = max(all)
plot_skeleton_colorfield(w_hybrid, plt.cm.magma,
    "Hybrid Width on Skeleton", "skeleton_hybrid.png")

# (e) LEFM fit vs measured
plt.figure(figsize=(8,4))
plt.scatter(dists, w_prof, s=4, alpha=0.5, label='Profile-Measured')
plt.plot(dists, w_fm, 'r-', lw=2, label=f'LEFM fit: C={C:.2f}')
plt.xlabel("Distance from tip (mm)")
plt.ylabel("Width (px)")
plt.title("LEFM √r Crack Opening Fit")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"lefm_fit.png"), dpi=300)
plt.close()

print("[8] Plots saved in", OUT_DIR)
