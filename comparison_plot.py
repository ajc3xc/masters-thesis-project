import numpy as np
import matplotlib.pyplot as plt
from time import time

# GPU-accelerated morphology & EDT
import cupy as cp
from cucim.skimage.morphology import medial_axis
from scipy.ndimage         import distance_transform_edt

# Classic image ops
from skimage.filters      import sobel_h, sobel_v
from skimage.feature      import structure_tensor, structure_tensor_eigenvalues
from skimage.measure      import profile_line
from skimage.morphology   import skeletonize
from sklearn.decomposition import PCA

# DSE pruning
from dsepruning.dsepruning import skel_pruning_DSE

# --- 0) PARAMETERS & INPUT ---
img_path    = '/mnt/e/camerer_ml/datasets/METU_Concrete_Crack_Seg/BW_visible/236.png'
threshold   = 0.5
min_area_px = 1000

# --- 1) LOAD & BINARIZE ---
from skimage.io import imread
img = imread(img_path, as_gray=True)
bw  = (img > threshold)

# --- 2) SKELETONIZE & DISTANCE (ONCE) ---
bw_bool    = bw.astype(bool)
sk_gpu, dist_gpu = medial_axis(cp.array(bw_bool), return_distance=True)
sk, dist   = sk_gpu.get(), dist_gpu.get()

# --- 3) DSE PRUNE (ONCE) ---
pruned = skel_pruning_DSE(sk, dist, min_area_px=min_area_px, return_graph=False)
ys, xs = np.nonzero(pruned)

# --- 4) WIDTH ESTIMATORS (use `dist` for dynamic sampling) ---

def width_medial(bw, pruned_skel):
    # baseline
    w = np.zeros_like(bw, float)
    w[pruned_skel] = dist[pruned_skel] * 2
    return w

def width_profile(bw, pruned_skel):
    """
    Subpixel crack width via profile_line + structure-tensor normals,
    sampling dynamically out to each point’s medial radius.
    """
    EDGE_THRESHOLD = 0.2   # <<--- Lower threshold is more robust!
    im_f = bw.astype(np.float32)

    # 1) precompute structure tensor & principal direction
    Axx, Axy, Ayy = structure_tensor(im_f, sigma=1)
    lam1, _       = structure_tensor_eigenvalues([Axx, Axy, Ayy])
    vx =  Axy
    vy =  lam1 - Axx
    nrm = np.hypot(vx, vy) + 1e-8
    vx /= nrm; vy /= nrm
    # tangent = (vx, vy); normal = rotate 90° → (-vy, vx)
    nx, ny = -vy, vx

    w = np.zeros_like(bw, float)
    ys, xs = np.nonzero(pruned_skel)
    for y, x in zip(ys, xs):
        L = int(np.ceil(dist[y, x] + 1))
        r1, c1 = y + ny[y, x]*L, x + nx[y, x]*L
        r2, c2 = y - ny[y, x]*L, x - nx[y, x]*L
        prof = profile_line(im_f, (r1, c1), (r2, c2),
                            order=1, mode='constant', cval=0)
        edges = np.where(np.diff((prof > EDGE_THRESHOLD).astype(int)) != 0)[0]
        if len(edges) >= 2:
            w[y, x] = edges[-1] - edges[0]
        # Debug visualization (optional)
        if np.random.rand() < 0.005:
            plt.figure()
            plt.plot(prof)
            plt.title(f"Profile at ({y},{x}), width={w[y,x]:.2f}")
            plt.xlabel("Sample along normal")
            plt.ylabel("Intensity")
            plt.show()
    return w

# stubs—fill in the SAME pattern of dynamic L and profile_line for each
def width_sobel(bw, pruned_skel):
    w = np.zeros_like(bw, float)
    # 1) compute grad_x, grad_y
    # 2) for each (y,x) in ys,xs:
    #      L = ceil(dist[y,x]+1)
    #      t = average gradient in L×L patch
    #      n = [-t[1], t[0]]
    #      sample profile and detect edges
    return w

def width_pca(bw, pruned_skel):
    w = np.zeros_like(bw, float)
    # 1) for each (y,x) in ys,xs:
    #      L = ceil(dist[y,x]+1)
    #      extract L×L patch, find px coords of patch>0
    #      PCA to find tangent, normal = orthogonal
    #      sample profile and detect edges 
    return w

# list of methods to compare
methods = [
    ("Medial-EDT",   width_medial),
    ("Profile-Line", width_profile),
    # uncomment once you fill these stubs:
    # ("Sobel",        width_sobel),
    # ("PCA",          width_pca),
]

# --- 5) VISUALIZE A) Baseline ---
plt.figure(figsize=(6,6))
plt.imshow(bw, cmap='gray')
plt.scatter(xs, ys, c=dist[pruned]*2, cmap='viridis', s=15)
plt.title("Baseline: Medial-Axis ×2 (proxy GT)")
plt.axis('off')
plt.colorbar(label='Width (px)')
plt.show()

# --- 6) VISUALIZE B) Each Method vs. Baseline ---
baseline_w = dist * 2

for name, fn in methods:
    wmap   = fn(bw, pruned)
    errmap = wmap - baseline_w

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))
    ax1.imshow(bw, cmap='gray')
    im1 = ax1.scatter(xs, ys, c=wmap[pruned], cmap='viridis', s=15)
    ax1.set_title(f"{name} Widths"); ax1.axis('off')
    fig.colorbar(im1, ax=ax1, label='Width (px)')

    ax2.imshow(bw, cmap='gray')
    im2 = ax2.scatter(xs, ys, c=errmap[pruned], cmap='coolwarm',
                     vmin=-baseline_w.max(), vmax=baseline_w.max(), s=15)
    ax2.set_title(f"{name} Error vs Medial"); ax2.axis('off')
    fig.colorbar(im2, ax=ax2, label='Error (px)')

    plt.tight_layout()
    plt.show()
