#!/usr/bin/env python3
import os
import sys
import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

# GPU/array backends
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Image I/O & filters
from skimage.io import imread
from skimage.filters import sobel, frangi
import cupy as cp
from cucim.skimage.morphology import medial_axis
from scipy.ndimage import distance_transform_edt

# Crack‐width methods
from dsepruning.dsepruning import skel_pruning_DSE
from subpixel_edges import subpixel_edges

# PCA for Sobel+PCA
from sklearn.decomposition import PCA

# PiDiNet (optional)
import torch
'''from models.pidinet import pidinet_small
from models.config import config_model'''
from skimage.measure import profile_line

# ---------------- User Settings ----------------
IMG_PATH       = '/mnt/e/camerer_ml/datasets/METU_Concrete_Crack_Seg/BW/236.JPG'
THRESHOLD      = 0.25
MIN_AREA_PX    = 1000
PATCH_SIZE     = 100  # for Sobel+PCA
PCA_THRESH     = 0.1
PIDINET_WEIGHTS = '/path/to/table5_pidinet-small.pth'
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ------------------------------------------------

def width_medial_dse(bw):
    """Medial‐axis + EDT + DSE pruning."""
    # choose GPU or CPU medial axis
    if GPU_AVAILABLE:
        print("Using GPU for medial axis...")
        sk_gpu, dist_gpu = medial_axis(cp.array(bw), return_distance=True)
        sk, dist = sk_gpu.get(), dist_gpu.get()
    else:
        sk, dist = medial_axis(bw, return_distance=True)
    pruned = skel_pruning_DSE(sk, dist, min_area_px=MIN_AREA_PX, return_graph=False)
    w = np.zeros_like(bw, float)
    w[pruned] = dist[pruned] * 2
    return w, pruned

def width_subpixel(bw, pruned):
    """Subpixel‐edges + normal‐projection width."""
    edges = subpixel_edges(bw.astype(float), threshold=0.05, iters=1, order=1)
    coords_edge = np.column_stack([edges.y, edges.x])
    # compute structure‐tensor normals
    from skimage.feature import structure_tensor, structure_tensor_eigenvalues
    Axx, Axy, Ayy = structure_tensor(bw.astype(float), sigma=1)
    lam1, _       = structure_tensor_eigenvalues([Axx, Axy, Ayy])
    vx, vy        = Axy, lam1 - Axx
    nrm           = np.hypot(vx, vy) + 1e-8
    vx, vy        = vx/nrm, vy/nrm
    nx, ny        = -vy, vx

    w = np.zeros_like(bw, float)
    for (y, x) in zip(*np.nonzero(pruned)):
        n_vec = np.array([ny[y,x], nx[y,x]])
        disp = coords_edge - np.array([y, x])
        proj = disp @ n_vec
        perp = np.abs(disp @ np.array([vx[y,x], vy[y,x]]))
        valid = perp < 6
        if np.sum(valid) >= 2:
            vals = proj[valid]
            wval = vals.max() - vals.min()
            if wval > 0.5:
                w[y,x] = wval
    return w

from skimage.filters import sobel
import numpy as np

from skimage.filters import gaussian, sobel
import numpy as np

def sobel_direction_and_edges(gray, skel, bw, dist_map, edge_thresh=0.5, rmax=1000, sigma=2.0):
    """
    For each skeleton point:
        - Compute smoothed local patch
        - Estimate direction by averaging Sobel gradients in patch
        - Compute normal vector
        - Trace along normal in both directions until leaving the crack (bw < edge_thresh)
    Returns:
        width_map:      width at each skeleton point (float)
        left_edges:     array (N, 2) of (y, x) coords
        right_edges:    array (N, 2) of (y, x) coords
        skel_points:    array (N, 2) of (y, x) coords
    """
    width_map = np.zeros_like(skel, float)
    left_edges = []
    right_edges = []
    skel_points = []
    H, W = gray.shape
    Y, X = np.nonzero(skel)
    for y, x in zip(Y, X):
        r = min(max(3, int(dist_map[y, x]) + 2), rmax)
        y0, y1 = max(0, y - r), min(H, y + r + 1)
        x0, x1 = max(0, x - r), min(W, x + r + 1)
        patch = gray[y0:y1, x0:x1]
        patch_smooth = gaussian(patch, sigma=sigma, preserve_range=True)
        gx_patch = sobel(patch_smooth, axis=1)
        gy_patch = sobel(patch_smooth, axis=0)
        gx = np.mean(gx_patch)
        gy = np.mean(gy_patch)
        v = np.array([gy, gx])
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-4:
            continue
        v /= v_norm
        n = np.array([-v[1], v[0]])

        left = None
        right = None
        # Trace in +normal direction
        for i in range(1, rmax):
            p = np.round([y, x] + n * i).astype(int)
            if not (0 <= p[0] < H and 0 <= p[1] < W):
                break
            if bw[p[0], p[1]] < edge_thresh:
                right = p
                break
        # Trace in -normal direction
        for i in range(1, rmax):
            p = np.round([y, x] - n * i).astype(int)
            if not (0 <= p[0] < H and 0 <= p[1] < W):
                break
            if bw[p[0], p[1]] < edge_thresh:
                left = p
                break

        if right is not None and left is not None:
            width = np.linalg.norm(right - left)
            width_map[y, x] = width
            left_edges.append(left)
            right_edges.append(right)
            skel_points.append([y, x])
    left_edges = np.array(left_edges)
    right_edges = np.array(right_edges)
    skel_points = np.array(skel_points)
    return width_map, left_edges, right_edges, skel_points

from sklearn.decomposition import PCA

def pca_local_width(bw, skel, dist_map, patch_scale=1.5, edge_thresh=0.5, min_points=4):
    width_map = np.zeros_like(skel, float)
    Y, X = np.nonzero(skel)
    for y, x in zip(Y, X):
        r = int(max(4, dist_map[y, x] * patch_scale))
        y0, y1 = max(0, y - r), min(bw.shape[0], y + r + 1)
        x0, x1 = max(0, x - r), min(bw.shape[1], x + r + 1)
        patch = bw[y0:y1, x0:x1]
        # Get coordinates of edge points (bw==1 and neighbor==0)
        edges = np.argwhere((patch > edge_thresh) & (
            (np.pad(patch, 1)[1:-1, :-2] == 0) |
            (np.pad(patch, 1)[1:-1, 2:] == 0) |
            (np.pad(patch, 1)[:-2, 1:-1] == 0) |
            (np.pad(patch, 1)[2:, 1:-1] == 0)
        ))
        if len(edges) < min_points:
            continue
        edges += [y0, x0]
        pca = PCA(n_components=2)
        pca.fit(edges)
        minor_axis = pca.components_[1]
        proj = (edges - np.array([y, x])) @ minor_axis
        width_map[y, x] = proj.max() - proj.min()
    return width_map

'''def pca_local_width(bw, skel, dist_map, patch_scale=1.5, edge_thresh=0.5, min_points=4):
    """For each skeleton point, find edge points in a patch (using bw mask),
    run PCA, take width as minor axis length."""
    width_map = np.zeros_like(skel, float)
    Y, X = np.nonzero(skel)
    for y, x in zip(Y, X):
        r = int(max(4, dist_map[y, x] * patch_scale))
        y0, y1 = max(0, y - r), min(bw.shape[0], y + r + 1)
        x0, x1 = max(0, x - r), min(bw.shape[1], x + r + 1)
        patch = bw[y0:y1, x0:x1]
        # Get coordinates of edge points (bw==1 and 4-neighbor to 0)
        edges = np.argwhere((patch > edge_thresh) & (
            (np.pad(patch, 1)[1:-1, :-2] == 0) |
            (np.pad(patch, 1)[1:-1, 2:] == 0) |
            (np.pad(patch, 1)[:-2, 1:-1] == 0) |
            (np.pad(patch, 1)[2:, 1:-1] == 0)
        ))
        if len(edges) < min_points:
            continue
        # shift to image coords
        edges += [y0, x0]
        # PCA: principal axes
        pca = PCA(n_components=2)
        pca.fit(edges)
        # Take the length of the minor axis (project points to minor axis)
        minor_axis = pca.components_[1]
        proj = (edges - np.array([y, x])) @ minor_axis
        width_map[y, x] = proj.max() - proj.min()
    return width_map'''

def width_frangi_on_skel(gray, skel, sigmas=np.arange(2, 30, 2)):
    """Estimate width along a given skeleton using Frangi filter."""
    vesselness = np.zeros((len(sigmas),) + gray.shape)
    # Compute vesselness at each sigma
    for i, sigma in enumerate(sigmas):
        vesselness[i] = frangi(gray, sigmas=[sigma])
    # For each pixel, find sigma with max vesselness
    max_response = vesselness.max(axis=0)
    best_sigma = sigmas[np.argmax(vesselness, axis=0)]
    # At each skeleton point, width = 2 × sigma at max vesselness
    w = np.zeros_like(gray, float)
    yx = np.nonzero(skel)
    w[yx] = best_sigma[yx] * 2
    return w, skel

def width_pidinet(gray, pruned):
    """PiDiNet + profile-line width."""
    # load model
    pdcs = config_model('carv4')
    model = pidinet_small(pdcs).to(DEVICE).eval()
    ckpt = torch.load(PIDINET_WEIGHTS, map_location=DEVICE)
    state = ckpt.get('state_dict', ckpt)
    # strip module prefixes if any
    from collections import OrderedDict
    new_st = OrderedDict((k.replace('module.',''),v) for k,v in state.items())
    model.load_state_dict(new_st, strict=False)

    img3 = np.stack([gray,gray,gray], axis=0)[None].astype(np.float32)
    tensor = torch.from_numpy(img3).to(DEVICE)
    with torch.no_grad():
        edges = model(tensor)[-1].cpu().numpy()[0,0]
    edges = (edges - edges.min())/(edges.ptp()+1e-8)

    # use existing DSE‐pruned skeleton from binary
    from skimage.feature import structure_tensor, structure_tensor_eigenvalues
    Axx, Axy, Ayy = structure_tensor(bw.astype(float), sigma=1)
    lam1, _       = structure_tensor_eigenvalues([Axx, Axy, Ayy])
    vx, vy        = Axy, lam1 - Axx
    nrm           = np.hypot(vx, vy) + 1e-8
    vx, vy        = vx/nrm, vy/nrm
    nx, ny        = -vy, vx

    w = np.zeros_like(bw, float)
    for (y,x) in zip(*np.nonzero(pruned)):
        L = int(np.ceil(dist[y,x] + 1))
        p1 = (y + ny[y,x]*L, x + nx[y,x]*L)
        p2 = (y - ny[y,x]*L, x - nx[y,x]*L)
        prof = profile_line(edges, p1, p2, order=1, mode='constant', cval=0)
        eds = np.where(np.diff((prof > 0.05).astype(int)) != 0)[0]
        if len(eds) >= 2:
            w[y,x] = eds[-1] - eds[0]
    return w

if __name__ == "__main__":
    # 1) Load
    gray = imread(IMG_PATH, as_gray=True)
    bw   = (gray > THRESHOLD)
    dist_map = distance_transform_edt(bw)

    # Prepare storage
    results = {}
    timings = {}

    # 2) Medial + DSE
    t0 = perf_counter()
    w_med, pruned = width_medial_dse(bw)
    timings['Medial+DSE'] = perf_counter() - t0
    results['Medial+DSE'] = (w_med, pruned)

    # 3) Subpixel‐edges
    '''t0 = perf_counter()
    w_sub = width_subpixel(bw, pruned)
    timings['Subpixel'] = perf_counter() - t0
    results['Subpixel'] = (w_sub, pruned)'''

    # Sobel-based (2025)
    t0 = perf_counter()
    w_sobel, left_edges, right_edges, skel_points = sobel_direction_and_edges(
        gray, pruned, bw, dist_map, edge_thresh=0.5, rmax=1000, sigma=2.0)
    timings['Sobel-Dir'] = perf_counter() - t0
    results['Sobel-Dir'] = (w_sobel, pruned)

    # PCA-based (2023)
    t0 = perf_counter()
    w_pca = pca_local_width(bw, pruned, dist_map)
    timings['PCA-Local'] = perf_counter() - t0
    results['PCA-Local'] = (w_pca, pruned)
    
    # 5) Frangi+Skeleton
    t0 = perf_counter()
    w_fr, sk_fr = width_frangi_on_skel(gray, pruned)
    timings['Frangi+Skel'] = perf_counter() - t0
    results['Frangi+Skel'] = (w_fr, sk_fr)

    # 6) PiDiNet (if weights present)
    '''try:
        t0 = perf_counter()
        w_pid = width_pidinet(gray, pruned)
        timings['PiDiNet'] = perf_counter() - t0
        results['PiDiNet'] = (w_pid, pruned)
    except Exception as e:
        print("PiDiNet failed:", e)'''

    # 7) Display results
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    for ax, (name, (wmap, skel)) in zip(axes, results.items()):
        ys, xs = np.nonzero(skel)
        ax.imshow(bw, cmap='gray')
        sc = ax.scatter(xs, ys, c=wmap[skel], cmap='viridis', s=10)
        ax.set_title(f"{name}\n{1/timings[name]:.1f} FPS")
        ax.axis('off')
        fig.colorbar(sc, ax=ax, label='Width (px)')
    plt.tight_layout()
    plt.show()

    # 8) Print timings
    print("\n=== Timings (per frame) ===")
    for name, t in timings.items():
        print(f"{name:15s}: {t*1000:7.1f} ms ({1/t:.1f} FPS)")
