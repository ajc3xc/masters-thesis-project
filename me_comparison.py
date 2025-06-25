import numpy as np
import matplotlib.pyplot as plt
from time import time

# GPU-accelerated morphology & EDT
from cucim.skimage.morphology import medial_axis    # cuCIM medial axis on GPU :contentReference[oaicite:5]{index=5}
from scipy.ndimage import distance_transform_edt

# Classic image ops
from skimage.filters import sobel_h, sobel_v
#from skimage.morphology import sketonize
from skimage.feature import structure_tensor, structure_tensor_eigenvalues
from skimage.measure  import profile_line
import numpy as np
from skimage.filters import gabor_kernel               # steerable/Gabor filters :contentReference[oaicite:7]{index=7}

# PCA
from sklearn.decomposition import PCA                  # PCA for orientation :contentReference[oaicite:8]{index=8}

# PST (Phase Stretch Transform)
from phycv import PST                                  # PhyCV’s PST implementation :contentReference[oaicite:9]{index=9}

# Subpixel-Edges
from subpixel_edges import subpixel_edges            # pure-Python subpixel library :contentReference[oaicite:10]{index=10}

# PiDiNet (lightweight CNN edge detector)
import torch
#from pidinet import PiDiNet                             # PyTorch implementation :contentReference[oaicite:11]{index=11}
import cupy as cp

# --- 1. Synthetic Crack Generator ---
def make_crack(shape=(256,256), width=5, noise=0):
    img = np.zeros(shape, dtype=np.float32)
    rr = np.arange(20, shape[0]-20)
    for dx in range(-width//2, width//2+1):
        img[rr, rr + dx] = 1
    # optional gaussian blur to simulate fuzzy edges
    if noise>0:
        from scipy.ndimage import gaussian_filter
        img = gaussian_filter(img, sigma=noise)
    return (img > 0.5).astype(np.uint8)

# --- 2. Measurement Methods ---

def width_pca(bw, sk, half_length=5):
    
    w = np.zeros_like(bw, float)
    ys, xs = np.nonzero(sk)
    for y, x in zip(ys, xs):
        y0, y1 = max(0, y-half_length), min(bw.shape[0], y+half_length+1)
        x0, x1 = max(0, x-half_length), min(bw.shape[1], x+half_length+1)
        patch = bw[y0:y1, x0:x1]
        edge_pts = np.column_stack(np.nonzero(patch))
        if len(edge_pts) > 2:
            pca = PCA(n_components=2).fit(edge_pts)
            direction = pca.components_[0]  # main axis
            normal = np.array([-direction[1], direction[0]])
            center = np.array([y - y0, x - x0])
            # Trace along normal
            vals = []
            for d in np.linspace(-half_length, half_length, 2*half_length+1):
                pt = center + normal * d
                yy, xx = np.round(pt).astype(int)
                if 0 <= yy < patch.shape[0] and 0 <= xx < patch.shape[1]:
                    vals.append(patch[yy, xx])
                else:
                    vals.append(0)
            ups = np.where(np.array(vals) > 0.5)[0]
            if len(ups) > 1:
                w[y, x] = ups[-1] - ups[0]
    return w

def width_sobel(bw, sk, half_length=5):
    
    w = np.zeros_like(bw, float)
    grad_x = sobel_h(bw.astype(float))
    grad_y = sobel_v(bw.astype(float))
    ys, xs = np.nonzero(sk)
    for y, x in zip(ys, xs):
        # Local patch for direction
        y0, y1 = max(0, y-half_length), min(bw.shape[0], y+half_length+1)
        x0, x1 = max(0, x-half_length), min(bw.shape[1], x+half_length+1)
        gx_patch = grad_x[y0:y1, x0:x1]
        gy_patch = grad_y[y0:y1, x0:x1]
        gx_mean = np.mean(gx_patch)
        gy_mean = np.mean(gy_patch)
        tangent = np.array([gy_mean, gx_mean])
        if np.linalg.norm(tangent) < 1e-5:
            continue
        tangent /= np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]])
        vals = []
        for d in np.linspace(-half_length, half_length, 2*half_length+1):
            pt = np.array([y, x]) + normal * d
            yy, xx = np.round(pt).astype(int)
            if 0 <= yy < bw.shape[0] and 0 <= xx < bw.shape[1]:
                vals.append(bw[yy, xx])
            else:
                vals.append(0)
        ups = np.where(np.array(vals) > 0.5)[0]
        if len(ups) > 1:
            w[y, x] = ups[-1] - ups[0]
    return w

def width_profile(bw, sk, half_length=5, sigma=1, thresh=0.5):
    """
    Subpixel crack width via profile_line + structure-tensor normals.
    
    bw           : 2D binary mask (uint8 or bool)
    half_length  : how far along the normal to sample (in pixels)
    sigma        : smoothing for structure tensor
    thresh       : cutoff on profile to detect edges
    """
    # 1) sketon
    

    # 2) Precompute structure tensor on float image
    im_f = bw.astype(np.float32)
    Axx, Axy, Ayy = structure_tensor(im_f, sigma=sigma)

    # 3) Eigenvalues (shape (2, H, W)): lam1 ≥ lam2
    lam1, lam2 = structure_tensor_eigenvalues([Axx, Axy, Ayy])

    # 4) Principal eigenvector (tangent) for lam1:
    #    Solve [Axx-λ, Axy; Axy, Ayy-λ] · v = 0
    #    One valid eigenvector is [Axy, λ-Axx]
    vx = Axy
    vy = lam1 - Axx
    norm = np.hypot(vx, vy) + 1e-8
    vx /= norm
    vy /= norm

    # 5) Rotate 90° to get normal: n = (-vy, vx)
    nx = -vy
    ny = vx

    # 6) Sample profiles and compute width
    w = np.zeros_like(bw, dtype=float)
    ys, xs = np.nonzero(sk)
    for y, x in zip(ys, xs):
        dx, dy = nx[y, x], ny[y, x]
        p1 = (y + dy * half_length, x + dx * half_length)
        p2 = (y - dy * half_length, x - dx * half_length)
        profile = profile_line(im_f, p1, p2,
                               order=1, mode='constant', cval=0)
        # find transitions 0→1 and 1→0
        edges = np.where(np.diff((profile > thresh).astype(int)) != 0)[0]
        if len(edges) >= 2:
            w[y, x] = edges[-1] - edges[0]

    return w

def width_steerable(bw, sk):
    
    w = np.zeros_like(bw, float)
    # precompute bank of Gabor kernels at different orientations
    thetas = np.linspace(0, np.pi, 8, endpoint=False)
    kernels = [(np.real(gabor_kernel(frequency=0.2, theta=t)),
                np.imag(gabor_kernel(frequency=0.2, theta=t))) for t in thetas]
    responses = []
    for real,imag in kernels:
        responses.append(np.sqrt(
            np.square(np.real(np.fft.ifft2(np.fft.fft2(bw)*np.fft.fft2(real, bw.shape)))) +
            np.square(np.real(np.fft.ifft2(np.fft.fft2(bw)*np.fft.fft2(imag, bw.shape))))))
    # for each sketon point, pick orientation with max response & trace along normal
    # (left as exercise)
    return w

def width_pst(bw, sk):
    
    w = np.zeros_like(bw, float)
    from phycv import PST
    pst = PST()
    print(dir(pst))
    edges = pst.run(bw.astype(np.float32))  # Not .process!
    ys, xs = np.nonzero(sk)
    for y,x in zip(ys,xs):
        # trace normal via PST orientation, sample edges
        w[y,x] = np.nan                      # placeholder
    return w

def width_subpixel(bw, sk, threshold=0.2, iters=1, order=1, half_length=5):
    edges = subpixel_edges(bw.astype(float), threshold=threshold, iters=iters, order=order)
    
    w = np.zeros_like(bw, float)
    ys, xs = np.nonzero(sk)
    for sy, sx in zip(ys, xs):
        # (example, you may want to do nearest two edge points, etc)
        dists = np.hypot(edges.y - sy, edges.x - sx)
        if np.sum(dists < half_length) >= 2:
            idx = np.argsort(dists)[:2]
            width = np.hypot(edges.x[idx[0]] - edges.x[idx[1]], edges.y[idx[0]] - edges.y[idx[1]])
            w[sy, sx] = width
    return w

'''def width_pidinet(bw, model=None, device='cuda'):
    if model is None:
        model = PiDiNet().to(device).eval()
    with torch.no_grad():
        inp = torch.from_numpy(bw.astype(float))[None,None,...].to(device)
        edge_map = model(inp).cpu().numpy()[0,0]
    
    w = np.zeros_like(bw, float)
    # trace normals on edge_map (left as exercise)
    return w'''

def width_medial(bw, pruned_skel):
    w = np.zeros_like(bw, dtype=float)
    w[pruned_skel > 0] = dist[pruned_skel > 0] * 2
    return w

# --- 3. Benchmarking Harness ---
methods = [
    ("Medial+EDT", width_medial),
    ("Sobel", width_sobel),
    ("PCA", width_pca),
    ("Profile-Line", width_profile),
    ("Steerable", width_steerable),
    #("PST", width_pst),
    ("SubPixLib", width_subpixel),
]

# Synthetic test
img_path = '/mnt/e/camerer_ml/datasets/METU_Concrete_Crack_Seg/BW_visible/236.png'  # <-- Change this to your actual image file

# ----- Load and binarize image -----
from skimage.io import imread
img = imread(img_path, mode='L')  # Load as grayscale
bw = (img > 0.5)
# 1. Skeletonize ONCE
skel, distance = medial_axis(cp.array(bw), return_distance=True)
sk, dist = skel.get(), distance.get()

# 2. Prune once
from dsepruning.dsepruning import skel_pruning_DSE
pruned_skel = skel_pruning_DSE(sk, dist, min_area_px=1000, return_graph=False)

results = {}
for name, fn in methods:
    start = time()
    w = fn(bw, pruned_skel)
    elapsed = (time() - start)*1000
    # compute mean error vs. known width=8
    mean_err = np.nanmean(np.abs(w[pruned_skel>0] - 8))
    results[name] = (elapsed, mean_err)

# --- 4. Results Table & Plot ---
print("Method          Time (ms)   Mean Error (px)")
for name,(t,err) in results.items():
    print(f"{name:12s}   {t:8.2f}       {err:6.2f}")

def to_numpy(x):
    # Converts a CuPy array to NumPy, or returns unchanged if already NumPy
    try:
        import cupy
        if isinstance(x, cupy.ndarray):
            return x.get()
    except ImportError:
        pass
    return np.array(x)

# Simple bar chart
names = list(results.keys())
times = [to_numpy(results[n][0]) for n in names]
errs  = [to_numpy(results[n][1]) for n in names]

fig, ax = plt.subplots(1,2,figsize=(12,4))
ax[0].bar(names, times)
ax[0].set_title("Runtime (ms)")
ax[0].set_xticks(np.arange(len(names)))
ax[0].set_xticklabels(names, rotation=45)

ax[1].bar(names, errs)
ax[1].set_title("Mean Error (px)")
ax[1].set_xticks(np.arange(len(names)))
ax[1].set_xticklabels(names, rotation=45)

plt.tight_layout()
plt.show()
