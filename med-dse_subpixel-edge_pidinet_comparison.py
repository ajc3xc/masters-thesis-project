#!/usr/bin/env python3
import numpy as np
import cupy as cp
import torch
from time import time
import matplotlib.pyplot as plt
from skimage.io import imread
from cucim.skimage.morphology import medial_axis  # GPU medial axis :contentReference[oaicite:9]{index=9}
from scipy.ndimage import distance_transform_edt
from dsepruning.dsepruning import skel_pruning_DSE
from subpixel_edges import subpixel_edges
from skimage.measure import profile_line
import sys
sys.path.append('/mnt/c/Users/13144/Documents/Masters_Thesis/crack_width_measurement/pidinet')  # <- SET THIS to where you cloned PiDiNet

#from models.pidinet import PiDiNet
#from models.config import config_model

# ------- User settings -------
img_path    = '/mnt/e/camerer_ml/datasets/METU_Concrete_Crack_Seg/BW_visible/236.png'  # <<--- Your test image
img_path = '/mnt/e/camerer_ml/datasets/METU_Concrete_Crack_Seg/BW/236.JPG'
threshold   = 0.25
min_area_px = 1000
pidinet_weights = '/mnt/c/Users/13144/Documents/Masters_Thesis/crack_width_measurement/pidinet/trained_models/table5_pidinet-small.pth'  # <<--- SET THIS!
# 1) Load & Binarize
img = imread(img_path, as_gray=True)
bw  = (img > threshold)

# 2) Skeleton + Medial‐Axis + DSE Prune
t0 = time()
sk_gpu, dist_gpu = medial_axis(cp.array(bw, bool), return_distance=True)
sk, dist         = sk_gpu.get(), dist_gpu.get()
pruned           = skel_pruning_DSE(sk, dist, min_area_px=min_area_px, return_graph=False)
ys, xs           = np.nonzero(pruned)
w_medial         = np.zeros_like(bw, float)
w_medial[pruned] = dist[pruned] * 2
print(f"Medial+EDT+DSE pruning time: {(time()-t0)*1000:.1f} ms")

# 3) Subpixel‐Edges Width
def width_subpixel_normal(bw, pruned, edges=None, sigma=1, angle_tol=30):
    if edges is None:
        from subpixel_edges import subpixel_edges
        edges = subpixel_edges(bw.astype(float), threshold=0.05, iters=1, order=1)
    w = np.zeros_like(bw, float)
    edge_coords = np.column_stack([edges.y, edges.x])  # (N,2)
    sk_points   = np.column_stack(np.nonzero(pruned))  # (M,2)

    # Compute normals using structure tensor
    from skimage.feature import structure_tensor, structure_tensor_eigenvalues
    Axx, Axy, Ayy = structure_tensor(bw.astype(float), sigma=sigma)
    lam1, _       = structure_tensor_eigenvalues([Axx, Axy, Ayy])
    vx, vy        = Axy, lam1 - Axx
    nrm           = np.hypot(vx, vy) + 1e-8
    vx, vy        = vx/nrm, vy/nrm
    nx, ny        = -vy, vx

    for y, x in sk_points:
        n_vec = np.array([ny[y, x], nx[y, x]])
        # Vector from skeleton point to every edge point
        disp = edge_coords - np.array([y, x])
        # Project onto normal direction
        proj = disp @ n_vec
        # Also: perpendicular (tangent) projection
        perp_proj = np.abs(disp @ np.array([vx[y, x], vy[y, x]]))
        # Keep edge points within a perpendicular "distance" (say, <6px from the normal line)
        valid = (perp_proj < 6)
        if np.sum(valid) >= 2:
            proj_vals = proj[valid]
            # width = distance between furthest projections (along normal)
            width = np.max(proj_vals) - np.min(proj_vals)
            if width > 0.5:
                w[y, x] = width
    return w

t0 = time()
w_subpix = width_subpixel_normal(bw, pruned)
print(f"Subpixel-Edges time: {(time()-t0)*1000:.1f} ms")

# 4) PiDiNet + Profile‐Line Width
'''device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#pdcs   = config_model('carv4')
from argparse import Namespace
from models.pidinet import pidinet_small

args = Namespace(config='carv4', sa=False, dil=True)
model = pidinet_small(args).to(device).eval()
# load weights
ckpt   = torch.load(pidinet_weights, map_location=device)
state  = ckpt.get('state_dict', ckpt)
# strip DataParallel prefix
from collections import OrderedDict
new_st = OrderedDict((k.replace('module.',''),v) for k,v in state.items())
model.load_state_dict(new_st, strict=False)

# edge map inference

img3 = np.stack([img, img, img], axis=0)  # [3, H, W]
input_tensor = torch.from_numpy(img3[None].astype(np.float32)).to(device)

# 4. Run PiDiNet inference
t0 = time()
with torch.no_grad():
    pidinet_edges = model(input_tensor)[-1].cpu().numpy()[0, 0]
edges = (pidinet_edges - pidinet_edges.min()) / (pidinet_edges.ptp() + 1e-8)
print(f"PiDiNet edge inference time: {(time()-t0)*1000:.1f} ms")

# width along normal from PiDiNet edge map
def width_pidinet(pruned, dist, edges, bw):
    from skimage.feature import structure_tensor, structure_tensor_eigenvalues
    Axx,Axy,Ayy = structure_tensor(bw.astype(float), sigma=1)
    lam1, _     = structure_tensor_eigenvalues([Axx,Axy,Ayy])
    vx, vy      = Axy, lam1 - Axx
    nrm         = np.hypot(vx,vy)+1e-8
    vx, vy      = vx/nrm, vy/nrm
    nx, ny      = -vy, vx
    w = np.zeros_like(bw, float)
    for y, x in zip(*np.nonzero(pruned)):
        L = int(np.ceil(dist[y,x]+1))
        p1 = (y+ny[y,x]*L, x+nx[y,x]*L)
        p2 = (y-ny[y,x]*L, x-nx[y,x]*L)
        prof = profile_line(edges, p1, p2, order=1, mode='constant', cval=0)
        eds  = np.where(np.diff((prof>0.05).astype(int))!=0)[0]
        if len(eds)>=2:
            w[y,x] = eds[-1]-eds[0]
    return w

t0 = time()
w_pid = width_pidinet(pruned, dist, edges, bw)
print(f"PiDiNet width extraction time: {(time()-t0)*1000:.1f} ms")'''

# 5) Plot Results
fig, axes = plt.subplots(1,2,figsize=(15,5))
for ax, wmap, title in zip(axes, [w_medial, w_subpix],
                           ["Medial+DSE","Subpixel-Edges"]):
    ax.imshow(bw, cmap='gray')
    sc = ax.scatter(xs, ys, c=wmap[pruned], cmap='viridis', s=12)
    ax.set_title(title)
    ax.axis('off')
    plt.colorbar(sc, ax=ax, label='Width (px)')
plt.tight_layout()
plt.show()
