import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from time import time

from skimage.io import imread
import cupy as cp
from cucim.skimage.morphology import medial_axis
from dsepruning.dsepruning import skel_pruning_DSE
from skimage.feature import structure_tensor, structure_tensor_eigenvalues
from skimage.measure import profile_line

import sys
sys.path.append('/mnt/c/Users/13144/Documents/Masters_Thesis/crack_width_measurement/pidinet')  # <- SET THIS to where you cloned PiDiNet
from models.pidinet import *
#from models.config import config_model
import torch

# ------------------- User parameters -------------------
img_path    = '/mnt/e/camerer_ml/datasets/METU_Concrete_Crack_Seg/BW_visible/236.png'  # <<--- Your test image
img_path = '/mnt/e/camerer_ml/datasets/METU_Concrete_Crack_Seg/BW/236.JPG'
threshold   = 0.25
min_area_px = 1000
pidinet_weights = '/mnt/c/Users/13144/Documents/Masters_Thesis/crack_width_measurement/pidinet/trained_models/table5_pidinet-small.pth'  # <<--- SET THIS!

# 1. LOAD + BINARIZE (your way)
img = imread(img_path, as_gray=True)
#print(np.unique(img))
#import sys; sys.exit(0)
bw  = (img > threshold)

# 2. SKELETON + MEDIAL AXIS + DSE
t0 = time()
sk_gpu, dist_gpu = medial_axis(cp.array(bw, dtype=bool), return_distance=True)
sk, dist   = sk_gpu.get(), dist_gpu.get()
pruned = skel_pruning_DSE(sk, dist, min_area_px=min_area_px, return_graph=False)
print(f"Medial axis+DSE time: {(time()-t0)*1000:.2f} ms")
ys, xs = np.nonzero(pruned)
w_medial = np.zeros_like(bw, float)
w_medial[pruned] = dist[pruned] * 2

# 3. PROFILE-LINE WIDTH (mask only)
def width_profile(bw, pruned, dist):
    im_f = bw.astype(np.float32)
    # structure tensor for local orientation
    Axx, Axy, Ayy = structure_tensor(im_f, sigma=1)
    lam1, _ = structure_tensor_eigenvalues([Axx, Axy, Ayy])
    vx = Axy
    vy = lam1 - Axx
    nrm = np.hypot(vx, vy) + 1e-8
    vx, vy = vx/nrm, vy/nrm
    nx, ny = -vy, vx

    w = np.zeros_like(bw, float)
    ys, xs = np.nonzero(pruned)
    for y, x in zip(ys, xs):
        L = int(np.ceil(dist[y, x] + 1))
        r1, c1 = y + ny[y, x]*L, x + nx[y, x]*L
        r2, c2 = y - ny[y, x]*L, x - nx[y, x]*L
        prof = profile_line(im_f, (r1, c1), (r2, c2),
                            order=1, mode='constant', cval=0)
        # Lower threshold for anti-aliased, noisy, or thin edges
        edges = np.where(np.diff((prof > 0.2).astype(int)) != 0)[0]
        if len(edges) >= 2:
            w[y, x] = edges[-1] - edges[0]
    return w

'''t0 = time()
w_profile = width_profile(bw, pruned, dist)
print(f"Profile-line time: {(time()-t0)*1000:.2f} ms")'''

# 4. PiDiNet EDGE WIDTH
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load('/mnt/c/Users/13144/Documents/Masters_Thesis/crack_width_measurement/pidinet/trained_models/table5_pidinet-small.pth', map_location=device)

from argparse import Namespace
#from models import pidinet

args = Namespace(config='carv4', sa=False, dil=True)
model = pidinet_small(args).to(device).eval()

#ckpt = torch.load(ckpt, map_location=device)
state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
from collections import OrderedDict
new_state = OrderedDict((k.replace('module.', ''), v) for k, v in state.items())
model.load_state_dict(new_state, strict=False)

# 3. Prepare your image as 3-channel (even if grayscale)
img3 = np.stack([img, img, img], axis=0)  # [3, H, W]
input_tensor = torch.from_numpy(img3[None].astype(np.float32)).to(device)

# 4. Run PiDiNet inference
t0 = time()
with torch.no_grad():
    pidinet_edges = model(input_tensor)[-1].cpu().numpy()[0, 0]
pidinet_edges = (pidinet_edges - pidinet_edges.min()) / (pidinet_edges.ptp() + 1e-8)
print(f"PiDiNet edge inference time: {(time() - t0) * 1000:.2f} ms")

# PiDiNet width extraction
def width_pidinet(pruned, dist, pidinet_edges, bw):
    Axx, Axy, Ayy = structure_tensor(bw.astype(float), sigma=1)
    lam1, _ = structure_tensor_eigenvalues([Axx, Axy, Ayy])
    vx =  Axy
    vy =  lam1 - Axx
    nrm = np.hypot(vx, vy) + 1e-8
    vx /= nrm; vy /= nrm
    nx, ny = -vy, vx

    ys, xs = np.nonzero(pruned)
    w = np.zeros_like(bw, float)
    for y, x in zip(ys, xs):
        L = int(np.ceil(dist[y, x] + 1))
        r1, c1 = y + ny[y, x]*L, x + nx[y, x]*L
        r2, c2 = y - ny[y, x]*L, x - nx[y, x]*L
        prof = profile_line(pidinet_edges, (r1, c1), (r2, c2),
                            order=1, mode='constant', cval=0)
        edges = np.where(np.diff((prof > 0.2).astype(int)) != 0)[0]
        if len(edges) >= 2:
            w[y, x] = edges[-1] - edges[0]
    return w

t0 = time()
w_pidinet = width_pidinet(pruned, dist, pidinet_edges, bw)
print(f"PiDiNet width extraction time: {(time()-t0)*1000:.2f} ms")

# ----------------- Visualization -----------------
'''plots = 2
fig, axes = plt.subplots(1, plots, figsize=(5 * plots, 5))
for ax, w, title in zip(axes, [w_medial, w_pidinet],
                        ["Medial+DSE", "PiDiNet Profile-Line"]):
    ax.imshow(bw, cmap='gray')
    im = ax.scatter(xs, ys, c=w[pruned], cmap='viridis', s=5 * plots)
    ax.set_title(title)
    ax.axis('off')
    fig.colorbar(im, ax=ax, label='Width (px)')
plt.tight_layout()
plt.show()'''

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.subplot(1,2,2)
plt.imshow(pidinet_edges, cmap='inferno')
plt.title('PiDiNet Edge Map')
plt.show()

# Optionally, plot error maps (vs. Medial Axis)
'''for w, title in zip([w_profile, w_pidinet], ["Profile-Line", "PiDiNet Profile-Line"]):
    plt.figure(figsize=(6,5))
    plt.imshow(bw, cmap='gray')
    plt.scatter(xs, ys, c=(w[pruned] - w_medial[pruned]), cmap='coolwarm', s=15)
    plt.title(f"{title} Error vs Medial")
    plt.axis('off')
    plt.colorbar(label='Width Error (px)')
    plt.show()'''
