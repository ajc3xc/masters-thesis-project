import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.morphology import skeletonize
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA

# --- User settings ---
IMG_PATH = '/mnt/e/camerer_ml/datasets/METU_Concrete_Crack_Seg/BW_visible/236.png'  # binary mask image
PIXEL_SIZE = 0.1  # mm per pixel (set according to your imaging system)

# Material properties
E = 30e3  # MPa
nu = 0.2
E_eff = E / (1 - nu**2)

# --- Helpers ---
def ctod_model(r, C):
    return C * np.sqrt(r)

def get_local_tangent_normal(skel, idx, window=5):
    y, x = idx
    y0, y1 = max(0, y - window), min(skel.shape[0], y + window + 1)
    x0, x1 = max(0, x - window), min(skel.shape[1], x + window + 1)
    pts = np.column_stack(np.nonzero(skel[y0:y1, x0:x1]))
    if pts.shape[0] < 3:
        return np.array([1.0, 0.0]), np.array([0.0, 1.0])
    pts += [y0, x0]
    pca = PCA(n_components=2).fit(pts)
    tang = pca.components_[0] / np.linalg.norm(pca.components_[0])
    norm = np.array([-tang[1], tang[0]])
    return tang, norm

# --- Load image and mask ---
mask = imread(IMG_PATH, as_gray=True) > 0.5
skel = skeletonize(mask)  # skeletonize the binary crack mask

# --- Order skeleton pixels (simple BFS from one endpoint) ---
ys, xs = np.nonzero(skel)
degree = { (y, x): sum(1 for dy,dx in [(1,0),(-1,0),(0,1),(0,-1)]
                       if 0 <= y+dy < skel.shape[0] and 0 <= x+dx < skel.shape[1]
                       and skel[y+dy, x+dx])
           for y, x in zip(ys, xs) }
endpoints = [pt for pt, d in degree.items() if d == 1]
# BFS to order points
from collections import deque
start = endpoints[0]
visited = set([start])
order = [start]
dq = deque([start])
while dq:
    y, x = dq.popleft()
    for dy, dx in [(1,0),(-1,0),(0,1),(0,-1)]:
        nb = (y+dy, x+dx)
        if nb not in visited and 0 <= nb[0] < skel.shape[0] and 0 <= nb[1] < skel.shape[1] and skel[nb]:
            visited.add(nb)
            order.append(nb)
            dq.append(nb)

# --- Measure raw widths and distances ---
rs, ws = [], []
for idx in order:
    r = len(rs) * PIXEL_SIZE  # approximate distance along skeleton
    _, normal = get_local_tangent_normal(skel, idx)
    y, x = idx
    forw = back = None
    for i in range(1, 100):
        yf, xf = int(round(y + normal[0] * i)), int(round(x + normal[1] * i))
        yb, xb = int(round(y - normal[0] * i)), int(round(x - normal[1] * i))
        if not (0 <= yf < mask.shape[0] and 0 <= xf < mask.shape[1]): break
        if not mask[yf, xf]:
            forw = np.array([yf, xf]); break
    for i in range(1, 100):
        yb2, xb2 = int(round(y - normal[0] * i)), int(round(x - normal[1] * i))
        if not (0 <= yb2 < mask.shape[0] and 0 <= xb2 < mask.shape[1]): break
        if not mask[yb2, xb2]:
            back = np.array([yb2, xb2]); break
    if forw is not None and back is not None:
        width_px = np.linalg.norm(forw - back)
        ws.append(width_px * PIXEL_SIZE)
        rs.append(r)

rs, ws = np.array(rs), np.array(ws)

# --- Fit CTOD model ---
popt, _ = curve_fit(ctod_model, rs, ws)
C_fit = popt[0]

# Compute stress intensity factor K
K = C_fit * E_eff * np.sqrt(np.pi / 8)  # MPa·√mm

# Physics-based width prediction
w_phys = ctod_model(rs, C_fit)

# --- Visualization ---
plt.figure(figsize=(6,4))
plt.scatter(rs, ws, s=10, label='Measured width')
plt.plot(rs, w_phys, 'r-', label='CTOD fit')
plt.xlabel('Distance from tip (mm)')
plt.ylabel('Width (mm)')
plt.title(f'Fitted CTOD: C={C_fit:.3f}, K={K:.1f} MPa·√mm')
plt.legend()
plt.tight_layout()
plt.show()
