import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, remove_small_objects
from skimage.graph import route_through_array
from scipy.spatial.distance import cdist
from skimage.draw import line

# ==== 1. Load image/mask ====
mask = cv2.imread(r'D:\camerer_ml\datasets\METU_Concrete_Crack_Seg\BW_visible\001.png')
if mask.ndim == 3:
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
else:
    mask_gray = mask.copy()
mask_bin = (mask_gray > 128).astype(np.uint8)
mask_bin = remove_small_objects(mask_bin.astype(bool), 64).astype(np.uint8)  # Remove noise

# ==== 2. Get and thin main skeleton ====
skeleton = skeletonize(mask_bin).astype(np.uint8)
ys, xs = np.where(skeleton)

# Find endpoints and use the longest path as the main skeleton (publication standard)
def find_skeleton_endpoints(skel):
    """Return coordinates of endpoints in the skeleton."""
    kernel = np.array([[1,1,1],[1,10,1],[1,1,1]])
    neighbor_count = cv2.filter2D(skel.astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT)
    endpoints = np.where((skel==1) & (neighbor_count==11))
    return list(zip(endpoints[1], endpoints[0]))  # (x, y)

endpoints = find_skeleton_endpoints(skeleton)
if len(endpoints) >= 2:
    # Find longest path between any two endpoints (main path)
    img_cost = np.where(skeleton, 1, 1000).astype(np.float32)
    best_path = []
    max_len = 0
    for i in range(len(endpoints)):
        for j in range(i+1, len(endpoints)):
            start, end = endpoints[i], endpoints[j]
            path, cost = route_through_array(img_cost, (start[1], start[0]), (end[1], end[0]), fully_connected=True)
            if len(path) > max_len:
                best_path = path
                max_len = len(path)
    skel_path = np.zeros_like(skeleton)
    for y, x in best_path:
        skel_path[y, x] = 1
else:
    skel_path = skeleton.copy()
ys, xs = np.where(skel_path)

# Make path_xy from best_path (ordered (x, y) points of main path)
if best_path and len(best_path) > 1:
    path_xy = np.array([(x, y) for y, x in best_path])  # (N, 2)
else:
    path_xy = np.column_stack((xs, ys))  # fallback to all skeleton points if no best_path

from scipy.interpolate import splprep, splev

# 1. Fit a smoothing spline to the path
tck, u = splprep([path_xy[:,0], path_xy[:,1]], s=3)

# 2. Arc-length for regular sampling
arc_length = np.sum(np.sqrt(np.sum(np.diff(path_xy, axis=0)**2, axis=1)))
N = max(2, int(arc_length // 15))  # sample every ~15 pixels
us = np.linspace(0, 1, N)
xs_new, ys_new = splev(us, tck)
smooth_coords = np.vstack([xs_new, ys_new]).T

# ==== 3. Filter skeleton points: spacing & curvature ====
# a) Downsample by minimum distance (no closer than 15 pixels apart)
coords = np.column_stack((xs, ys))
filtered_coords = []
min_dist = 15
for pt in coords:
    if not filtered_coords or np.min(cdist([pt], filtered_coords)) > min_dist:
        filtered_coords.append(pt)
filtered_coords = np.array(filtered_coords)

# b) Prune high-curvature points (remove if angle between tangents > threshold)
def local_tangent(pt_idx, coords, win=4):
    N = len(coords)
    idxs = np.arange(max(pt_idx-win,0), min(pt_idx+win+1,N))
    if len(idxs) < 2:
        return np.array([1,0])
    p_mean = coords[idxs].mean(axis=0)
    u, s, vh = np.linalg.svd(coords[idxs] - p_mean)
    direction = vh[0]
    return direction / np.linalg.norm(direction)

angle_threshold = np.pi / 4  # 45 degrees
pruned_coords = []
for i, pt in enumerate(filtered_coords):
    t1 = local_tangent(i, filtered_coords, win=4)
    if i == 0 or i == len(filtered_coords)-1:
        pruned_coords.append(pt)
    else:
        t0 = local_tangent(i-1, filtered_coords, win=4)
        angle = np.arccos(np.clip(np.dot(t0, t1), -1.0, 1.0))
        if angle < angle_threshold:
            pruned_coords.append(pt)
pruned_coords = np.array(pruned_coords)

# ==== 4. Draw perpendicular lines and extract widths ====
if mask.ndim == 2:
    img_disp = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
elif mask.ndim == 3 and mask.shape[2] == 3:
    img_disp = mask.copy()
else:
    raise ValueError("Unexpected mask shape!")
img_disp[skel_path > 0] = [0,255,0]  # skeleton as green

line_length = 40
microns_per_pixel = 3.96875

def get_mask_edge_points(x, y, normal, mask, max_length=40):
    h, w = mask.shape
    x0, y0 = int(round(x)), int(round(y))
    d = max_length // 2
    x1, y1 = int(round(x0 - normal[0]*d)), int(round(y0 - normal[1]*d))
    x2, y2 = int(round(x0 + normal[0]*d)), int(round(y0 + normal[1]*d))
    x1, y1 = np.clip(x1, 0, w-1), np.clip(y1, 0, h-1)
    x2, y2 = np.clip(x2, 0, w-1), np.clip(y2, 0, h-1)
    rr, cc = line(y1, x1, y2, x2)
    mask_vals = mask[rr, cc]
    center = len(mask_vals) // 2
    left = center
    while left > 0 and mask_vals[left]:
        left -= 1
    right = center
    while right < len(mask_vals)-1 and mask_vals[right]:
        right += 1
    pt1 = (cc[left], rr[left])
    pt2 = (cc[right], rr[right])
    return pt1, pt2

all_widths = []
all_centers = []
for i, (x, y) in enumerate(smooth_coords):
    # Compute tangent from spline derivative
    dx, dy = splev(us[i], tck, der=1)
    tangent = np.array([dx, dy])
    tangent /= np.linalg.norm(tangent)
    normal = np.array([-tangent[1], tangent[0]])
    pt1, pt2 = get_mask_edge_points(x, y, normal, mask_bin, max_length=line_length)
    cv2.line(img_disp, pt1, pt2, (0, 0, 255), 2)
    rr, cc = line(int(round(pt1[1])), int(round(pt1[0])), int(round(pt2[1])), int(round(pt2[0])))
    profile = mask_bin[rr, cc]
    width_pixels = np.sum(profile)
    width_um = width_pixels * microns_per_pixel
    all_widths.append(width_um)
    all_centers.append((x, y))

print("Measured crack widths at each ROI (um):", all_widths)

plt.figure(figsize=(10,12))
plt.imshow(cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB))
plt.title('Main Skeleton (green) and Perpendicular ROIs (red)')
plt.axis('off')
plt.show()

# ==== Save for reproducibility ====
import pandas as pd
df = pd.DataFrame({'x': [c[0] for c in all_centers],
                   'y': [c[1] for c in all_centers],
                   'width_um': all_widths})
df.to_csv('crack_widths_professional.csv', index=False)
