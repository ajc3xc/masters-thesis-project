import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label
from scipy.spatial.distance import cdist
from skimage.draw import line

# ==== 1. Load and preprocess mask ====
mask = cv2.imread(r'D:\camerer_ml\datasets\METU_Concrete_Crack_Seg\BW_visible\001.png')
if mask.ndim == 3:
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
else:
    mask_gray = mask.copy()
mask_bin = (mask_gray > 128).astype(np.uint8)
mask_bin = remove_small_objects(mask_bin.astype(bool), 64).astype(np.uint8)

# ==== 2. Skeletonize ====
skeleton = skeletonize(mask_bin).astype(np.uint8)

# ==== 3. Label each branch ====
labels = label(skeleton)
n_labels = np.max(labels)
print(f"Found {n_labels} cracks/branches in skeleton")

img_disp = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)
img_disp[skeleton > 0] = [0,255,0]  # skeleton in green

line_length = 40
microns_per_pixel = 3.96875

def get_tangent(x, y, skeleton, win=4):
    h, w = skeleton.shape
    points = []
    for dx in range(-win, win+1):
        for dy in range(-win, win+1):
            xx, yy = x+dx, y+dy
            if 0 <= xx < w and 0 <= yy < h and skeleton[yy, xx]:
                points.append([xx, yy])
    if len(points) < 2:
        return np.array([1,0])
    points = np.array(points)
    mean = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - mean)
    direction = vh[0]
    return direction / np.linalg.norm(direction)

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
    return pt1, pt2, np.sum(mask_vals)

all_widths, all_centers, all_branch_ids = [], [], []

for branch_id in range(1, n_labels+1):
    skel_branch = (labels == branch_id)
    ys, xs = np.where(skel_branch)
    coords = np.column_stack((xs, ys))
    # Skip if too short (e.g. < 10 pixels)
    if len(coords) < 10:
        continue
    # Sort by y for vertical cracks, or x for horizontal (change as needed)
    coords = coords[coords[:,1].argsort()]
    # Sample points along the branch
    min_spacing = 15
    filtered_coords = []
    for pt in coords:
        if not filtered_coords or np.min(cdist([pt], filtered_coords)) > min_spacing:
            filtered_coords.append(pt)
    filtered_coords = np.array(filtered_coords)
    for x, y in filtered_coords:
        tangent = get_tangent(x, y, skel_branch, win=4)
        normal = np.array([-tangent[1], tangent[0]])
        pt1, pt2, width_pixels = get_mask_edge_points(x, y, normal, mask_bin, max_length=line_length)
        cv2.line(img_disp, pt1, pt2, (0, 0, 255), 2)
        width_um = width_pixels * microns_per_pixel
        all_widths.append(width_um)
        all_centers.append((x, y))
        all_branch_ids.append(branch_id)

plt.figure(figsize=(8,12))
plt.imshow(cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB))
plt.title('Perpendicular ROIs along all Crack Skeleton Branches')
plt.axis('off')
plt.show()

import pandas as pd
df = pd.DataFrame({
    'x': [c[0] for c in all_centers],
    'y': [c[1] for c in all_centers],
    'width_um': all_widths,
    'branch_id': all_branch_ids
})
df.to_csv('crack_widths_all_branches.csv', index=False)
print("Saved ROI coordinates, widths, and branch IDs to crack_widths_all_branches.csv")
