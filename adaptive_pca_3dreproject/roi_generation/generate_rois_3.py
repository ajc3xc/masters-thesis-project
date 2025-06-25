import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label, regionprops
from skimage.draw import line
from scipy.spatial.distance import cdist

# ==== 1. Load and preprocess mask (keep largest component) ====
mask = cv2.imread(r'D:\camerer_ml\datasets\METU_Concrete_Crack_Seg\BW_visible\001.png')
if mask.ndim == 3:
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
else:
    mask_gray = mask.copy()
mask_bin = (mask_gray > 128).astype(np.uint8)
mask_bin = remove_small_objects(mask_bin.astype(bool), 64).astype(np.uint8)

# Keep only the largest connected component
from skimage.measure import label
labels = label(mask_bin)
largest = np.argmax(np.bincount(labels.flat)[1:]) + 1
mask_bin = (labels == largest).astype(np.uint8)

# ==== 2. Thinning-based skeletonization ====
skeleton = skeletonize(mask_bin).astype(np.uint8)

# ==== 3. Spur (branch) pruning ====
'''def prune_skeleton(skel, min_branch_length=20):
    # Label all skeleton pixels
    from skimage.graph import MCP
    import networkx as nx
    skel = skel.copy()
    # Find branch points (pixel with >2 neighbors)
    neighbor_kernel = np.array([[1,1,1],[1,10,1],[1,1,1]])
    neighbor_count = cv2.filter2D(skel.astype(np.uint8), -1, neighbor_kernel, borderType=cv2.BORDER_CONSTANT)
    branch_points = ((skel==1) & (neighbor_count > 12))
    skel_labeled = label(skel)
    # Get all endpoints (pixels with only one neighbor)
    endpoints = ((skel==1) & (neighbor_count==11))
    # Trace all branch segments
    coords = np.column_stack(np.where(skel))
    to_remove = []
    for y, x in coords:
        if endpoints[y,x]:
            # Trace out from each endpoint
            path = [(y, x)]
            visited = set(path)
            prev = (y, x)
            for _ in range(min_branch_length+1):
                # Find next neighbor
                nbrs = [(yy, xx) for yy in range(y-1,y+2) for xx in range(x-1,x+2)
                        if (yy,xx)!=(y,x) and 0<=yy<skel.shape[0] and 0<=xx<skel.shape[1] and skel[yy,xx]]
                nbrs = [p for p in nbrs if p not in visited]
                if not nbrs:
                    break
                nxt = nbrs[0]
                path.append(nxt)
                visited.add(nxt)
                y, x = nxt
                if branch_points[y,x]:
                    break
            if len(path) <= min_branch_length:
                to_remove.extend(path)
    for y, x in to_remove:
        skel[y, x] = 0
    return skel

skeleton = prune_skeleton(skeleton, min_branch_length=.1)'''

# ==== 4. Downsample skeleton points by min spacing (as before) ====
ys, xs = np.where(skeleton)
coords = np.column_stack((xs, ys))
coords = coords[coords[:,1].argsort()]
min_spacing = 15
filtered_coords = []
for pt in coords:
    if not filtered_coords or np.min(cdist([pt], filtered_coords)) > min_spacing:
        filtered_coords.append(pt)
filtered_coords = np.array(filtered_coords)

# ==== 5. Perpendicular ROI measurement (as before) ====
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

img_disp = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)
img_disp[skeleton > 0] = [0,255,0]  # green skeleton

line_length = 40
microns_per_pixel = 3.96875

all_widths, all_centers, all_normals = [], [], []
for x, y in filtered_coords:
    tangent = get_tangent(x, y, skeleton, win=4)
    normal = np.array([-tangent[1], tangent[0]])
    pt1, pt2, width_pixels = get_mask_edge_points(x, y, normal, mask_bin, max_length=line_length)
    cv2.line(img_disp, pt1, pt2, (0, 0, 255), 2)
    width_um = width_pixels * microns_per_pixel
    all_widths.append(width_um)
    all_centers.append((x, y))
    all_normals.append(normal)

plt.figure(figsize=(7,14))
plt.imshow(cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB))
plt.title('Pruned Skeleton & Fast Perpendicular ROIs (Green: skeleton, Red: ROIs)')
plt.axis('off')
plt.show()

# ==== 6. Width outlier filtering (optional) ====
widths_arr = np.array(all_widths)
med_width = np.median(widths_arr)
good_idx = np.where(widths_arr < 2.5 * med_width)[0]

all_widths_filtered = [all_widths[i] for i in good_idx]
all_centers_filtered = [all_centers[i] for i in good_idx]

# ==== 7. Save measurements ====
import pandas as pd
df = pd.DataFrame({'x': [c[0] for c in all_centers_filtered],
                   'y': [c[1] for c in all_centers_filtered],
                   'width_um': all_widths_filtered})
df.to_csv('crack_widths_pruned.csv', index=False)
print("Saved pruned ROI coordinates and widths to crack_widths_pruned.csv")