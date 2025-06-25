import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.ndimage import gaussian_filter1d
# ==== 1. Load image (mask or raw image) ====
mask = cv2.imread(r'D:\camerer_ml\datasets\METU_Concrete_Crack_Seg\BW_visible\001.png')   # or any PNG/JPG/TIF
if mask.ndim == 3:
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
else:
    mask_gray = mask.copy()
mask_bin = (mask_gray > 128).astype(np.uint8)
skeleton = skeletonize(mask_bin).astype(np.uint8)

# 2. Get skeleton (y, x) coordinates
if skeleton.ndim > 2:
    skeleton = skeleton[..., 0]
skeleton = (skeleton > 0).astype(np.uint8)
ys, xs = np.where(skeleton)
#print("Skeleton points:", len(xs))
#print("Sample points:", xs[:10], ys[:10])

# 3. Sample skeleton points at intervals (e.g., every 20 pixels)
indices = np.arange(0, len(xs), 20)
sampled_points = list(zip(xs[indices], ys[indices]))

# 4. For each sampled skeleton point, estimate local tangent using finite difference
def get_tangent(x, y, skeleton, win=5):
    h, w = skeleton.shape
    # Get neighboring points within a window
    points = []
    for dx in range(-win, win+1):
        for dy in range(-win, win+1):
            xx, yy = x+dx, y+dy
            if 0 <= xx < w and 0 <= yy < h and skeleton[yy, xx]:
                points.append([xx, yy])
    if len(points) < 2:
        return np.array([1, 0])  # default horizontal
    points = np.array(points)
    # PCA to get main direction
    points_mean = points.mean(axis=0)
    uu, dd, vv = np.linalg.svd(points - points_mean)
    direction = vv[0]  # principal direction
    return direction / np.linalg.norm(direction)

# 5. Draw short lines perpendicular to skeleton
print("mask shape:", mask.shape)
if mask.ndim == 2:
    img_disp = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
elif mask.ndim == 3 and mask.shape[2] == 3:
    img_disp = mask.copy()  # Already BGR/RGB
elif mask.ndim == 3 and mask.shape[2] == 1:
    img_disp = cv2.cvtColor(mask[...,0], cv2.COLOR_GRAY2BGR)
else:
    raise ValueError("Unexpected mask shape!")
line_length = 40  # pixels (total), so half = 20

from skimage.draw import line

def get_mask_edge_points(x, y, normal, mask, max_length=40):
    h, w = mask.shape
    x0, y0 = int(round(x)), int(round(y))
    d = max_length // 2

    # Compute end points of the full ROI line
    x1, y1 = int(round(x0 - normal[0]*d)), int(round(y0 - normal[1]*d))
    x2, y2 = int(round(x0 + normal[0]*d)), int(round(y0 + normal[1]*d))
    # Clip to image bounds
    x1 = np.clip(x1, 0, w-1)
    y1 = np.clip(y1, 0, h-1)
    x2 = np.clip(x2, 0, w-1)
    y2 = np.clip(y2, 0, h-1)

    rr, cc = line(y1, x1, y2, x2)
    mask_vals = mask[rr, cc]
    center = len(mask_vals) // 2

    # Find left edge
    left = center
    while left > 0 and mask_vals[left]:
        left -= 1
    # Find right edge
    right = center
    while right < len(mask_vals)-1 and mask_vals[right]:
        right += 1
    # Output endpoints at the crack edge
    pt1 = (cc[left], rr[left])
    pt2 = (cc[right], rr[right])
    return pt1, pt2

# Overlay skeleton in green first
img_disp[skeleton > 0] = [0,255,0]  # skeleton as green

# Now draw red perpendicular lines as before
all_widths = []
all_centers = []

for (x, y) in sampled_points:
    tangent = get_tangent(x, y, skeleton)
    normal = np.array([-tangent[1], tangent[0]])
    pt1, pt2 = get_mask_edge_points(x, y, normal, mask_bin, max_length=40)
    cv2.line(img_disp, pt1, pt2, (0, 0, 255), 2)

    # Extract the profile along the ROI
    from skimage.draw import line
    rr, cc = line(pt1[1], pt1[0], pt2[1], pt2[0])
    profile = mask_bin[rr, cc]  # 1D array along the ROI

    # Crack width = number of foreground pixels
    width_pixels = np.sum(profile)
    # (Optional) Convert to microns
    microns_per_pixel = 3.96875  # or your value
    width_um = width_pixels * microns_per_pixel

    all_widths.append(width_um)
    all_centers.append((x, y))

print("Measured crack widths at each ROI (um):", all_widths)

# 6. Visualize
plt.figure(figsize=(8,10))
plt.imshow(cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB))
plt.title('Perpendicular ROIs along Crack Skeleton')
plt.axis('off')
plt.show()