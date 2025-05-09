import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from backprojection import backproject_surface_and_crack

data_dir = "projection_data00"
K = np.load(os.path.join(data_dir, "K.npy"))
depth = np.load(os.path.join(data_dir, "depth_00.npy"))
mask = cv2.imread(os.path.join(data_dir, "mask_00.png"), cv2.IMREAD_GRAYSCALE)
T_cw = np.load(os.path.join(data_dir, "Tcw_00.npy"))

# Project all points (crack and non-crack)
points_mean, points_var, is_crack = backproject_surface_and_crack(
    K, depth, mask, T_cw, perturbation_samples=25, include_noncrack=True
)

# Print crack count summary
unique, counts = np.unique(is_crack, return_counts=True)
print("Crack flags:", dict(zip(unique, counts)))

# Plot 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot non-crack surface points in gray
ax.scatter(points_mean[~is_crack, 0], points_mean[~is_crack, 1], points_mean[~is_crack, 2],
           s=0.5, c='gray', alpha=0.3, label='Surface', zorder=1)

# Plot crack points in red
ax.scatter(points_mean[is_crack, 0], points_mean[is_crack, 1], points_mean[is_crack, 2],
           s=1.5, c='red', alpha=0.9, label='Crack', zorder=5)

ax.set_title("3D Projection of Slab with Crack Overlay and Pose Perturbation")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.tight_layout()
plt.show()