
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from backprojection import backproject_surface_and_crack

data_dir = "multi_crack_projection_test_data"
K = np.load(os.path.join(data_dir, "K.npy"))
depth = np.load(os.path.join(data_dir, "depth_00.npy"))
mask = cv2.imread(os.path.join(data_dir, "mask_00.png"), cv2.IMREAD_GRAYSCALE)
T_cw = np.load(os.path.join(data_dir, "Tcw_00.npy"))

points, is_crack = backproject_surface_and_crack(K, depth, mask, T_cw)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[~is_crack, 0], points[~is_crack, 1], points[~is_crack, 2], s=0.5, c='gray', alpha=0.3)
ax.scatter(points[is_crack, 0], points[is_crack, 1], points[is_crack, 2], s=1.5, c='red', label='Crack')
ax.set_title("3D Projection of Slab Surface with Crack Overlay")
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.legend()
plt.tight_layout()
plt.show()
