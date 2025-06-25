import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import medial_axis
from dsepruning.dsepruning import skel_pruning_DSE

# ----- Hardcoded path -----
img_path = '/mnt/d/camerer_ml/datasets/METU_Concrete_Crack_Seg/BW_visible/236.png'  # <-- Change this to your actual image file

# ----- Load and binarize image -----
img = imread(img_path, mode='L')  # Load as grayscale
bw = (img > 0.5)

# ----- Medial axis skeletonization -----
skeleton, dist = medial_axis(bw, return_distance=True)

# ----- DSE pruning -----
min_area_px = 1000  # Try adjusting this (3, 5, 10) for more/less pruning
from time import time
start_time = time()
pruned_skel = skel_pruning_DSE(skeleton, dist, min_area_px=min_area_px, return_graph=False)
end_time = time()
print(f"DSE pruning took {end_time - start_time:.2f} seconds")
from scipy.ndimage import distance_transform_edt, binary_dilation
pruned_vis = binary_dilation(pruned_skel, iterations=2)
skel_vis = binary_dilation(skeleton, iterations=2)

# ----- Show results -----
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(bw, cmap='gray')
axes[0].set_title('Binary Input')
axes[0].axis('off')

axes[1].imshow(skel_vis, cmap='gray')
axes[1].set_title('Medial Axis Skeleton')
axes[1].axis('off')

axes[2].imshow(pruned_vis, cmap='gray')
axes[2].set_title(f'DSE Pruned (min_area_px={min_area_px})')
axes[2].axis('off')

plt.tight_layout()
plt.show()
