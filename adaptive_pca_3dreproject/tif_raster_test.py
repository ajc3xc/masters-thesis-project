import paddle
import numpy as np
from tifffile import imread
import cv2
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
import pandas as pd
from pathlib import Path
from models.hrsegnet_b16 import HrSegNetB16

DEVICE = 'gpu'
WEIGHTS = r"E:\camerer_ml\finished_models\hrsegnetb16_tl\model.pdparams"

# set Paddle device
pd_dev = 'gpu' if (DEVICE=='gpu' and paddle.is_compiled_with_cuda()) else 'cpu'
print(paddle.is_compiled_with_cuda())
print(f"Using device: {pd_dev}")
paddle.set_device(pd_dev)

# load HRSegNet
model = HrSegNetB16(3, base=16, num_classes=2)
state_dict = paddle.load(str(WEIGHTS))
model.set_state_dict(state_dict)
model.eval()

# ---- Load and preprocess a patch from TIF image ----
img = imread(r"C:\Users\ajc3xc\Downloads\krkCMd_images\CMd_0.23_20mths\CMd_0.23_20mths_Image1.tif")
# If multi-page, select first frame
if img.ndim == 4:
    img = img[0]
    print("Selected first page, new shape:", img.shape)
# If img has more than 3 channels (e.g., (C, H, W)), convert to (H, W, C)
if img.ndim == 3:
    if img.shape[0] <= 4:  # (C, H, W)
        img = np.transpose(img, (1, 2, 0))
    elif img.shape[2] > 4:
        # Possibly multispectral, pick first 3 channels or convert as needed
        img = img[:, :, :3]
elif img.ndim == 2:
    # If grayscale, stack to make 3-channel
    img = np.stack([img]*3, axis=-1)

# Now img is (H, W, 3)
#img_resized = cv2.resize(img, (512, 512))
img_norm = img / 255.0

# Confirm shape before transpose
assert img_norm.ndim == 3 and img_norm.shape[2] == 3, f"Shape is {img_norm.shape}, expected (H, W, 3)"

img_input = np.transpose(img_norm, (2, 0, 1)).astype(np.float32)  # (C, H, W)
#img_input = img_input[np.newaxis, :]  # (1, C, H, W)
img_input = paddle.to_tensor(img_input[np.newaxis, :])  # (1, C, H, W)

#print(np.unique(img_input))

# ---- Run model inference ----
with paddle.no_grad():
    logits = model(img_input)
    # If model returns list of logits (aux heads), use the main head
    if isinstance(logits, (list, tuple)):
        mask_pred = logits[0][0].numpy()  # (C, H, W)
    else:
        mask_pred = logits[0].numpy()     # (C, H, W)
    # If mask_pred shape is (2, H, W): get crack class (e.g., channel 1)
    if mask_pred.shape[0] > 1:
        mask_pred = mask_pred[1]          # Use the second channel for cracks
    # (H, W)
mask_bin = (mask_pred > 0.0).astype(np.uint8)

#print(np.unique(logits))
#print(np.unique(mask_bin))

import matplotlib.pyplot as plt

# ---- Classical width estimation ----
from skimage.morphology import binary_dilation, disk

# Step 1: Dilate the crack mask to thicken cracks
mask_thick = binary_dilation(mask_bin, disk(3))  # Try disk(2), increase if cracks are still too thin

# Step 2: Skeletonize the thickened mask
skeleton = skeletonize(mask_thick)

# Step 3: Compute distance transform (on thickened mask)
distance = distance_transform_edt(mask_thick == 1)

ys, xs = np.where(skeleton)
for idx in range(0, len(ys), max(1, len(ys)//20)):  # sample 20 skeleton points evenly
    y, x = ys[idx], xs[idx]
    print(f"Skeleton at ({y},{x}): mask={mask_thick[y,x]}, distance={distance[y,x]}")

# Step 4: Crack widths at skeleton pixels
crack_widths = 2 * distance[skeleton]
mean_width = np.mean(crack_widths) if crack_widths.size else 0

# --- Diagnostic block: Check values at each step ---
'''print("mask_bin unique:", np.unique(mask_bin))
print("mask_thick unique:", np.unique(mask_thick))
print("Number of skeleton pixels:", np.sum(skeleton))
print("Distance transform min/max:", distance.min(), distance.max())
print("Sample of distance[skeleton]:", distance[skeleton][:20])  # Show the first 20 values
print("Crack widths (in pixels) at skeleton:", crack_widths[:20])  # First 20 widths
print("Nonzero crack widths:", crack_widths[crack_widths > 0])
print("Max crack width (pixels):", np.max(crack_widths) if crack_widths.size else 0)
print("Mean crack width (pixels):", np.mean(crack_widths) if crack_widths.size else 0)'''


# Step 5: Convert to microns
microns_per_pixel = 3.96875
mean_width_um = mean_width * microns_per_pixel

print("Estimated crack width after dilation (um):", mean_width_um)

# Optional: Plot the results for visual confirmation
mask_rgb = np.stack([mask_bin*255]*3, axis=-1).astype(np.uint8)  # shape (H, W, 3)

# Color the skeleton in red over the mask
overlay = mask_rgb.copy()
overlay[skeleton == 1] = [255, 0, 0]  # Red for skeleton

mask_rgb = np.stack([mask_thick.astype(np.uint8)*255]*3, axis=-1)
overlay = mask_rgb.copy()
overlay[skeleton == 1] = [255, 0, 0]
plt.figure(figsize=(6,6))
plt.title("Skeleton over Dilated Mask")
plt.imshow(overlay)
plt.axis('off')
plt.show()

distance = distance_transform_edt(mask_thick == 0)
crack_widths_all = 2 * distance[mask_thick == 1]
mean_width_um_all = np.mean(crack_widths_all) * microns_per_pixel if crack_widths_all.size else 0
print("Mean crack width (all crack pixels, dilated mask):", mean_width_um_all, np.unique(mask_thick))

# ---- Compare to ground truth from CSV ----
csv = pd.read_csv(r"C:\Users\ajc3xc\Downloads\krkCMd_table.csv")
# TODO: Lookup the correct row for this patch/ROI (based on your workflow)
# Find by image file (matches 'CMd_0.23_2mths' series and 'Image' == 1)
row = csv[(csv['Series'] == 'CMd_0.23_2mths') & (csv['Image'] == 1)]

if not row.empty:
    ground_truth_width = row.iloc[0]['MANwidth']
    print("Ground truth width (um):", ground_truth_width)
    print("Difference (um):", mean_width_um - ground_truth_width)
else:
    print("No matching row found for Series='CMd_0.23_2mths', Image=1")

# ---- (Optional) Regression Model ----
from sklearn.linear_model import LinearRegression

# Suppose you have a batch of patches/masks, build up X and y
# feature_list = [[mean_width_patch1], [mean_width_patch2], ...]
# label_list = [width_gt1, width_gt2, ...]
# X = np.array(feature_list)
# y = np.array(label_list)
# reg = LinearRegression().fit(X, y)
# predicted_width = reg.predict([[mean_width_new_patch]])[0]
