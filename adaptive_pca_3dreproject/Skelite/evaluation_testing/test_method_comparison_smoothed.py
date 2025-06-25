import torch
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.morphology import skeletonize
from skimage.measure import label
from scipy.ndimage import convolve, distance_transform_edt, binary_opening

# --- Metrics ---
def count_connected_components(skel):
    return label(skel).max()

def count_endpoints(skel):
    kernel = np.array([[1,1,1],[1,10,1],[1,1,1]])
    filtered = convolve(skel.astype(np.uint8), kernel, mode='constant')
    return np.sum(filtered == 11)

def mean_thickness(skel_img):
    if np.sum(skel_img) == 0:
        return 0
    dist = distance_transform_edt(skel_img)
    return dist[skel_img > 0].mean()

def print_metrics(name, skel):
    print(f"\n{name} Skeleton:")
    print("  Connected components:", count_connected_components(skel))
    print("  Endpoints:", count_endpoints(skel))
    print("  Mean thickness:", mean_thickness(skel))
    print("  Skeleton pixel count:", np.sum(skel))

# --- Pruning ---
def prune_branches(skel, max_length=10):
    skel = skel.astype(np.uint8).copy()
    endpoints = np.argwhere(convolve(skel, np.array([[1,1,1],[1,10,1],[1,1,1]])) == 11)
    for (y, x) in endpoints:
        path = [(y, x)]
        skel[y, x] = 0
        for _ in range(max_length):
            neighbors = [(y+dy, x+dx) for dy in [-1,0,1] for dx in [-1,0,1] if not (dy==0 and dx==0)]
            neighbors = [(ny,nx) for ny,nx in neighbors if 0 <= ny < skel.shape[0] and 0 <= nx < skel.shape[1] and skel[ny,nx]]
            if len(neighbors) != 1:
                break
            y, x = neighbors[0]
            path.append((y, x))
            skel[y, x] = 0
        if len(path) > max_length:
            for yy, xx in path:
                skel[yy, xx] = 1
    return skel.astype(bool)

# --- Skelite inference ---
'''def run_skelite(model: torch.jit.ScriptModule, img_tensor: torch.Tensor):
    """
    Runs Skelite on img_tensor (1×1×H×W). Returns:
      - raw_mask:     float32 NumPy array of shape (H, W), values in [0,1]
      - skel_mask:    boolean NumPy array of shape (H, W), thresholded at 0.5
    """
    with torch.no_grad():
        output, _ = model(img_tensor, z=None, no_iter=20)
    raw_mask = output[0, 0].cpu().numpy()
    skel_mask = raw_mask > 0.5
    return raw_mask, skel_mask'''

def load_and_prep_image(image_path, device):
    img = imageio.imread(image_path)
    if img.ndim == 3:
        img = img.mean(axis=2)
    img = img / 255.0
    tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    return tensor, img

# --- Keep largest connected region ---
def keep_largest_connected(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea)
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(cleaned, [largest], -1, 1, thickness=cv2.FILLED)
    return cleaned.astype(bool)

# === Main ===
image_path = r"D:\camerer_ml\datasets\METU_Concrete_Crack_Seg\BW_visible\024.png"
model_path = r'D:\camerer_ml\adaptive_pca_3dreproject\Skelite\skelite_scripted.pt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.jit.load(model_path, map_location=device)
model.eval()

img_tensor, img = load_and_prep_image(image_path, device)
from scipy.ndimage import binary_dilation

# === Skelite inference ===
with torch.no_grad():
    output, _ = model(img_tensor, z=None, no_iter=5, val_mode=True)
    raw_mask = output[0, 0].cpu().numpy()
    skel_mask = raw_mask > 0.5

# === Constrained skeletonization ===

# 1. Get original binary mask from input image
original_bw = img > 0.5

# 2. Slight dilation to allow for alignment mismatch
dilated_orig = binary_dilation(original_bw, structure=np.ones((3, 3)))

# 3. Skeletonize Skelite mask (1-pixel centerline)
skeleton = skeletonize(skel_mask)

# 4. Combine only where both masks agree (preserve high-confidence skeleton)
combined = skeleton & skel_mask & dilated_orig

# 5. Prune short spurs
skeleton_pruned = prune_branches(combined, max_length=10)

# --- Display ---
print_metrics("Final Skeleton (pruned)", skeleton_pruned)

plt.figure(figsize=(16, 4))
titles = ['Original', 'Skelite Mask (raw)', 'Dilated Original Mask', 'Combined Mask', 'Final Skeleton']
images = [img, skel_mask, dilated_orig, combined, skeleton_pruned]

for i, im in enumerate(images):
    plt.subplot(1, len(images), i+1)
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
