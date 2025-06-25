import torch
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, medial_axis
from scipy.ndimage import convolve, distance_transform_edt
from pathlib import Path
import os

def run_skelite(model, img_tensor):
    with torch.no_grad():
        output, _ = model(img_tensor, z=None, no_iter=5)
    mask = output[0,0].cpu().numpy()
    skel_mask = mask > 0.5
    return mask, skel_mask

def load_and_prep_image(image_path, device):
    img = imageio.imread(image_path)
    if img.ndim == 3:
        img = img.mean(axis=2)
    img = img / 255.0
    tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    return tensor, img

def find_endpoints(skel):
    kernel = np.array([[1,1,1],[1,10,1],[1,1,1]])
    filtered = convolve(skel.astype(np.uint8), kernel, mode='constant')
    return np.argwhere(filtered == 11)

def prune_branches(skel, max_length=15):
    # Prune branches shorter than max_length pixels from endpoints to nearest junction
    skel = skel.astype(np.uint8).copy()
    endpoints = find_endpoints(skel)
    for (y, x) in endpoints:
        path = [(y, x)]
        skel[y, x] = 0  # Remove this endpoint for the trace
        for _ in range(max_length):
            # Check all neighbors
            neighbors = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y+dy, x+dx
                    if (0 <= ny < skel.shape[0]) and (0 <= nx < skel.shape[1]):
                        if skel[ny, nx]:
                            neighbors.append((ny, nx))
            if len(neighbors) != 1:
                # Stop if at a junction or at the end
                break
            y, x = neighbors[0]
            path.append((y, x))
            skel[y, x] = 0  # Mark as visited
        # If this branch is short, erase it
        if len(path) <= max_length:
            for (yy, xx) in path:
                skel[yy, xx] = 0
        else:
            # Restore branch if it's not too short
            for (yy, xx) in path:
                skel[yy, xx] = 1
    return skel.astype(bool)

def save_binary_skeleton(skel, out_path):
    # Save as uint8 PNG (pure white on black)
    imageio.imwrite(out_path, (skel.astype(np.uint8)*255))

def save_comparison_plot(image_name, img, skel1, skel2, skel3, skel4, save_dir):
    plt.figure(figsize=(24, 6), dpi=200)
    titles = [
        'Original', 
        'skimage.skeletonize + prune', 
        'Medial Axis + prune', 
        'Skelite+Thinning+prune', 
        'Skelite Mask (>0.5)'
    ]
    images = [
        img, 
        skel1, 
        skel2, 
        skel3, 
        skel4
    ]
    for i in range(len(images)):
        plt.subplot(1, len(images), i+1)
        plt.imshow(images[i], cmap='gray', vmin=0, vmax=1)
        plt.title(titles[i], fontsize=14)
        plt.axis('off')
    plt.tight_layout()
    out_path = os.path.join(save_dir, f"{Path(image_name).stem}_comparison.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

# --- Batch processing setup ---
image_dir = r"D:\camerer_ml\datasets\METU_Concrete_Crack_Seg\BW_visible"
model_path = r'D:\camerer_ml\adaptive_pca_3dreproject\Skelite\skelite_scripted.pt'
output_dir = r"D:\camerer_ml\adaptive_pca_3dreproject\Skelite\skeleton_comparisons_v2"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "skel_skimage_pruned"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "skel_medial_pruned"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "skel_skelite_thin_pruned"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "skelite_mask"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "comparison_plots"), exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.jit.load(model_path, map_location=device)
model.eval()

all_image_paths = sorted(list(Path(image_dir).glob("*.png")))

for img_path in all_image_paths:
    img_tensor, img = load_and_prep_image(str(img_path), device)
    bw = img > 0.5

    # Classic and medial
    skel_skimage = skeletonize(bw)
    skel_medial = medial_axis(bw)
    _, skel_skelite = run_skelite(model, img_tensor)
    skel_skelite_thin = skeletonize(skel_skelite)

    # Prune short branches (set max_length as needed)
    skel_skimage_pruned = prune_branches(skel_skimage, max_length=15)
    skel_medial_pruned = prune_branches(skel_medial, max_length=15)
    skel_skelite_thin_pruned = prune_branches(skel_skelite_thin, max_length=15)

    # Save individual binary images (skeletons)
    base = Path(img_path).stem
    save_binary_skeleton(skel_skimage_pruned, os.path.join(output_dir, "skel_skimage_pruned", f"{base}_skimage_pruned.png"))
    save_binary_skeleton(skel_medial_pruned, os.path.join(output_dir, "skel_medial_pruned", f"{base}_medial_pruned.png"))
    save_binary_skeleton(skel_skelite_thin_pruned, os.path.join(output_dir, "skel_skelite_thin_pruned", f"{base}_skelite_thin_pruned.png"))
    save_binary_skeleton(skel_skelite, os.path.join(output_dir, "skelite_mask", f"{base}_skelite_mask.png"))

    # Save side-by-side visual (for qualitative review)
    save_comparison_plot(
        img_path.name, img, skel_skimage_pruned*1.0, skel_medial_pruned*1.0, skel_skelite_thin_pruned*1.0, skel_skelite*1.0, 
        os.path.join(output_dir, "comparison_plots")
    )

print("Batch processing complete! Outputs saved in:", output_dir)
