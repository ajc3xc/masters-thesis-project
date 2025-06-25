import torch
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path
from skimage.morphology import skeletonize
from scipy.ndimage import convolve
import time
import csv

# ---- Metric helpers ----
def count_endpoints(skel):
    kernel = np.array([[1,1,1],[1,10,1],[1,1,1]])
    filtered = convolve(skel.astype(np.uint8), kernel, mode='constant')
    return np.sum(filtered == 11)

def count_connected_components(skel):
    from skimage.measure import label
    return label(skel).max()

def prune_branches(skel, max_length=30):
    skel = skel.astype(np.uint8).copy()
    kernel = np.array([[1,1,1],[1,10,1],[1,1,1]])
    endpoints = np.argwhere(convolve(skel, kernel, mode='constant') == 11)
    for (y, x) in endpoints:
        path = [(y, x)]
        skel[y, x] = 0
        for _ in range(max_length):
            neighbors = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y+dy, x+dx
                    if (0 <= ny < skel.shape[0]) and (0 <= nx < skel.shape[1]) and skel[ny, nx]:
                        neighbors.append((ny, nx))
            if len(neighbors) != 1:
                break
            y, x = neighbors[0]
            path.append((y, x))
            skel[y, x] = 0
        if len(path) <= max_length:
            for (yy, xx) in path:
                skel[yy, xx] = 0
        else:
            for (yy, xx) in path:
                skel[yy, xx] = 1
    return skel.astype(bool)

def run_skelite(model, img_tensor, no_iter=5, mask_thresh=0.6):
    with torch.no_grad():
        output, _ = model(img_tensor, z=None, no_iter=no_iter)
    mask = output[0,0].cpu().numpy()
    skel_mask = mask > mask_thresh
    return mask, skel_mask

def overlay_skel(skel, img_gray, color=(0,255,0)):
    overlay = cv2.cvtColor((img_gray * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    ys, xs = np.nonzero(skel)
    for (y, x) in zip(ys, xs):
        cv2.circle(overlay, (int(x), int(y)), 1, color, -1)
    return overlay

# ---- Settings ----
input_folder = r"D:\camerer_ml\datasets\METU_Concrete_Crack_Seg\BW_visible"
model_path = r"D:\camerer_ml\adaptive_pca_3dreproject\Skelite\skelite_scripted.pt"
output_folder = r"D:\camerer_ml\adaptive_pca_3dreproject\Skelite\training_data\testing_outputs"

#input_folder = r"your_input_folder"        # CHANGE
#output_folder = r"your_output_folder"      # CHANGE or set to None
#model_path = r"skelite_scripted.pt"        # CHANGE
prune_len = 50

os.makedirs(output_folder, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.jit.load(model_path, map_location=device)
model.eval()

metrics = []

image_paths = sorted(list(Path(input_folder).glob("*.png")))
for img_path in image_paths:
    img = imageio.imread(str(img_path))
    if img.ndim == 3:
        img_gray = img.mean(axis=2)
    else:
        img_gray = img
    img_norm = img_gray / 255.0
    img_tensor = torch.tensor(img_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # ---- Classic skeleton ----
    t0 = time.time()
    mask_classic = (img_norm > 0.5).astype(np.uint8)
    skel_classic = skeletonize(mask_classic)
    skel_classic_pruned = prune_branches(skel_classic, max_length=prune_len)
    t1 = time.time()

    # ---- NN skeleton ----
    t2 = time.time()
    skelite_mask_raw, skelite_mask_bin = run_skelite(model, img_tensor, no_iter=5, mask_thresh=0.6)
    kernel = np.ones((3,3), np.uint8)
    mask_closed = cv2.morphologyEx(skelite_mask_bin.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1)
    skel_nn = skeletonize(mask_closed)
    skel_nn_pruned = prune_branches(skel_nn, max_length=prune_len)
    t3 = time.time()

    # ---- Metrics ----
    m = {
        "filename": img_path.name,
        "classic_time_s": t1-t0,
        "nn_time_s": t3-t2,
        "classic_endpoints": count_endpoints(skel_classic_pruned),
        "classic_components": count_connected_components(skel_classic_pruned),
        "classic_pixels": int(np.sum(skel_classic_pruned)),
        "nn_endpoints": count_endpoints(skel_nn_pruned),
        "nn_components": count_connected_components(skel_nn_pruned),
        "nn_pixels": int(np.sum(skel_nn_pruned)),
    }
    print(f"{img_path.name}: classic_time={m['classic_time_s']:.3f}s nn_time={m['nn_time_s']:.3f}s classic_branches={m['classic_endpoints']} nn_branches={m['nn_endpoints']}")
    metrics.append(m)

    # ---- Overlays (optional) ----
    if output_folder:
        ov_classic = overlay_skel(skel_classic_pruned, img_norm, color=(0,255,0))
        ov_nn = overlay_skel(skel_nn_pruned, img_norm, color=(255,0,0))
        cv2.imwrite(str(Path(output_folder) / f"{img_path.stem}_classic_overlay.png"), ov_classic)
        cv2.imwrite(str(Path(output_folder) / f"{img_path.stem}_nn_overlay.png"), ov_nn)
        # Save raw masks too if needed:
        # imageio.imwrite(str(Path(output_folder) / f"{img_path.stem}_classic_skel.png"), skel_classic_pruned.astype(np.uint8)*255)
        # imageio.imwrite(str(Path(output_folder) / f"{img_path.stem}_nn_skel.png"), skel_nn_pruned.astype(np.uint8)*255)

# ---- Save metrics ----
metrics_csv = Path(output_folder) / "metrics_summary.csv"
with open(metrics_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
    writer.writeheader()
    writer.writerows(metrics)

print(f"\nMetrics summary written to: {metrics_csv}")
