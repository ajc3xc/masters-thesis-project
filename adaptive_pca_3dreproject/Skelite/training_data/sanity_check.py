import torch
import imageio.v2 as imageio
import numpy as np
import os
from pathlib import Path
from skimage.morphology import skeletonize
from scipy.ndimage import convolve, label
import cv2

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

def count_endpoints(skel):
    kernel = np.array([[1,1,1],[1,10,1],[1,1,1]])
    filtered = convolve(skel.astype(np.uint8), kernel, mode='constant')
    return np.sum(filtered == 11)

def count_branches(skel):
    # Number of junctions: pixel with >2 neighbors (not endpoints)
    kernel = np.ones((3,3), np.uint8)
    neighbors = convolve(skel.astype(np.uint8), kernel, mode='constant') - skel
    return np.sum((skel) & (neighbors > 3))

def count_components(skel):
    lbl, num = label(skel)
    return num

# --- Folder paths ---
folder = r"D:\my_crack_masks_folder"   # CHANGE THIS!
output_csv = r"D:\my_crack_skeleton_metrics.csv"   # Results

image_files = sorted(list(Path(folder).glob("*.png")))

metrics = []
for imfile in image_files:
    img = imageio.imread(imfile)
    if img.ndim == 3:
        img = img.mean(axis=2)
    mask = (img > 128).astype(np.uint8)
    kernel = np.ones((3,3), np.uint8)
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    skel = skeletonize(mask_closed)
    skel_pruned = prune_branches(skel, max_length=30)
    # --- Metrics ---
    endpoints = count_endpoints(skel_pruned)
    branches = count_branches(skel_pruned)
    comps = count_components(skel_pruned)
    length = np.sum(skel_pruned)
    metrics.append((imfile.name, endpoints, branches, comps, length))

    # Optionally, save skeleton overlay for review
    overlay = cv2.cvtColor((img).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    ys, xs = np.nonzero(skel_pruned)
    for (y, x) in zip(ys, xs):
        cv2.circle(overlay, (int(x), int(y)), 1, (0,255,0), -1)
    cv2.imwrite(str(Path(folder)/f"{imfile.stem}_skeleton_overlay.png"), overlay)

# --- Save results to CSV
import csv
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "endpoints", "branches", "components", "skeleton_length"])
    for row in metrics:
        writer.writerow(row)

print(f"Processed {len(metrics)} images! Results saved to {output_csv}")

image_path = r"D:\camerer_ml\datasets\METU_Concrete_Crack_Seg\BW_visible\064.png"
model_path = r"D:\camerer_ml\adaptive_pca_3dreproject\Skelite\skelite_scripted.pt"
