#!/usr/bin/env python3
"""
Script: evaluate_models_on_metu.py

Runs a sanity-check evaluation of one or more exported Skelite models
against a classical skeletonization pipeline on 100 pseudo-random METU images.

For each model, measures:
  - Model inference time (per image, averaged)
  - Model inference + skeletonize time (per image, averaged)
  - Skeleton stats: endpoint count, connected components, pixel count

For the classical method (refined from skeleton_to_graph_v3), measures:
  - Classical pipeline time
  - Classical skeleton stats

Outputs a CSV “metrics_summary.csv” under output_folder with columns:
filename,
[for each model] modelX_time_s, modelX_skel_time_s, modelX_endpoints, modelX_components, modelX_pixels,
classic_time_s, classic_endpoints, classic_components, classic_pixels

Finally prints FPS (images/sec) for each model (inference-only and inference+skel),
and for the classical pipeline.

Usage:
  - Edit the HARD-CODED section below to set folders and model paths.
  - Run: python evaluate_models_on_metu.py
"""

import os
import random
import time
import csv
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, convolve
from skimage.measure import label
import torch

# ==== HARDCODED SETTINGS ====

# Folder containing METU images (grayscale .png, values in [0,255])
INPUT_FOLDER = Path(r"D:\camerer_ml\datasets\METU_Concrete_Crack_Seg\BW_visible")

# List of exported TorchScript Skelite model files (full paths)
MODEL_PATHS = [
    r"D:\camerer_ml\adaptive_pca_3dreproject\Skelite\skelite_scripted.pt",
    r"D:\camerer_ml\adaptive_pca_3dreproject\Skelite\fine_tuned_outputs\concrete3k\exported_last\skelite_ft_cc3k_scripted.pt",
    r"D:\camerer_ml\adaptive_pca_3dreproject\Skelite\fine_tuned_outputs\crackseg9k\exported_last\skelite_ft_cs9k_scripted.pt",
    # Add more if needed
]

# Output folder (will contain overlays and metrics CSV)
OUTPUT_FOLDER = Path(r"D:\camerer_ml\adaptive_pca_3dreproject\Skelite\full_evaluation_outputs")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Number of images to sample
N_IMAGES = 100

# Seed for pseudo-random sampling
SAMPLE_SEED = 42

# Threshold for model output → binary mask
MODEL_THRESH = 0.6

# Maximum branch length for classical prune (if used)
CLASSIC_PRUNE_LEN = 50

# Device for model inference ("cuda" or "cpu")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# =============================


# ----- Classical pipeline functions (from skeleton_to_graph_v3.py) -----

MASK_THRESH = 0.5  # threshold to binarize grayscale → mask
BASE_LEN = 10
WIDTH_SCALE = 2.0
NORMAL_RADIUS = 10
MIN_MAIN_PATH_LEN = 40  # minimum length to keep a component

def load_binarize(image_path: Path, thresh: float = MASK_THRESH):
    """
    Load an image (RGB or grayscale), convert to float [0,1], threshold → binary mask.
    Returns (bin_mask_uint8, float_gray_image).
    """
    img = imageio.imread(str(image_path))
    if img.ndim == 3:
        img = img.mean(axis=2)
    img = img.astype(np.float32) / 255.0
    bin_mask = (img > thresh).astype(np.uint8)
    return bin_mask, img

def find_all_branches(skel: np.ndarray):
    """
    Identify all branches in a binary skeleton (0/1 uint8).
    A "branch" is a path from an endpoint (degree=1) inward until a junction.
    Returns a list of (endpoint, path_list) pairs.
    """
    H, W = skel.shape
    kernel = np.array([[1,1,1],[1,10,1],[1,1,1]], dtype=np.uint8)
    conv = convolve(skel.astype(np.uint8), kernel, mode='constant', cval=0)
    endpoints = np.argwhere(conv == 11)
    visited = set()
    branches = []
    for (y0, x0) in endpoints:
        ep = (int(y0), int(x0))
        if ep in visited:
            continue
        path = [ep]
        y, x = ep
        prev = None
        while True:
            visited.add((y, x))
            nbrs = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and skel[ny, nx]:
                        if (ny, nx) != prev:
                            nbrs.append((ny, nx))
            if len(nbrs) != 1:
                break
            nxt = nbrs[0]
            path.append(nxt)
            prev = (y, x)
            y, x = nxt
        for node in path:
            visited.add(node)
        branches.append((ep, path))
    return branches

def normal_valid(endpoint, path, mask, radius=NORMAL_RADIUS):
    """
    From an endpoint, compute local tangent then normal, sample up to radius on both sides:
      - If sample leaves bounds or hits mask, return True (valid)
      - Else return False (prune)
    """
    H, W = mask.shape
    if len(path) < 2:
        return False
    y0, x0 = endpoint
    y1, x1 = path[1]
    dy = y1 - y0
    dx = x1 - x0
    mag = np.hypot(dx, dy)
    if mag == 0:
        return False
    ny = -dx / mag
    nx = dy / mag
    for s in (-1, 1):
        for r in range(1, radius + 1):
            yy = int(round(y0 + s * ny * r))
            xx = int(round(x0 + s * nx * r))
            if not (0 <= yy < H and 0 <= xx < W):
                return True
            if mask[yy, xx] == 1:
                return True
    return False

def adaptive_prune(skel: np.ndarray, width_map: np.ndarray, mask: np.ndarray):
    """
    Prune short branches adaptively based on local width:
      dynamic_thresh = BASE_LEN + (width / WIDTH_SCALE).
    Returns pruned skeleton (0/1 uint8).
    """
    branches = find_all_branches(skel)
    pruned = skel.copy()
    for endpoint, path in branches:
        y_ep, x_ep = endpoint
        w = width_map[y_ep, x_ep]
        dynamic_thresh = BASE_LEN + (w / WIDTH_SCALE)
        if len(path) <= int(round(dynamic_thresh)):
            if not normal_valid(endpoint, path, mask, NORMAL_RADIUS):
                for (y, x) in path:
                    pruned[y, x] = 0
    return pruned

def keep_major_components(skel: np.ndarray, min_len=MIN_MAIN_PATH_LEN):
    """
    Keep only connected components of length >= min_len.
    """
    labels = label(skel, connectivity=2)
    kept = np.zeros_like(skel)
    for val in range(1, labels.max() + 1):
        comp = (labels == val)
        if comp.sum() >= min_len:
            kept |= comp
    return kept.astype(np.uint8)

# ----- End classical pipeline functions -----


def count_endpoints(skel: np.ndarray):
    """
    Count endpoints in a binary skeleton using the 3x3 kernel trick.
    """
    kernel = np.array([[1,1,1],[1,10,1],[1,1,1]], dtype=np.uint8)
    filtered = convolve(skel.astype(np.uint8), kernel, mode='constant', cval=0)
    return int(np.sum(filtered == 11))

def count_connected_components(skel: np.ndarray):
    """
    Count number of connected components using skimage.label.
    """
    return int(label(skel).max())

def evaluate():
    # 1) Gather and sample images
    all_images = sorted(INPUT_FOLDER.glob("*.png"))
    random.seed(SAMPLE_SEED)
    sampled = all_images.copy()
    random.shuffle(sampled)
    sampled = sampled[:N_IMAGES]

    # 2) Load scripted models
    models = []
    model_names = []
    for mpath in MODEL_PATHS:
        model = torch.jit.load(mpath, map_location=DEVICE)
        model.eval()
        models.append(model)
        model_names.append(Path(mpath).stem)

    # 3) Prepare metrics list
    metrics = []

    # 4) Loop over sampled images
    for img_path in sampled:
        row = {"filename": img_path.name}

        # Load image, to grayscale uint8
        img = imageio.imread(str(img_path))
        if img.ndim == 3:
            img_gray = img.mean(axis=2).astype(np.uint8)
        else:
            img_gray = img.astype(np.uint8)
        # Normalize to [0,1] float
        img_norm = img_gray.astype(np.float32) / 255.0

        # --- Classical pipeline ---
        t0 = time.time()
        bin_mask, _ = load_binarize(img_path, thresh=MASK_THRESH)
        raw_skel_cl = skeletonize(bin_mask.astype(bool)).astype(np.uint8)
        width_map = 2.0 * distance_transform_edt(bin_mask > 0)
        pruned_cl = adaptive_prune(raw_skel_cl, width_map, bin_mask)
        skel_cl_final = keep_major_components(pruned_cl, min_len=max(30, min(bin_mask.shape)//64))
        t1 = time.time()
        classic_time = t1 - t0
        row["classic_time_s"] = classic_time
        row["classic_endpoints"] = count_endpoints(skel_cl_final)
        row["classic_components"] = count_connected_components(skel_cl_final)
        row["classic_pixels"] = int(np.sum(skel_cl_final))

        # Optionally save overlay
        ov_cl = None
        # Uncomment to save overlays:
        # ov_cl = overlay_skel(skel_cl_final, img_norm, color=(0,255,0))
        # imageio.imwrite(str(OUTPUT_FOLDER / f"{img_path.stem}_classic_overlay.png"), (ov_cl).astype(np.uint8))

        # --- Neural models ---
        for idx, model in enumerate(models):
            mname = model_names[idx]
            # Model inference
            img_t = torch.tensor(img_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
            t2 = time.time()
            with torch.no_grad():
                out_mask, _ = model(img_t, z=None, no_iter=5)
            t3 = time.time()
            # out_mask: tensor [1,1,H,W], float [0,1]
            out_np = out_mask[0,0].cpu().numpy()
            bin_nn = (out_np > MODEL_THRESH).astype(np.uint8)

            model_time = t3 - t2
            row[f"{mname}_time_s"] = model_time

            # Skeletonize nn mask
            t4 = time.time()
            skel_nn_raw = skeletonize(bin_nn.astype(bool)).astype(np.uint8)
            skel_nn_final = prune_simple(skel_nn_raw, max_length=CLASSIC_PRUNE_LEN)
            t5 = time.time()

            nn_skel_time = t5 - t4
            row[f"{mname}_skel_time_s"] = model_time + nn_skel_time
            row[f"{mname}_endpoints"] = count_endpoints(skel_nn_final)
            row[f"{mname}_components"] = count_connected_components(skel_nn_final)
            row[f"{mname}_pixels"] = int(np.sum(skel_nn_final))

            # Optionally save overlay
            ov_nn = None
            # Uncomment to save overlays:
            # ov_nn = overlay_skel(skel_nn_final, img_norm, color=(255,0,0))
            # imageio.imwrite(str(OUTPUT_FOLDER / f"{img_path.stem}_{mname}_overlay.png"), (ov_nn).astype(np.uint8))

        metrics.append(row)
        print(f"Processed {img_path.name}")

    # 5) Write CSV
    csv_path = OUTPUT_FOLDER / "metrics_summary.csv"
    # Collect all fieldnames (columns)
    if metrics:
        fieldnames = list(metrics[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics)
        print(f"\nMetrics table saved to: {csv_path}")

    # 6) Compute and print FPS
    total_imgs = len(metrics)
    # Classic FPS
    total_classic_time = sum(r["classic_time_s"] for r in metrics)
    print(f"Classical pipeline: {total_imgs / total_classic_time:.2f} FPS (avg over {total_imgs} images)")

    for mname in model_names:
        total_model_time = sum(r[f"{mname}_time_s"] for r in metrics)
        total_model_skel_time = sum(r[f"{mname}_skel_time_s"] for r in metrics)
        print(f"{mname}:")
        print(f"  Inference-only FPS: {total_imgs / total_model_time:.2f}")
        print(f"  Inference+skeleton FPS: {total_imgs / total_model_skel_time:.2f}")

# Simple pruning (no adaptive logic), same as sanity_check_full_run.prune_branches
def prune_simple(skel: np.ndarray, max_length=CLASSIC_PRUNE_LEN):
    sk = skel.astype(np.uint8).copy()
    kernel = np.array([[1,1,1],[1,10,1],[1,1,1]], dtype=np.uint8)
    endpoints = np.argwhere(convolve(sk, kernel, mode='constant') == 11)
    for (y, x) in endpoints:
        path = [(y, x)]
        sk[y, x] = 0
        for _ in range(max_length):
            neighbors = []
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y+dy, x+dx
                    if 0 <= ny < sk.shape[0] and 0 <= nx < sk.shape[1] and sk[ny, nx]:
                        neighbors.append((ny, nx))
            if len(neighbors) != 1:
                break
            y, x = neighbors[0]
            path.append((y, x))
            sk[y, x] = 0
        if len(path) <= max_length:
            for (yy, xx) in path:
                sk[yy, xx] = 0
        else:
            for (yy, xx) in path:
                sk[yy, xx] = 1
    return sk.astype(bool)

if __name__ == "__main__":
    evaluate()
