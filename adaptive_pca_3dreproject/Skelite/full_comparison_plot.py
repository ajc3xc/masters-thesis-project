#!/usr/bin/env python3
"""
evaluate_and_plot_from_scratch.py

Runs a fresh evaluation of classical vs. exported Skelite models on 100 pseudo-random METU images,
computes metrics (FPS, endpoints, components, pixel count), prints a concise summary, and saves overlay plots.

For each of N=100 images (seed 42):
  - Classical pipeline:
      • Binarize → skeletonize → adaptive prune → keep_major_components
      • Record time, endpoints, components, pixels
  - Each TorchScript model:
      • Infer mask → threshold → skeletonize → simple prune
      • Record inference time alone and combined (inference+skel), endpoints, components, pixels

At the end:
  - Prints one-line summary for classical and each model: FPS, avg endpoints, avg components, avg pixels
  - Saves for each image:
      1) IMAGE_comparison.png   – subplots: [Classical | Model1 | Model2 | …], skeleton (blue) over original mask
      2) IMAGE_neural_only.png  – subplots: [Model1 | Model2 | …], skeleton (blue) over original mask

Usage:
  - Edit the HARD‐CODED SETTINGS below
  - Run: python evaluate_and_plot_from_scratch.py
"""

import os
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import torch
import csv

from pathlib import Path
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, convolve
from skimage.measure import label

# ===================== HARD‐CODED SETTINGS =====================

# Folder containing METU images (grayscale .png or .jpg, values 0–255)
DATA_ROOT = Path(r"D:\camerer_ml\datasets\METU_Concrete_Crack_Seg\BW_visible")

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

# Name of the CSV to write
CSV_NAME = OUTPUT_FOLDER / "metrics_summary.csv"

# Number of images to sample
N_IMAGES = 100
SAMPLE_SEED = 42

# Threshold for converting model output to binary
MODEL_THRESH = 0.6

# Device for model inference
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Classical-level parameters
MASK_THRESH = 0.5
BASE_LEN = 10
WIDTH_SCALE = 2.0
NORMAL_RADIUS = 10
MIN_MAIN_PATH_LEN = 40

# Simple prune max branch length
SIMPLE_PRUNE_LEN = 50

# ================================================================


def load_binarize(image_path, thresh=MASK_THRESH):
    """
    Load an image (RGB or grayscale), convert to float [0,1], threshold → binary mask.
    Returns (bin_mask_uint8, norm_float_gray).
    """
    img = imageio.imread(str(image_path)).astype(np.float32)
    if img.ndim == 3:
        img = img.mean(axis=2)
    img = img / 255.0
    bin_mask = (img > thresh).astype(np.uint8)
    return bin_mask, img


def find_all_branches(skel):
    """
    Identify skeleton branches. Returns list of (endpoint, path_list).
    """
    H, W = skel.shape
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.uint8)
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
    From endpoint, compute normal and march radius pixels:
      - If leave bounds or hit mask=1, return True; else False.
    """
    H, W = mask.shape
    if len(path) < 2:
        return False
    y0, x0 = endpoint
    y1, x1 = path[1]
    dy, dx = y1 - y0, x1 - x0
    mag = np.hypot(dx, dy)
    if mag == 0:
        return False
    ny, nx = -dx / mag, dy / mag
    for s in (-1, 1):
        for r in range(1, radius + 1):
            yy = int(round(y0 + s * ny * r))
            xx = int(round(x0 + s * nx * r))
            if not (0 <= yy < H and 0 <= xx < W):
                return True
            if mask[yy, xx] == 1:
                return True
    return False


def adaptive_prune(skel, width_map, mask):
    """
    Prune short branches adaptively: dynamic_thresh = BASE_LEN + (width/WIDTH_SCALE).
    """
    pruned = skel.copy()
    for ep, path in find_all_branches(skel):
        y_ep, x_ep = ep
        w = width_map[y_ep, x_ep]
        thresh = BASE_LEN + (w / WIDTH_SCALE)
        if len(path) <= int(round(thresh)):
            if not normal_valid(ep, path, mask):
                for (y, x) in path:
                    pruned[y, x] = 0
    return pruned


def keep_major_components(skel, min_len=MIN_MAIN_PATH_LEN):
    """
    Keep only connected components with size >= min_len.
    """
    labels = label(skel, connectivity=2)
    kept = np.zeros_like(skel)
    for val in range(1, labels.max() + 1):
        comp = (labels == val)
        if comp.sum() >= min_len:
            kept |= comp
    return kept.astype(np.uint8)


def prune_simple(skel, max_length=SIMPLE_PRUNE_LEN):
    """
    Remove branches of length <= max_length, keep the rest.
    """
    sk = skel.astype(np.uint8).copy()
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.uint8)
    endpoints = np.argwhere(convolve(sk, kernel, mode='constant') == 11)
    for (y, x) in endpoints:
        path = [(y, x)]
        sk[y, x] = 0
        for _ in range(max_length):
            nbrs = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < sk.shape[0] and 0 <= nx < sk.shape[1] and sk[ny, nx]:
                        nbrs.append((ny, nx))
            if len(nbrs) != 1:
                break
            y, x = nbrs[0]
            path.append((y, x))
            sk[y, x] = 0
        if len(path) > max_length:
            for (yy, xx) in path:
                sk[yy, xx] = 1
    return sk.astype(bool)


def count_endpoints(skel):
    """
    Count number of endpoints via convolution.
    """
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.uint8)
    conv = convolve(skel.astype(np.uint8), kernel, mode='constant', cval=0)
    return int(np.sum(conv == 11))


def count_components(skel):
    """
    Number of connected components.
    """
    return int(label(skel).max())


def main():
    # 1) Sample 100 images
    all_imgs = sorted(DATA_ROOT.glob("*.png"))
    random.seed(SAMPLE_SEED)
    random.shuffle(all_imgs)
    sampled = all_imgs[:N_IMAGES]

    # 2) Load TorchScript models
    models = []
    model_names = []
    for mp in MODEL_PATHS:
        mdl = torch.jit.load(str(mp), map_location=DEVICE)
        mdl.eval()
        models.append(mdl)
        model_names.append(Path(mp).stem)

    # 3) Prepare accumulators for summary & CSV rows
    total_classic_time = 0.0
    total_classic_end = 0
    total_classic_comp = 0
    total_classic_pix = 0

    total_model_time = {name: 0.0 for name in model_names}
    total_model_skel_time = {name: 0.0 for name in model_names}
    total_model_end = {name: 0 for name in model_names}
    total_model_comp = {name: 0 for name in model_names}
    total_model_pix = {name: 0 for name in model_names}

    csv_rows = []

    # 4) Loop over each sampled image
    for img_path in sampled:
        img_name = img_path.name

        # Load original (0–255) and normalize to [0,1]
        img = imageio.imread(str(img_path))
        if img.ndim == 3:
            img_gray = img.mean(axis=2).astype(np.uint8)
        else:
            img_gray = img.astype(np.uint8)
        mask_norm = img_gray.astype(np.float32) / 255.0

        row = {"filename": img_name}

        # ---- Classical pipeline ----
        t0 = time.time()
        bin_mask, _ = load_binarize(img_path, thresh=MASK_THRESH)
        raw_skel_cl = skeletonize(bin_mask.astype(bool)).astype(np.uint8)
        width_map = 2.0 * distance_transform_edt(bin_mask > 0)
        pruned_cl = adaptive_prune(raw_skel_cl, width_map, bin_mask)
        skel_cl = keep_major_components(pruned_cl, min_len=max(30, min(bin_mask.shape) // 64))
        t1 = time.time()

        classic_time = t1 - t0
        total_classic_time += classic_time
        e_cl = count_endpoints(skel_cl)
        c_cl = count_components(skel_cl)
        p_cl = int(np.sum(skel_cl))

        total_classic_end += e_cl
        total_classic_comp += c_cl
        total_classic_pix += p_cl

        row["classic_time_s"] = classic_time
        row["classic_endpoints"] = e_cl
        row["classic_components"] = c_cl
        row["classic_pixels"] = p_cl

        # ---- Neural models ----
        skels_nn = {}
        for mdl, mname in zip(models, model_names):
            # Inference only
            img_t = torch.tensor(mask_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
            t2 = time.time()
            with torch.no_grad():
                out_mask, _ = mdl(img_t, z=None, no_iter=5)
            t3 = time.time()

            model_time = t3 - t2
            total_model_time[mname] += model_time

            out_np = out_mask[0, 0].cpu().numpy()
            bin_nn = (out_np > MODEL_THRESH).astype(np.uint8)

            # Skeletonize + simple prune
            t4 = time.time()
            raw_skel_nn = skeletonize(bin_nn.astype(bool)).astype(np.uint8)
            skel_nn = prune_simple(raw_skel_nn, max_length=SIMPLE_PRUNE_LEN)
            t5 = time.time()

            skel_time = t5 - t4
            total_model_skel_time[mname] += (model_time + skel_time)
            e_nn = count_endpoints(skel_nn)
            c_nn = count_components(skel_nn)
            p_nn = int(np.sum(skel_nn))

            total_model_end[mname] += e_nn
            total_model_comp[mname] += c_nn
            total_model_pix[mname] += p_nn

            row[f"{mname}_time_s"] = model_time
            row[f"{mname}_skel_time_s"] = model_time + skel_time
            row[f"{mname}_endpoints"] = e_nn
            row[f"{mname}_components"] = c_nn
            row[f"{mname}_pixels"] = p_nn

            skels_nn[mname] = skel_nn

        csv_rows.append(row)

        # ---- Plot overlays for this image ----
        n_methods = 1 + len(models)
        fig, axes = plt.subplots(1, n_methods, figsize=(4 * n_methods, 4))

        # (a) Classical
        ax0 = axes[0]
        ax0.imshow(mask_norm, cmap="gray", vmin=0, vmax=1)
        Hc, Wc = skel_cl.shape
        overlay_cl = np.zeros((Hc, Wc, 4), dtype=np.float32)
        overlay_cl[skel_cl == 1] = [0.0, 0.0, 1.0, 0.6]  # blue, alpha=0.6
        ax0.imshow(overlay_cl)
        ax0.set_title("Classical")
        ax0.axis("off")

        # (b) Neural models
        for i, mname in enumerate(model_names, start=1):
            ax = axes[i]
            ax.imshow(mask_norm, cmap="gray", vmin=0, vmax=1)
            skel_nn = skels_nn[mname]
            Hn, Wn = skel_nn.shape
            overlay_nn = np.zeros((Hn, Wn, 4), dtype=np.float32)
            overlay_nn[skel_nn == 1] = [0.0, 0.0, 1.0, 0.6]
            ax.imshow(overlay_nn)
            ax.set_title(mname)
            ax.axis("off")

        plt.tight_layout()
        comp_out = OUTPUT_FOLDER / f"{img_name}_comparison.png"
        plt.savefig(str(comp_out), dpi=1000)
        plt.close(fig)

        # Neural-only plot
        fig2, axes2 = plt.subplots(1, len(models), figsize=(4 * len(models), 4))
        for idx, mname in enumerate(model_names):
            ax = axes2[idx]
            ax.imshow(mask_norm, cmap="gray", vmin=0, vmax=1)
            skel_nn = skels_nn[mname]
            Hn2, Wn2 = skel_nn.shape
            overlay_nn = np.zeros((Hn2, Wn2, 4), dtype=np.float32)
            overlay_nn[skel_nn == 1] = [0.0, 0.0, 1.0, 0.6]
            ax.imshow(overlay_nn)
            ax.set_title(mname)
            ax.axis("off")
        plt.tight_layout()
        nn_out = OUTPUT_FOLDER / f"{img_name}_neural_only.png"
        plt.savefig(str(nn_out), dpi=1000)
        plt.close(fig2)

        print(f"Processed and plotted {img_name}")

    # 5) Write CSV
    # Build the header (field names)
    fieldnames = ["filename",
                  "classic_time_s", "classic_endpoints", "classic_components", "classic_pixels"]
    for mname in model_names:
        fieldnames += [
            f"{mname}_time_s",
            f"{mname}_skel_time_s",
            f"{mname}_endpoints",
            f"{mname}_components",
            f"{mname}_pixels"
        ]

    with open(CSV_NAME, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
        f.flush()
    print(f"\nMetrics CSV written to: {CSV_NAME}")

    # 6) Compute and print one-line summary
    n = N_IMAGES
    fps_classic = n / total_classic_time
    avg_end_classic = total_classic_end / n
    avg_comp_classic = total_classic_comp / n
    avg_pix_classic = total_classic_pix / n

    print("\n===== SUMMARY =====")
    print(f"Dataset: {n} images")
    print(
        f"CLASSICAL -> FPS: {fps_classic:.2f}, "
        f"Endpoints: {avg_end_classic:.1f}, "
        f"Components: {avg_comp_classic:.1f}, "
        f"Pixels: {avg_pix_classic:.1f}"
    )

    for mname in model_names:
        fps_inf = n / total_model_time[mname]
        fps_inf_skel = n / total_model_skel_time[mname]
        avg_end = total_model_end[mname] / n
        avg_comp = total_model_comp[mname] / n
        avg_pix = total_model_pix[mname] / n
        print(
            f"{mname.upper():<15} -> Inference FPS: {fps_inf:.2f}, "
            f"Inf+Skel FPS: {fps_inf_skel:.2f}, "
            f"Endpoints: {avg_end:.1f}, "
            f"Components: {avg_comp:.1f}, "
            f"Pixels: {avg_pix:.1f}"
        )


if __name__ == "__main__":
    main()