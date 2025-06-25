#!/usr/bin/env python3
"""
skeleton_overlay.py

Automated skeletonization + branch‐pruning, producing a single “overlay” image:
  - Background: original binary mask (white on black).
  - Foreground: pruned skeleton colored by local width (1–64 pixels mapped to a colormap).

For each image in INPUT_DIR, the script will:
  1) Load & binarize (thresh = MASK_THRESH) → bin_mask.
  2) Skeletonize bin_mask, build graph, prune branches ≤ PRUNE_LENGTH → pruned_skel.
  3) Compute a per‐pixel width_map via distance_transform_edt.
  4) Clamp widths to [1, 64], map those to an RGB colormap.
  5) Create an overlay: bin_mask (gray) + colored skeleton (width‐coded).
  6) Save only the overlay under OUTPUT_DIR/overlay/<stem>_overlay.png.

Adjust INPUT_DIR, OUTPUT_DIR, MASK_THRESH, PRUNE_LENGTH at the top before running.

Requires:
    pip install numpy imageio scikit-image scipy networkx matplotlib
"""

import os
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, convolve
import networkx as nx
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# CONFIGURATION: adjust these paths / parameters as needed.
# ------------------------------------------------------------

#INPUT_DIR = Path(r"D:\camerer_ml\datasets\Final-Dataset-Vol1\labels_visible")
#INPUT_DIR = Path(r"D:\camerer_ml\datasets\concrete3k\concrete3k\labels_01_visible")
INPUT_DIR = Path(r"D:\camerer_ml\datasets\METU_Concrete_Crack_Seg\BW_visible")
#OUTPUT_DIR = Path(r"D:\camerer_ml\adaptive_pca_3dreproject\Skelite\training_data\concrete3k")
OUTPUT_DIR = Path(r"D:\camerer_ml\adaptive_pca_3dreproject\Skelite\training_data\METU")

# Folder where all intermediates and finals will be saved
#OUTPUT_DIR = Path(r"D:\camerer_ml\adaptive_pca_3dreproject\Skelite\training_data\crackseg9k")
#INPUT_DIR    = Path("/path/to/crack_images")
#OUTPUT_DIR   = Path("/path/to/save_outputs")
MASK_THRESH  = 0.5     # threshold for binarizing grayscale → mask
PRUNE_LENGTH = 15      # remove any skeleton‐branch of length ≤ PRUNE_LENGTH
# ------------------------------------------------------------


def load_and_binarize(image_path: Path, thresh: float = 0.5):
    """
    Load an image (RGB or grayscale), convert to float [0,1], then threshold → binary mask.
    Returns (binary_mask_uint8, float_image).
    """
    img = imageio.imread(str(image_path))
    if img.ndim == 3:
        img = img.mean(axis=2)  # convert RGB → grayscale
    img = img.astype(np.float32) / 255.0
    bin_mask = (img > thresh).astype(np.uint8)
    return bin_mask, img


def save_png(arr: np.ndarray, outpath: Path):
    """
    Save a 2D or 3D array to PNG.
    - If arr dtype is bool or uint8 (0/1), writes 0/255.
    - If arr dtype is float in [0,1], rescales to 0–255.
    - If arr is 3‐channel uint8, writes directly.
    """
    if arr.ndim == 2:
        if arr.dtype == np.bool_ or arr.dtype == np.uint8:
            imageio.imwrite(str(outpath), (arr.astype(np.uint8) * 255))
        else:
            imageio.imwrite(str(outpath), (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8))
    elif arr.ndim == 3 and arr.dtype == np.uint8:
        imageio.imwrite(str(outpath), arr)
    else:
        raise ValueError("Unsupported array format for save_png")


def skeleton_to_graph(skel: np.ndarray) -> nx.Graph:
    """
    Convert a 1‐pixel‐wide binary skeleton (2D uint8 with 0/1) into a graph:
      - Each ON pixel is a node (y,x).
      - Edges connect 8‐neighbors that are both ON.
    """
    G = nx.Graph()
    H, W = skel.shape
    ys, xs = np.nonzero(skel)
    for (y, x) in zip(ys, xs):
        G.add_node((y, x))
    for (y, x) in zip(ys, xs):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx_ < W and skel[ny, nx_]:
                    G.add_edge((y, x), (ny, nx_))
    return G


def prune_graph_branches(G: nx.Graph, max_length: int) -> set:
    """
    Given a skeleton graph, find all spur‐branches of length ≤ max_length and
    return a set of their pixel nodes to remove.
    """
    to_remove = set()
    endpoints = [n for n, deg in G.degree() if deg == 1]
    visited = set()

    for ep in endpoints:
        if ep in visited:
            continue
        path = [ep]
        current = ep
        prev = None

        while True:
            visited.add(current)
            neighbors = [nbr for nbr in G.neighbors(current) if nbr != prev]
            if len(neighbors) != 1:
                break
            nxt = neighbors[0]
            path.append(nxt)
            prev = current
            current = nxt
            visited.add(current)
            if G.degree(current) != 2:
                break

        if len(path) <= max_length:
            to_remove.update(path)

    return to_remove


def automated_skeleton_pruning(bin_mask: np.ndarray, prune_len: int):
    """
    1) Skeletonize bin_mask → raw_skel (0/1 uint8).
    2) Build graph, prune any branches of length ≤ prune_len.
    3) Return (raw_skel_uint8, pruned_skel_uint8).
    """
    raw_skel = skeletonize(bin_mask.astype(bool)).astype(np.uint8)
    G = skeleton_to_graph(raw_skel)
    to_remove = prune_graph_branches(G, max_length=prune_len)
    pruned = raw_skel.copy()
    for (y, x) in to_remove:
        pruned[y, x] = 0
    return raw_skel, pruned


def compute_width_map(bin_mask: np.ndarray, pruned_skel: np.ndarray):
    """
    Compute a per-pixel width map only at skeleton pixels:
      - dmap = distance_transform_edt(bin_mask > 0)
      - width_map[y,x] = 2 * dmap[y,x] for each pruned_skel pixel
    Returns a float32 array of same shape, zero everywhere except skeleton pixels.
    """
    dmap = distance_transform_edt(bin_mask > 0)
    width_map = np.zeros_like(dmap, dtype=np.float32)
    ys, xs = np.nonzero(pruned_skel)
    for (y, x) in zip(ys, xs):
        width_map[y, x] = 2.0 * dmap[y, x]
    return width_map


def create_overlay(bin_mask: np.ndarray, pruned_skel: np.ndarray, width_map: np.ndarray):
    """
    Create a 3‐channel uint8 overlay:
      - Background: bin_mask (white=255 where mask=1, black=0 where mask=0).
      - Foreground: for each pruned_skel pixel, take width_map[y,x], clamp to [1,64],
                    map to a color via matplotlib’s 'jet' colormap, override background.
    Returns an (H, W, 3) uint8 RGB array.
    """
    H, W = bin_mask.shape
    # Start with gray background: white for mask, black for background
    overlay = np.zeros((H, W, 3), dtype=np.uint8)
    overlay[bin_mask == 1] = 200  # light gray background

    # Prepare colormap: 'jet', with norm from 1→64
    cmap = plt.get_cmap('jet')
    norm = plt.Normalize(vmin=1.0, vmax=64.0)

    ys, xs = np.nonzero(pruned_skel)
    for (y, x) in zip(ys, xs):
        w = width_map[y, x]
        if w < 1.0:
            w = 1.0
        elif w > 64.0:
            w = 64.0
        rgba = cmap(norm(w))  # RGBA float in [0,1]
        rgb = [int(255 * c) for c in rgba[:3]]
        overlay[y, x] = rgb

    return overlay


def main():
    # 1) Check directories
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"INPUT_DIR does not exist: {INPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    overlay_dir = OUTPUT_DIR / "overlay"
    overlay_dir.mkdir(exist_ok=True)

    # 2) Process each image
    img_paths = sorted(INPUT_DIR.glob("*.*"))
    for img_path in img_paths:
        stem = img_path.stem
        print(f"Processing {stem}...")

        # Load & binarize
        bin_mask, float_img = load_and_binarize(img_path, thresh=MASK_THRESH)

        # Skeletonize & pruned skeleton
        raw_skel, pruned_skel = automated_skeleton_pruning(bin_mask, prune_len=PRUNE_LENGTH)

        # Compute width_map
        width_map = compute_width_map(bin_mask, pruned_skel)

        # Create overlay
        overlay_rgb = create_overlay(bin_mask, pruned_skel, width_map)

        # Save overlay
        out_path = overlay_dir / f"{stem}_overlay.png"
        save_png(overlay_rgb, out_path)
        print(f"  → Saved overlay: {out_path}")

    print("\nDone. Overlay images are in:", overlay_dir)


if __name__ == "__main__":
    main()


# Folder containing your crack images (grayscale or RGB)
IMAGE_DIR = Path(r"D:\camerer_ml\datasets\Final-Dataset-Vol1\labels_visible")

# Folder where all intermediates and finals will be saved
OUTPUT_DIR = Path(r"D:\camerer_ml\adaptive_pca_3dreproject\Skelite\training_data\crackseg9k")