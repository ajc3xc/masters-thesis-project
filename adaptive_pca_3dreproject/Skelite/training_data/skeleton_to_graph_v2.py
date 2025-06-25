#!/usr/bin/env python3
"""
skeleton_compare.py

For each large crack image (e.g., 4032×3024), this script produces two overlays:

  1) **Non‐stitched overlay**:
       - Run single-shot skeletonization + auto‐prune on the entire binary mask.
       - Compute a width map on that pruned skeleton.
       - Create an overlay of the original binary mask + color‐coded skeleton (by width).
       - Save as “overlay_nonstitched.png”.

  2) **Stitched overlay** (tile‐based):
       - Break the image into overlapping tiles (default 1024×1024 with 64 px overlap).
       - For each tile: binarize → skeletonize → auto‐prune → compute that tile’s width map.
       - Stitch all pruned tile skeletons and width maps back into full‐image canvases.
       - Create an overlay of the original binary mask + color‐coded stitched skeleton.
       - Save as “overlay_stitched.png”.

Finally, the script also creates a side‐by‐side comparison figure (non‐stitched vs. stitched)
and saves it as “overlay_comparison.png”.

Adjust the paths and parameters under **CONFIGURATION** before running.

Requirements:
    pip install numpy imageio scikit-image scipy networkx matplotlib
"""

import os
import csv
from pathlib import Path
from typing import Tuple

import numpy as np
import imageio.v2 as imageio
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, label
import networkx as nx
import matplotlib.pyplot as plt


# ---------------------------------------
# CONFIGURATION (edit these before running)
# ---------------------------------------

# Input folder containing large crack images (e.g., 4032×3024)
INPUT_DIR = Path(r"D:\camerer_ml\datasets\METU_Concrete_Crack_Seg\BW_visible")
#OUTPUT_DIR = Path(r"D:\camerer_ml\adaptive_pca_3dreproject\Skelite\training_data\concrete3k")
OUTPUT_DIR = Path(r"D:\camerer_ml\adaptive_pca_3dreproject\Skelite\training_data\METU_tiled")

# Threshold for binarizing grayscale → mask
MASK_THRESH  = 0.5

# Prune any skeleton spur ≤ this many pixels
PRUNE_LENGTH = 15

# Tile parameters for stitched overlay:
#   TILE_SIZE: the width/height of each tile (including margins)
#   MARGIN: number of pixels to overlap on each side
TILE_SIZE    = 1024
MARGIN       = 64

# Width‐map clamp: any width > WIDTH_CLAMP is mapped to the top of the colormap
WIDTH_CLAMP  = 64.0
# ---------------------------------------


def load_binarize(path: Path, thresh: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load an image (RGB or grayscale), convert to float [0,1], threshold → binary mask.
    Returns (bin_mask_uint8, float_gray_image).
    """
    img = imageio.imread(str(path))
    if img.ndim == 3:
        img = img.mean(axis=2)  # simple grayscale conversion
    img = img.astype(np.float32) / 255.0
    bin_mask = (img > thresh).astype(np.uint8)
    return bin_mask, img


def save_png(arr: np.ndarray, outpath: Path):
    """
    Save a 2D or 3D array to PNG:
      - If arr is bool or uint8 (0/1), writes 0/255.
      - If arr is float in [0,1], rescales to 0–255.
      - If arr is 3-channel uint8, writes directly.
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
    Convert a binary skeleton (0/1 uint8, one-pixel wide) into a NetworkX graph:
      - Node (y,x) for every white pixel.
      - Edges between 8-neighbors that are both white.
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


def prune_spurs(skel: np.ndarray, max_len: int) -> np.ndarray:
    """
    Remove any skeleton "spur" whose length ≤ max_len pixels.
    Returns a pruned 0/1 uint8 skeleton.
    """
    G = skeleton_to_graph(skel)
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
            nbrs = [nbr for nbr in G.neighbors(current) if nbr != prev]
            if len(nbrs) != 1:
                break
            nxt = nbrs[0]
            path.append(nxt)
            prev = current
            current = nxt
            visited.add(current)
            if G.degree(current) != 2:
                break

        if len(path) <= max_len:
            to_remove.update(path)

    pruned = skel.copy()
    for (y, x) in to_remove:
        pruned[y, x] = 0
    return pruned


def automated_skeleton_pruning(bin_mask: np.ndarray, prune_len: int) -> np.ndarray:
    """
    Single‐shot pipeline:
      1) skeletonize the entire bin_mask → raw_skel
      2) prune spurs ≤ prune_len → pruned_skel
      3) return pruned_skel (0/1 uint8)
    """
    raw_skel = skeletonize(bin_mask.astype(bool)).astype(np.uint8)
    pruned_skel = prune_spurs(raw_skel, max_len=prune_len)
    return pruned_skel


def compute_width_map(bin_mask: np.ndarray, pruned_skel: np.ndarray) -> np.ndarray:
    """
    Compute a width map for a given pruned skeleton and binary mask:
      - dmap = distance_transform_edt(bin_mask > 0)
      - width_map[y,x] = 2 * dmap[y,x] for each pruned_skel pixel
    Returns a float32 array of shape (H, W), zero everywhere except skeleton pixels.
    """
    dmap = distance_transform_edt(bin_mask > 0)
    width_map = np.zeros_like(dmap, dtype=np.float32)
    ys, xs = np.nonzero(pruned_skel)
    for (y, x) in zip(ys, xs):
        width_map[y, x] = 2.0 * dmap[y, x]
    return width_map


def create_overlay(bin_mask: np.ndarray, pruned_skel: np.ndarray, width_map: np.ndarray) -> np.ndarray:
    """
    Create a 3-channel uint8 overlay:
      - Background: set to light gray (200,200,200) where bin_mask=1, black (0,0,0) where bin_mask=0.
      - Foreground: for each pruned_skel pixel, take width_map[y,x], clamp to [1, WIDTH_CLAMP],
                    map to a color via matplotlib's 'jet' colormap, override background.
    Returns an (H, W, 3) uint8 RGB array.
    """
    H, W = bin_mask.shape
    overlay = np.zeros((H, W, 3), dtype=np.uint8)
    overlay[bin_mask == 1] = 200  # light gray

    cmap = plt.get_cmap('jet')
    norm = plt.Normalize(vmin=1.0, vmax=WIDTH_CLAMP)

    ys, xs = np.nonzero(pruned_skel)
    for (y, x) in zip(ys, xs):
        w = width_map[y, x]
        if w < 1.0:
            w = 1.0
        elif w > WIDTH_CLAMP:
            w = WIDTH_CLAMP
        rgba = cmap(norm(w))
        rgb = [int(255 * c) for c in rgba[:3]]
        overlay[y, x] = rgb

    return overlay


def tile_coords(full_h: int, full_w: int, tile_sz: int, margin: int):
    """
    Yield (y0, y1, x0, x1, iy0, iy1, ix0, ix1) for tiling a (full_h, full_w)
    image into tiles of size tile_sz×tile_sz with 'margin' overlap on all sides.

    - (y0, y1, x0, x1) describes the tile's full coordinates in the big image.
    - (iy0, iy1, ix0, ix1) describes the "interior" region of that tile, i.e.
      [y0+margin : y1-margin] × [x0+margin : x1-margin], which is what we copy
      into the final stitched output to avoid double-counting the overlap margins.
    """
    stride = tile_sz - 2 * margin
    ys = []
    y0 = 0
    while y0 < full_h:
        y1 = min(full_h, y0 + tile_sz)
        ys.append((y0, y1))
        if y1 == full_h:
            break
        y0 = y0 + stride

    xs = []
    x0 = 0
    while x0 < full_w:
        x1 = min(full_w, x0 + tile_sz)
        xs.append((x0, x1))
        if x1 == full_w:
            break
        x0 = x0 + stride

    for (y0, y1) in ys:
        for (x0, x1) in xs:
            iy0 = min(y0 + margin, y1)
            iy1 = max(y1 - margin, y0)
            ix0 = min(x0 + margin, x1)
            ix1 = max(x1 - margin, x0)
            yield y0, y1, x0, x1, iy0, iy1, ix0, ix1


def process_tile(bin_tile: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a binary tile (0/1 uint8), run:
      - raw_skel = skeletonize(tile)
      - pruned_skel = prune_spurs(raw_skel, PRUNE_LENGTH)
      - width_tile = compute width_map(tile, pruned_skel)

    Returns (pruned_skel, width_tile) for that tile.
    """
    raw_skel = skeletonize(bin_tile.astype(bool)).astype(np.uint8)
    pruned_skel = prune_spurs(raw_skel, max_len=PRUNE_LENGTH)

    dmap = distance_transform_edt(bin_tile > 0)
    width_tile = np.zeros_like(dmap, dtype=np.float32)
    ys, xs = np.nonzero(pruned_skel)
    for (y, x) in zip(ys, xs):
        width_tile[y, x] = 2.0 * dmap[y, x]

    return pruned_skel, width_tile


def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"INPUT_DIR does not exist: {INPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(INPUT_DIR.glob("*.*")):
        stem = img_path.stem
        print(f"\n=== Processing {stem} ===")

        # Create subfolder for this image
        out_sub = OUTPUT_DIR / stem
        out_sub.mkdir(exist_ok=True)

        # 1) Load & binarize full‐image
        bin_full, float_full = load_binarize(img_path, thresh=MASK_THRESH)
        H, W = bin_full.shape

        #
        # --- PART A: NON-STITCHED (single-shot) overlay ---
        #
        print("  • Running single-shot (non-stitched) skeletonization ...")
        pruned_full = automated_skeleton_pruning(bin_full, prune_len=PRUNE_LENGTH)
        width_full = compute_width_map(bin_full, pruned_full)
        overlay_non = create_overlay(bin_full, pruned_full, width_full)

        non_path = out_sub / "overlay_nonstitched.png"
        save_png(overlay_non, non_path)
        print(f"    → Saved non-stitched overlay: {non_path}")

        #
        # --- PART B: STITCHED (tile-based) overlay ---
        #
        print("  • Running tile-based (stitched) skeletonization ...")
        # Prepare empty canvases
        full_pruned_skel = np.zeros((H, W), dtype=np.uint8)
        full_width_map   = np.zeros((H, W), dtype=np.float32)
        stats_rows = [("tile_y0", "tile_x0", "n_skel_px", "n_components")]

        for (y0, y1, x0, x1, iy0, iy1, ix0, ix1) in tile_coords(H, W, TILE_SIZE, MARGIN):
            tile_bin = bin_full[y0:y1, x0:x1]
            pruned_tile, width_tile = process_tile(tile_bin)

            # Count skeleton pixels & connected components
            n_skel_px = int(pruned_tile.sum())
            comp_map, n_comp = label(pruned_tile, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
            stats_rows.append((y0, x0, n_skel_px, int(n_comp)))

            # Paste the interior region into full canvases
            tile_iy0 = iy0 - y0
            tile_iy1 = iy1 - y0
            tile_ix0 = ix0 - x0
            tile_ix1 = ix1 - x0

            full_pruned_skel[iy0:iy1, ix0:ix1] = pruned_tile[tile_iy0:tile_iy1, tile_ix0:tile_ix1]
            full_width_map[iy0:iy1, ix0:ix1] = width_tile[tile_iy0:tile_iy1, tile_ix0:tile_ix1]

            # (Optional) Save per-tile outputs for debugging
            tile_dir = out_sub / "tiles"
            tile_dir.mkdir(exist_ok=True)
            save_png(tile_bin,    tile_dir / f"tile_{y0}_{x0}_bin.png")
            save_png(pruned_tile, tile_dir / f"tile_{y0}_{x0}_skel.png")
            wnorm = (width_tile / width_tile.max()) if width_tile.max() > 0 else width_tile
            save_png(wnorm.astype(np.float32), tile_dir / f"tile_{y0}_{x0}_wid.png")

        # Save tile stats CSV
        stats_path = out_sub / "tile_stats.csv"
        with open(stats_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(stats_rows)
        print(f"    → Saved tile stats: {stats_path}")

        # Create stitched overlay
        overlay_stitched = np.zeros((H, W, 3), dtype=np.uint8)
        overlay_stitched[bin_full == 1] = 200  # light gray background
        cmap = plt.get_cmap('jet')
        norm = plt.Normalize(vmin=1.0, vmax=WIDTH_CLAMP)
        ys, xs = np.nonzero(full_pruned_skel)
        for (y, x) in zip(ys, xs):
            w = full_width_map[y, x]
            if w < 1.0:
                w = 1.0
            elif w > WIDTH_CLAMP:
                w = WIDTH_CLAMP
            rgba = cmap(norm(w))
            rgb = [int(255 * c) for c in rgba[:3]]
            overlay_stitched[y, x] = rgb

        stitched_path = out_sub / "overlay_stitched.png"
        save_png(overlay_stitched, stitched_path)
        print(f"    → Saved stitched overlay: {stitched_path}")

        #
        # --- PART C: COMPARISON FIGURE ---
        #
        print("  • Creating side-by-side comparison ...")
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].imshow(overlay_non)
        axes[0].set_title("Non-stitched Overlay (Single-Shot)")
        axes[0].axis('off')
        axes[1].imshow(overlay_stitched)
        axes[1].set_title("Stitched Overlay (Tiled)")
        axes[1].axis('off')
        fig.tight_layout()
        comp_path = out_sub / "overlay_comparison.png"
        fig.savefig(str(comp_path), dpi=150)
        plt.close(fig)
        print(f"    → Saved comparison figure: {comp_path}")

    print("\nAll images processed. Check OUTPUT_DIR for results.")


if __name__ == "__main__":
    main()
