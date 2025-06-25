#!/usr/bin/env python3
"""
adaptive_prune_test.py

Adaptive, width-aware skeleton pruning for wide vs. thin cracks.
For each image in INPUT_DIR:
  1) Load & binarize (thresh = MASK_THRESH) → bin_mask.
  2) Skeletonize bin_mask → raw_skel (0/1 uint8).
  3) Compute width_map from distance_transform on bin_mask.
  4) Find every skeleton branch from its endpoint to the nearest junction.
  5) For each branch:
       a) Look up local width at the endpoint (width_map[y, x]).
       b) Compute dynamic length threshold:
            dynamic_thresh = BASE_LEN + (width / WIDTH_SCALE).
       c) If branch_length ≤ dynamic_thresh AND the “normal test” fails 
          (i.e. marching out along the normal never hits mask=1 nor leaves image),
          then prune that entire branch.
  6) Save the resulting “adaptive_pruned_skel.png”.
  7) Print CSV‐style metrics:
       image, raw_pixels, pruned_pixels, num_branches, num_pruned_branches,
       time_skeletonize, time_prune

Adjust paths & parameters in CONFIGURATION before running.

Requires:
    pip install numpy imageio scikit-image scipy matplotlib
"""

import time
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, convolve
import math

# ---------------------------------------
# CONFIGURATION (edit before running)
# ---------------------------------------

# Folder containing crack images (any extension: .png, .jpg, etc.)
INPUT_DIR = Path(r"D:\camerer_ml\datasets\METU_Concrete_Crack_Seg\BW_visible")
#OUTPUT_DIR = Path(r"D:\camerer_ml\adaptive_pca_3dreproject\Skelite\training_data\concrete3k")
OUTPUT_DIR = Path(r"D:\camerer_ml\adaptive_pca_3dreproject\Skelite\training_data\METU_fine_pruning")

# Threshold to binarize grayscale → mask
MASK_THRESH = 0.5

# Base length (in px) for dynamic threshold
BASE_LEN = 10

# Scale factor: width (in px) divided by this is added to BASE_LEN
WIDTH_SCALE = 2.0

# How far (in px) to march along each normal for the “normal test”
NORMAL_RADIUS = 10
# ---------------------------------------

import matplotlib.pyplot as plt

def save_overlay(mask, skeleton, outpath):
    """
    Save an overlay PNG:
      - Background: mask (gray)
      - Skeleton: colored (red or blue)
    """
    # Normalize mask to [0,1] for display
    norm_mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6)
    rgb = np.zeros(mask.shape + (3,), dtype=np.float32)
    # Light gray for mask (or tune as needed)
    rgb[..., :] = 1
    rgb[..., 0] = rgb[..., 1] = rgb[..., 2] = norm_mask * 0.7 + 0.3

    # Draw skeleton in blue
    skel_yx = np.nonzero(skeleton)
    rgb[skel_yx[0], skel_yx[1], :] = [0.1, 0.5, 1.0]

    # Save as uint8 PNG
    imageio.imwrite(str(outpath), (rgb * 255).astype(np.uint8))

def load_binarize(image_path: Path, thresh: float):
    """
    Load an image (RGB or grayscale), convert to float [0,1], threshold → binary mask.
    Returns (bin_mask_uint8, float_gray_image).
    """
    img = imageio.imread(str(image_path))
    if img.ndim == 3:
        img = img.mean(axis=2)
    img = img.astype(np.float32) / 255.0
    bin_mask = (img < thresh).astype(np.uint8)
    return bin_mask, img

def load_binarize(image_path: Path, thresh: float):
    """
    Load an image (RGB or grayscale), convert to float [0,1], threshold → binary mask.
    Inverts the image so black becomes white and vice versa.
    Returns (bin_mask_uint8, float_gray_image).
    """
    img = imageio.imread(str(image_path))
    if img.ndim == 3:
        img = img.mean(axis=2)
    img = img.astype(np.float32) / 255.0
    # Invert: white <-> black for both mask and grayscale
    img_inv = 1.0 - img
    bin_mask = (img_inv > thresh).astype(np.uint8)
    return bin_mask, img_inv


def save_png(arr: np.ndarray, outpath: Path):
    """
    Save a 2D or 3D array to PNG:
      - If arr is bool or uint8 (0/1), writes 0/255.
      - If arr is float in [0,1], rescales to 0–255.
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

def find_all_branches(skel: np.ndarray):
    """
    Identify all branches in a binary skeleton (0/1 uint8).
    A “branch” is a path from an endpoint (degree=1) inward until a junction 
    (pixel with degree != 2). Returns a list of (endpoint, path_list) pairs.
    """
    H, W = skel.shape
    # Kernel that yields 11 at endpoints: 10 (center) + 1 neighbor
    kernel = np.array([[1,1,1],
                       [1,10,1],
                       [1,1,1]], dtype=np.uint8)
    conv = convolve(skel.astype(np.uint8), kernel, mode='constant', cval=0)
    endpoints = np.argwhere(conv == 11)  # each endpoint has exactly one neighbor

    visited = set()
    branches = []

    for (y0, x0) in endpoints:
        ep = (int(y0), int(x0))
        if ep in visited:
            continue

        path = [ep]
        y, x = ep
        prev = None

        # Walk inward until hitting a junction or dead‐end
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
            # If not exactly one neighbor, we’re at a junction or isolated node
            if len(nbrs) != 1:
                break
            nxt = nbrs[0]
            path.append(nxt)
            prev = (y, x)
            y, x = nxt

        # Mark all pixels on this path as visited
        for node in path:
            visited.add(node)
        branches.append((ep, path))

    return branches

def normal_valid(endpoint, path, mask, radius):
    """
    From an endpoint, compute its local tangent (using the next point in 'path'),
    then form a unit normal vector. March up to 'radius' pixels on both sides:
      - If any sample point leaves image bounds OR lands on mask=1, return True (valid).
      - Otherwise, return False (spurious).
    """
    H, W = mask.shape
    if len(path) < 2:
        return False

    y0, x0 = endpoint
    y1, x1 = path[1]
    dy = y1 - y0
    dx = x1 - x0
    mag = math.hypot(dx, dy)
    if mag == 0:
        return False

    # Normal vector to (dx, dy) is (-dy/mag, dx/mag), but swap sign to get both directions
    ny = -dx / mag
    nx = dy / mag

    for s in (-1, 1):
        for r in range(1, radius + 1):
            yy = int(round(y0 + s * ny * r))
            xx = int(round(x0 + s * nx * r))
            # If we step outside the image, treat as valid (the branch points outward)
            if not (0 <= yy < H and 0 <= xx < W):
                return True
            # If we hit the crack mask, valid
            if mask[yy, xx] == 1:
                return True

    # If we never left bounds or hit mask, that branch end is floating in background
    return False

def adaptive_prune(skel: np.ndarray, width_map: np.ndarray, mask: np.ndarray):
    """
    Prune short branches adaptively based on local width:
      dynamic_thresh = BASE_LEN + (width_at_endpoint / WIDTH_SCALE).

    For each skeleton branch:
      - Let L = branch_length, w = width_map[endpoint].
      - Compute T = BASE_LEN + (w / WIDTH_SCALE).
      - If L ≤ T AND normal_valid(endpoint, path, mask, NORMAL_RADIUS) == False,
        then prune that entire branch.

    Returns:
      pruned_skel (0/1 uint8),
      raw_pixel_count,
      pruned_pixel_count,
      num_total_branches,
      num_pruned_branches.
    """
    branches = find_all_branches(skel)
    pruned = skel.copy()
    raw_pixels = int(pruned.sum())
    pruned_pixels = 0

    num_branches = len(branches)
    num_pruned = 0

    for endpoint, path in branches:
        y_ep, x_ep = endpoint
        w = width_map[y_ep, x_ep]
        dynamic_thresh = BASE_LEN + (w / WIDTH_SCALE)

        if len(path) <= int(round(dynamic_thresh)):
            if not normal_valid(endpoint, path, mask, NORMAL_RADIUS):
                num_pruned += 1
                for (y, x) in path:
                    if pruned[y, x]:
                        pruned[y, x] = 0
                        pruned_pixels += 1

    return pruned, raw_pixels, pruned_pixels, num_branches, num_pruned

from skimage.measure import label

def aggressive_length_prune(skel: np.ndarray, width_map: np.ndarray, base_len=2, width_scale=1.5):
    """
    Prune branches based on length and local width, with no normal test.
    """
    branches = find_all_branches(skel)
    pruned = skel.copy()
    pruned_pixels = 0
    num_pruned = 0

    for endpoint, path in branches:
        y_ep, x_ep = endpoint
        w = width_map[y_ep, x_ep]
        dynamic_thresh = base_len + (w / width_scale)
        if len(path) <= int(round(dynamic_thresh)):
            num_pruned += 1
            for (y, x) in path:
                if pruned[y, x]:
                    pruned[y, x] = 0
                    pruned_pixels += 1
    return pruned, pruned_pixels, len(branches), num_pruned

def keep_largest_component(skel):
    """
    Keeps only the largest connected skeleton component.
    """
    labels = label(skel, connectivity=2)
    if labels.max() == 0:
        return skel  # nothing to keep
    largest_label = 1 + np.argmax(np.bincount(labels.flat)[1:])
    return (labels == largest_label).astype(np.uint8)

MIN_MAIN_PATH_LEN = 40  # or 50, adjust to taste

def keep_major_components(skel, min_len=MIN_MAIN_PATH_LEN):
    labels = label(skel, connectivity=2)
    kept = np.zeros_like(skel)
    for val in range(1, labels.max()+1):
        comp = (labels == val)
        if comp.sum() >= min_len:
            kept |= comp
    return kept.astype(np.uint8)

'''# === Replace the main loop with this: ===
def endpoint_surrounded_by_white(endpoint, mask, check_radius=6, frac_thresh=0.85):
    """Check if endpoint is surrounded by mask=1 in a (2*radius+1)x(2*radius+1) window."""
    y, x = endpoint
    H, W = mask.shape
    ymin = max(y - check_radius, 0)
    ymax = min(y + check_radius + 1, H)
    xmin = max(x - check_radius, 0)
    xmax = min(x + check_radius + 1, W)
    window = mask[ymin:ymax, xmin:xmax]
    frac_white = window.mean()
    return frac_white >= frac_thresh

def length_and_surrounded_prune(skel: np.ndarray, width_map: np.ndarray, mask: np.ndarray, 
                                base_len=2, width_scale=1.5, check_radius=6, frac_thresh=0.85):
    """
    Prune branches based on (a) dynamic length threshold, and (b) "surrounded by white" test.
    Prune if:
      - branch is short (as before), OR
      - endpoint is surrounded by white (new test)
    """
    branches = find_all_branches(skel)
    pruned = skel.copy()
    pruned_pixels = 0
    num_pruned = 0

    for endpoint, path in branches:
        y_ep, x_ep = endpoint
        w = width_map[y_ep, x_ep]
        dynamic_thresh = base_len + (w / width_scale)
        # New: check either short or surrounded by white
        if len(path) <= int(round(dynamic_thresh)) or endpoint_surrounded_by_white(
                endpoint, mask, check_radius, frac_thresh):
            num_pruned += 1
            for (y, x) in path:
                if pruned[y, x]:
                    pruned[y, x] = 0
                    pruned_pixels += 1
    return pruned, pruned_pixels, len(branches), num_pruned

def hybrid_prune(skel, width_map, mask,
                 base_len=10,
                 moderate_len_factor=0.02,  # 2% of min(image size) for moderate branch
                 check_radius=6,
                 frac_thresh=0.85):
    """
    - Always prune branches ≤ base_len (tiny, regardless of where they are).
    - Prune moderate-length branches only if endpoint is surrounded by white.
    - Never prune branches longer than moderate threshold.
    """
    from skimage.measure import label
    H, W = mask.shape
    moderate_len = max(int(moderate_len_factor * min(H, W)), base_len+1)

    branches = find_all_branches(skel)
    pruned = skel.copy()
    pruned_pixels = 0
    num_pruned = 0

    for endpoint, path in branches:
        branch_len = len(path)
        if branch_len <= base_len:
            # Always prune very short branches
            num_pruned += 1
            for (y, x) in path:
                if pruned[y, x]:
                    pruned[y, x] = 0
                    pruned_pixels += 1
        elif branch_len <= moderate_len:
            # Prune moderate-length only if endpoint is embedded
            if endpoint_surrounded_by_white(endpoint, mask, check_radius, frac_thresh):
                num_pruned += 1
                for (y, x) in path:
                    if pruned[y, x]:
                        pruned[y, x] = 0
                        pruned_pixels += 1
        # else: leave all longer branches alone!

    return pruned, pruned_pixels, len(branches), num_pruned

# (Same endpoint_surrounded_by_white as before)
def endpoint_surrounded_by_white(endpoint, mask, check_radius=6, frac_thresh=0.85):
    y, x = endpoint
    H, W = mask.shape
    ymin = max(y - check_radius, 0)
    ymax = min(y + check_radius + 1, H)
    xmin = max(x - check_radius, 0)
    xmax = min(x + check_radius + 1, W)
    window = mask[ymin:ymax, xmin:xmax]
    frac_white = window.mean()
    return frac_white >= frac_thresh'''

#MAX_PRUNE_LEN = 250  # never prune branches longer than this, adjust for your scale
#MIN_MAIN_PATH_LEN = 50  # for keep_major_components()

'''def endpoint_surrounded_by_white(endpoint, mask, check_radius=6, frac_thresh=0.85):
    y, x = endpoint
    H, W = mask.shape
    ymin = max(y - check_radius, 0)
    ymax = min(y + check_radius + 1, H)
    xmin = max(x - check_radius, 0)
    xmax = min(x + check_radius + 1, W)
    window = mask[ymin:ymax, xmin:xmax]
    frac_white = window.mean()
    return frac_white >= frac_thresh'''


def endpoint_surrounded_by_white(endpoint, mask, check_radius=6, frac_thresh=0.85):
    y, x = endpoint
    H, W = mask.shape
    ymin = max(y - check_radius, 0)
    ymax = min(y + check_radius + 1, H)
    xmin = max(x - check_radius, 0)
    xmax = min(x + check_radius + 1, W)
    window = mask[ymin:ymax, xmin:xmax]

    # Circular mask
    cy, cx = y - ymin, x - xmin  # center in window
    yy, xx = np.ogrid[:window.shape[0], :window.shape[1]]
    circle = (yy - cy) ** 2 + (xx - cx) ** 2 <= check_radius ** 2

    frac_white = window[circle].mean()
    return frac_white >= frac_thresh

def aggressive_length_prune(skel, width_map, base_len=10, width_scale=2.0, max_prune_len=None):
    branches = find_all_branches(skel)
    pruned = skel.copy()
    pruned_pixels = 0
    num_pruned = 0
    for endpoint, path in branches:
        branch_len = len(path)
        y_ep, x_ep = endpoint
        w = width_map[y_ep, x_ep]
        dynamic_thresh = base_len + (w / width_scale)
        if branch_len <= int(round(dynamic_thresh)) and branch_len <= max_prune_len:
            num_pruned += 1
            for (y, x) in path:
                if pruned[y, x]:
                    pruned[y, x] = 0
                    pruned_pixels += 1
    return pruned, pruned_pixels, len(branches), num_pruned

def near_image_border(endpoint, H, W, radius=15):
    y, x = endpoint
    return (y < radius or y >= H - radius or x < radius or x >= W - radius)

def length_and_surrounded_prune(skel, width_map, mask, base_len=10, width_scale=2.0,
                                check_radius=6, frac_thresh=0.85, max_prune_len=None,
                                edge_protect_radius=15):
    branches = find_all_branches(skel)
    pruned = skel.copy()
    pruned_pixels = 0
    num_pruned = 0
    H, W = mask.shape

    for endpoint, path in branches:
        # --- Edge protection: never prune if endpoint near image border ---
        if near_image_border(endpoint, H, W, edge_protect_radius):
            continue  # keep this branch

        branch_len = len(path)
        y_ep, x_ep = endpoint
        w = width_map[y_ep, x_ep]
        dynamic_thresh = base_len + (w / width_scale)
        if branch_len <= max_prune_len and (
            branch_len <= int(round(dynamic_thresh)) or
            endpoint_surrounded_by_white(endpoint, mask, check_radius, frac_thresh)
        ):
            num_pruned += 1
            for (y, x) in path:
                if pruned[y, x]:
                    pruned[y, x] = 0
                    pruned_pixels += 1
    return pruned, pruned_pixels, len(branches), num_pruned

def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"INPUT_DIR does not exist: {INPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    header = (
        "image",
        "raw_pixels",
        "pruned_pixels_len",
        "pruned_pixels_sur",
        "num_branches",
        "num_pruned_len",
        "num_pruned_sur",
        "time_len_prune",
        "time_sur_prune"
    )
    print(",".join(header))

    for img_path in sorted(INPUT_DIR.glob("*.*")):
        stem = img_path.stem
        bin_mask, float_img = load_binarize(img_path, thresh=MASK_THRESH)
        H, W = bin_mask.shape
        raw_skel = skeletonize(bin_mask.astype(bool)).astype(np.uint8)
        dmap = distance_transform_edt(bin_mask > 0)
        width_map = 2.0 * dmap

        # Compute dynamic thresholds per image size
        MAX_PRUNE_LEN = min(H, W) // 8
        MIN_MAIN_PATH_LEN = max(30, min(H, W) // 64)  # minimum 30 pixels just in case

        t0 = time.time()
        pruned_skel_len, pruned_px_len, num_br, num_pruned_len = aggressive_length_prune(
            raw_skel, width_map, base_len=BASE_LEN, width_scale=WIDTH_SCALE, max_prune_len=MAX_PRUNE_LEN
        )
        t1 = time.time()
        time_len = t1 - t0

        # Length+Surrounded prune (timed)
        t2 = time.time()
        pruned_skel_sur, pruned_px_sur, _, num_pruned_sur = length_and_surrounded_prune(
            raw_skel, width_map, bin_mask,
            base_len=BASE_LEN, width_scale=WIDTH_SCALE,
            check_radius=6, frac_thresh=0.85,
            max_prune_len=MAX_PRUNE_LEN
        )
        t3 = time.time()
        time_sur = t3 - t2

        skel_len_main = keep_major_components(pruned_skel_len, min_len=MIN_MAIN_PATH_LEN)
        skel_sur_main = keep_major_components(pruned_skel_sur, min_len=MIN_MAIN_PATH_LEN)

        # Save overlays
        save_overlay(float_img, skel_len_main, OUTPUT_DIR / f"{stem}_len_overlay.png")
        save_overlay(float_img, skel_sur_main, OUTPUT_DIR / f"{stem}_sur_overlay.png")

        row = (
            stem,
            int(raw_skel.sum()),
            pruned_px_len,
            pruned_px_sur,
            num_br,
            num_pruned_len,
            num_pruned_sur,
            f"{time_len:.4f}",
            f"{time_sur:.4f}"
        )
        print(",".join(str(x) for x in row))

if __name__ == "__main__":
    main()