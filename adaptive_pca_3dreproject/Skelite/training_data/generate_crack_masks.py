"""
generate_crack_widths.py

For each (image, mask) pair in the supplied directories:
  1) Preprocess the binary mask (close, erode) → more stable skeleton.
  2) Prune short spurs on that mask (≤ prune_pre).
  3) Skeletonize → 1‐pixel‐wide skeleton.
  4) Measure “width” at each skeleton pixel:
       • If the skeleton pixel has degree >2 → treat as “junction” and measure a local blob‐width.
       • Else → compute Sobel‐based normal → walk ± normal until background to find edges.
  5) Post‐prune any remaining skeleton spurs (≤ prune_post).
  6) Save a CSV (columns: y, x, normal_y, normal_x, edge1_x, edge1_y, edge2_x, edge2_y, width_px, junction).
  7) (Optional) Save an overlay: 
       • Grayscale background  
       • Red line segments (each measured width)  
       • Green dots (each surviving skeleton pixel)

Usage:
    python generate_crack_widths.py \
      --image_dir /path/to/images \
      --mask_dir  /path/to/binary_masks \
      --output_dir /path/to/save_csvs \
      --overlay_dir /path/to/save_overlays \
      --prune_pre 15 \
      --prune_post 10 \
      --max_search 20
"""

import os
import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
from scipy.ndimage import convolve, distance_transform_edt
from scipy.ndimage import binary_closing, binary_erosion
import argparse
from pathlib import Path


# --------------------------------------------------------------------------------
# 1) LOW‐LEVEL HELPER FUNCTIONS
# --------------------------------------------------------------------------------

def preprocess_mask(mask: np.ndarray, close_iter: int = 1, erode_iter: int = 1) -> np.ndarray:
    """
    Apply a 3×3 elliptical closing then a slight erosion to the binary mask
    to remove small holes and “jitter,” which stabilizes skeletonization.
    Input: mask should be a bool or 0/1 uint8 array.
    Returns: 0/1 uint8 array after morphological operations.
    """
    # Convert boolean → uint8 (0 or 255)
    m8 = (mask > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    m_closed = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    m_eroded = cv2.erode(m_closed, kernel, iterations=erode_iter)
    return (m_eroded > 0).astype(np.uint8)


def prune_branches(skel: np.ndarray, max_length: int = 10) -> np.ndarray:
    """
    Given a binary skeleton (bool or 0/1), remove any spur (branch) whose length ≤ max_length.
    Returns a pruned boolean skeleton.
    """
    sk = skel.astype(np.uint8).copy()
    # Kernel to detect endpoints (value == 11 ⇒ exactly one neighbor)
    k = np.array([[1,1,1],[1,10,1],[1,1,1]], dtype=np.uint8)

    # Find all endpoints (pixel where convolve(sk, k) == 11)
    endpoints = np.argwhere(convolve(sk, k, mode='constant') == 11)
    H, W = sk.shape

    for (y0, x0) in endpoints:
        path = [(y0, x0)]
        sk[y0, x0] = 0  # remove this endpoint
        y, x = y0, x0

        # Walk along the branch up to max_length pixels
        for _ in range(max_length):
            neighbors = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and sk[ny, nx]:
                        neighbors.append((ny, nx))
            # If there is not exactly one neighbor, we hit a junction or dead‐end
            if len(neighbors) != 1:
                break
            y, x = neighbors[0]
            path.append((y, x))
            sk[y, x] = 0

        # If the branch was too long, restore it; else leave it removed
        if len(path) > max_length:
            for (yy, xx) in path:
                sk[yy, xx] = 1

    return sk.astype(bool)


def find_neighbors(skel: np.ndarray, y: int, x: int) -> list:
    """
    Return a list of (ny,nx) coordinates of neighboring skeleton pixels
    (8‐neighborhood) for skeleton array `skel` at location (y,x), excluding (y,x) itself.
    """
    H, W = skel.shape
    nbrs = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and skel[ny, nx]:
                nbrs.append((ny, nx))
    return nbrs


def max_blob_width(mask: np.ndarray, center: tuple, window: int = 25) -> float:
    """
    For junction points (degree > 2), estimate a “blob” width by taking the largest
    pairwise distance among contour points within a window×window patch around `center`.
    Returns the maximum Euclidean distance in pixels.
    """
    y, x = center
    H, W = mask.shape
    half = window // 2
    y0, y1 = max(0, y-half), min(H, y+half+1)
    x0, x1 = max(0, x-half), min(W, x+half+1)
    submask = (mask[y0:y1, x0:x1] > 0).astype(np.uint8)

    # Find contours in the local patch
    contours, _ = cv2.findContours(submask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or len(contours[0]) < 3:
        return 0.0
    pts = contours[0].reshape(-1, 2).astype(np.float32)

    # Compute all pairwise distances (N×N) – okay for small N
    d2 = (pts[:, None, :] - pts[None, :, :]) ** 2
    dists = np.sqrt(d2.sum(axis=-1))
    return float(dists.max())


def sobel_normal(mask: np.ndarray, y: int, x: int, window: int = 9) -> np.ndarray:
    """
    Extract a local patch (window×window) around (y,x) from `mask` (0/1 binary),
    compute Sobel gradients, and return the normalized vector perpendicular to the crack.
    The “normal” direction is (−gy, +gx).
    Returns a NumPy array [normal_y, normal_x].
    """
    half = window // 2
    H, W = mask.shape
    y0, y1 = max(0, y-half), min(H, y+half+1)
    x0, x1 = max(0, x-half), min(W, x+half+1)

    patch = (mask[y0:y1, x0:x1] > 0).astype(np.float32)
    grad_x = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)

    # The “center” of the patch in patch‐coordinates
    cy = min(half, grad_x.shape[0] - 1)
    cx = min(half, grad_x.shape[1] - 1)

    gx = float(grad_x[cy, cx])
    gy = float(grad_y[cy, cx])
    # A normal to the crack is (−gy, +gx)
    nx = -gy
    ny =  gx
    nrm = np.hypot(nx, ny) + 1e-8
    return np.array([ny / nrm, nx / nrm], dtype=np.float32)  # (normal_y, normal_x)


def sample_edges_along_normal(
    center: tuple,
    normal: np.ndarray,
    binary_mask: np.ndarray,
    max_distance: int = 20
) -> list:
    """
    Starting from center = (y,x) and unit normal=(ny, nx),
    step in ±normal directions (pixel by pixel) until you hit background (mask==0).
    Once you step out, back up one pixel and take a midpoint → subpixel edge estimate.
    Returns a list of two “edge” points [(ex1, ey1), (ex2, ey2)]. If fewer than 2 were found, returns fewer.
    """
    H, W = binary_mask.shape
    y0, x0 = center
    ny, nx = normal

    edges = []
    for sign in (-1, +1):
        for d in range(1, max_distance + 1):
            yf = y0 + ny * (d * sign)
            xf = x0 + nx * (d * sign)
            yi = int(round(yf))
            xi = int(round(xf))
            if yi < 0 or yi >= H or xi < 0 or xi >= W or (binary_mask[yi, xi] == 0):
                # We stepped out of the crack region; record last in‐crack pixel
                d_prev = d - 1
                yp = y0 + ny * (d_prev * sign)
                xp = x0 + nx * (d_prev * sign)
                # Record midpoint between (xp,yp) and (xf,yf)
                alpha = 0.5
                edge_x = alpha * xi + (1 - alpha) * xp
                edge_y = alpha * yi + (1 - alpha) * yp
                edges.append((edge_x, edge_y))
                break
    return edges


# --------------------------------------------------------------------------------
# 2) MAIN FUNCTION TO PROCESS ONE (IMAGE, MASK) ⇒ CSV + OVERLAY
# --------------------------------------------------------------------------------

def process_one(
    image_path: str,
    mask_path:  str,
    csv_outpath: str,
    overlay_outpath: str = None,
    prune_pre:  int = 15,
    prune_post: int = 10,
    max_search:  int = 20,
    junction_width_factor: float = 1.0
) -> int:
    """
    1) Read grayscale image + 0/255 mask.
    2) Preprocess mask (closing + erosion).
    3) Pre‐prune spurs ≤ prune_pre.
    4) Skeletonize the cleaned mask.
    5) For each skeleton pixel:
         • If degree > 2 → measure max blob width (junction).
         • Else → compute sobel_normal + sample_edges along ±normal.
    6) Post‐prune skeleton spurs (≤ prune_post).
    7) Save CSV: 
       y, x, normal_y, normal_x, edge1_x, edge1_y, edge2_x, edge2_y, width_px, junction_flag
    8) Save optional overlay (grayscale background + red width lines + green skeleton).
    Returns the number of final skeleton points (rows in CSV).
    """
    # --- (1) Load image + mask ---
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    img_f = img_gray.astype(np.float32) / 255.0  # normalized [0,1]

    m8 = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m8 is None:
        raise RuntimeError(f"Could not read mask: {mask_path}")
    # Binarize mask → 0/1
    _, bw = cv2.threshold(m8, 127, 1, cv2.THRESH_BINARY)

    # --- (2) Preprocess + Pre‐prune mask ---
    bw_pre = preprocess_mask(bw, close_iter=1, erode_iter=1)
    bw_pre_pruned = prune_branches(bw_pre.astype(bool), max_length=prune_pre).astype(np.uint8)

    # --- (3) Skeletonize the pre‐pruned mask ---
    skel_pre = skeletonize(bw_pre_pruned.astype(bool)).astype(np.uint8)

    # --- (4) Measure width at each skeleton pixel ---
    H, W = bw.shape
    data_rows = []
    ys, xs = np.nonzero(skel_pre > 0)

    for (y, x) in zip(ys, xs):
        nbrs = find_neighbors(skel_pre, y, x)
        deg = len(nbrs)

        if deg > 2:
            # Junction point: measure local blob width
            w_px = max_blob_width(bw, (y, x), window=25) * junction_width_factor
            data_rows.append({
                'y': int(y),
                'x': int(x),
                'normal_y': 0.0,
                'normal_x': 0.0,
                'edge1_x': np.nan,
                'edge1_y': np.nan,
                'edge2_x': np.nan,
                'edge2_y': np.nan,
                'width_px': float(w_px),
                'junction': True
            })
        else:
            # Regular skeleton point: compute Sobel normal + find edges
            ny, nx = sobel_normal(bw, y, x, window=9)
            edges = sample_edges_along_normal((y, x), (ny, nx), bw, max_distance=max_search)
            if len(edges) == 2:
                (ex1, ey1), (ex2, ey2) = edges
                w_px = np.hypot(ex1 - ex2, ey1 - ey2)
                data_rows.append({
                    'y': int(y),
                    'x': int(x),
                    'normal_y': float(ny),
                    'normal_x': float(nx),
                    'edge1_x': float(ex1),
                    'edge1_y': float(ey1),
                    'edge2_x': float(ex2),
                    'edge2_y': float(ey2),
                    'width_px': float(w_px),
                    'junction': False
                })
            else:
                # Did not find two edges (too narrow or ambiguous), skip
                continue

    # --- (5) Build final skeleton mask (only those pixels we recorded) and post‐prune ---
    if data_rows:
        skel_mask_final = np.zeros_like(skel_pre, dtype=np.uint8)
        for r in data_rows:
            skel_mask_final[r['y'], r['x']] = 1
        skel_final_pruned = prune_branches(skel_mask_final.astype(bool), max_length=prune_post)
    else:
        skel_final_pruned = np.zeros_like(skel_pre, dtype=bool)

    # Filter data_rows to keep only those (y,x) that survived post‐prune
    final_rows = [r for r in data_rows if skel_final_pruned[r['y'], r['x']]]

    # --- (6) Save CSV ---
    os.makedirs(Path(csv_outpath).parent, exist_ok=True)
    if final_rows:
        df = pd.DataFrame(final_rows)
    else:
        df = pd.DataFrame(columns=[
            'y','x','normal_y','normal_x','edge1_x','edge1_y',
            'edge2_x','edge2_y','width_px','junction'
        ])
    df.to_csv(csv_outpath, index=False)

    # --- (7) OPTIONAL: Overlay (grayscale bg + green skeleton pts + red width lines) ---
    if overlay_outpath:
        # Convert grayscale→BGR
        overlay = cv2.cvtColor((img_f * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # 7a) DRAW THE FINAL SKELETON, FIRST (in bright GREEN)
        #    We'll treat skel_final_pruned (a boolean mask) as a set of points.
        #    To make it more visible, we'll draw each skeleton pixel as a small circle.
        ys2, xs2 = np.nonzero(skel_final_pruned)
        for yy, xx in zip(ys2, xs2):
            # Draw skeleton pixel at (xx, yy) with radius=1, color=green
            cv2.circle(overlay, (int(xx), int(yy)), radius=1, color=(0, 255, 0), thickness=-1)

        # 7b) NOW DRAW EACH WIDTH LINE (in RED) on top of that
        for r in final_rows:
            if (not r['junction']) and (not np.isnan(r['edge1_x'])) and (not np.isnan(r['edge2_x'])):
                pt1 = (int(round(r['edge1_x'])), int(round(r['edge1_y'])))
                pt2 = (int(round(r['edge2_x'])), int(round(r['edge2_y'])))
                # Draw red line between the two edge‐points
                cv2.line(overlay, pt1, pt2, color=(0, 0, 255), thickness=1)

        os.makedirs(Path(overlay_outpath).parent, exist_ok=True)
        cv2.imwrite(overlay_outpath, overlay)

    return len(final_rows)


# --------------------------------------------------------------------------------
# 3) SCRIPT ENTRY POINT: PARSE ARGS & LOOP OVER DIRECTORY
# -------------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate crack widths CSV from (image, mask) pairs.")
    p.add_argument("--image_dir",   type=str, default=r"D:\camerer_ml\datasets\Final-Dataset-Vol1\images",  help="Folder containing grayscale (or RGB) images.")
    p.add_argument("--mask_dir",    type=str, default=r"D:\camerer_ml\datasets\Final-Dataset-Vol1\labels_visible",  help="Folder containing binary masks (0/255).")
    p.add_argument("--output_dir",  type=str, default=r"D:\camerer_ml\adaptive_pca_3dreproject\Skelite\training_data\csvs",  help="Where to write per-image CSVs of widths.")
    p.add_argument("--overlay_dir", type=str, default=r"D:\camerer_ml\adaptive_pca_3dreproject\Skelite\training_data\overlays", help="(Optional) where to save overlay images.")
    p.add_argument("--prune_pre",   type=int, default=15, help="Prune skeleton spurs <= this length, BEFORE measuring widths.")
    p.add_argument("--prune_post",  type=int, default=10, help="Prune skeleton spurs <= this length, AFTER measuring widths.")
    p.add_argument("--max_search",  type=int, default=20, help="Max distance (in px) to search along normal for edges.")
    p.add_argument("--junction_factor", type=float, default=1.0, help="Width multiplier for junction points.")
    args = p.parse_args()

    img_dir  = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    out_dir  = Path(args.output_dir)
    ovl_dir  = Path(args.overlay_dir) if args.overlay_dir else None

    img_paths = sorted(list(img_dir.glob("*.*")))
    for img_path in img_paths:
        stem = img_path.stem
        # Attempt to find a matching mask (common extensions)
        mask_path = None
        for ext in [".png", ".jpg", ".bmp", ".tif"]:
            candidate = mask_dir / f"{stem}{ext}"
            if candidate.exists():
                mask_path = candidate
                break

        if mask_path is None:
            print(f"[WARNING] no mask found for {stem}, skipping.")
            continue

        csv_out = out_dir / f"{stem}_widths.csv"
        ovl_out = None
        if ovl_dir:
            ovl_out = ovl_dir / f"{stem}_overlay.png"

        n_pts = process_one(
            str(img_path),
            str(mask_path),
            str(csv_out),
            overlay_outpath = (str(ovl_out) if ovl_out else None),
            prune_pre = args.prune_pre,
            prune_post = args.prune_post,
            max_search = args.max_search,
            junction_width_factor = args.junction_factor
        )
        print(f"[{stem}] extracted width for {n_pts} skeleton pixels → saved to {csv_out}")
        break

    print("Done generating crack‐width CSVs for all images.")
