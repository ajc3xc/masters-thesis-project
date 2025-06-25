import os
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from skimage.morphology import skeletonize, medial_axis
from scipy.ndimage import convolve
from scipy.ndimage import distance_transform_edt

# =============================================================================
# ==  1) Helper functions: pruning, metrics, Skelite inference, and edge detection
# =============================================================================

def run_skelite(model: torch.jit.ScriptModule, img_tensor: torch.Tensor) -> (np.ndarray, np.ndarray):
    """
    Runs Skelite on img_tensor (1×1×H×W). Returns:
      - raw_mask:     float32 NumPy array of shape (H, W), values in [0,1]
      - skel_mask:    boolean NumPy array of shape (H, W), thresholded at 0.5
    """
    with torch.no_grad():
        output, _ = model(img_tensor, z=None, no_iter=5)
    raw_mask = output[0, 0].cpu().numpy()
    skel_mask = raw_mask > 0.5
    return raw_mask, skel_mask


def load_and_prep_image(image_path: str, device: torch.device) -> (torch.Tensor, np.ndarray):
    """
    Loads an image (possibly RGB), converts to grayscale, normalizes to [0,1], 
    returns a float32 torch Tensor of shape (1,1,H,W) on device and the raw np.array in [0,1].
    """
    img = imageio.imread(image_path)
    if img.ndim == 3:
        img = img.mean(axis=2)  # to grayscale
    img = img.astype(np.float32) / 255.0
    H, W = img.shape
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
    return tensor, img


def find_endpoints(skel: np.ndarray) -> np.ndarray:
    """
    Returns an array of (y,x) coordinates of skeleton endpoints 
    (i.e. pixels with exactly one neighbor).
    """
    kernel = np.array([[1,1,1],
                       [1,10,1],
                       [1,1,1]])
    filtered = convolve(skel.astype(np.uint8), kernel, mode='constant')
    return np.argwhere(filtered == 11)


def prune_branches(skel: np.ndarray, max_length: int = 15) -> np.ndarray:
    """
    Given a binary skeleton (bool or 0/1), remove any spur (branch) 
    that is ≤ max_length pixels long. Returns a pruned binary skeleton.
    """
    sk = skel.astype(np.uint8).copy()
    endpoints = find_endpoints(sk)
    H, W = sk.shape

    for (y0, x0) in endpoints:
        path = [(y0, x0)]
        sk[y0, x0] = 0
        y, x = y0, x0

        # Walk along the branch up to max_length steps
        for _ in range(max_length):
            neighbors = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and sk[ny, nx]:
                        neighbors.append((ny, nx))
            if len(neighbors) != 1:
                # Reached a junction or end
                break
            y, x = neighbors[0]
            path.append((y, x))
            sk[y, x] = 0

        if len(path) <= max_length:
            # Permanently erase
            for (yy, xx) in path:
                sk[yy, xx] = 0
        else:
            # Restore branch because it’s too long to prune
            for (yy, xx) in path:
                sk[yy, xx] = 1

    return sk.astype(bool)


def skeleton_metrics(skel: np.ndarray) -> dict:
    """
    Counts: 
      - connected components 
      - endpoints 
      - total skeleton pixels.
    Returns a dict.
    """
    from skimage.measure import label
    lbl = label(skel)
    n_comp = int(lbl.max())
    n_end = int(len(find_endpoints(skel)))
    n_pix = int(skel.sum())
    return {"components": n_comp, "endpoints": n_end, "pixels": n_pix}


def compute_edges(bw: np.ndarray, skel: np.ndarray, max_search: int = 50) -> (np.ndarray, np.ndarray):
    """
    Given a binary crack mask `bw` (True inside crack, False outside) and 
    a binary pruned skeleton `skel`, estimate for each skeleton pixel its 
    local gradient direction (via Sobel on a small Gaussian‐smoothed version 
    of `bw`) and then search along the perpendicular line (±) up to max_search 
    pixels to find the crack edge. Returns two binary arrays (left_edges, right_edges).
    """
    H, W = bw.shape
    # 1) Gaussian‐smooth the binary mask (to reduce discretization noise)
    smooth = cv2.GaussianBlur((bw.astype(np.uint8) * 255), (5,5), 0).astype(np.float32) / 255.0
    # 2) Sobel gradients on the smoothed image
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)

    ys, xs = np.nonzero(skel)
    left_edges = np.zeros_like(bw, dtype=bool)
    right_edges = np.zeros_like(bw, dtype=bool)

    for (y, x) in zip(ys, xs):
        gx_val = float(gx[y, x])
        gy_val = float(gy[y, x])
        norm = np.hypot(gx_val, gy_val)
        if norm < 1e-3:
            continue  # no well‐defined direction
        # direction = (gx_val/norm, gy_val/norm)
        # perpendicular = (-dir_y, dir_x)
        perp_x = - (gy_val / norm)
        perp_y =   (gx_val / norm)

        # Search along +perp direction until leaving crack (bw==0)
        for t in range(1, max_search + 1):
            yy = int(round(y + perp_y * t))
            xx = int(round(x + perp_x * t))
            if yy < 0 or yy >= H or xx < 0 or xx >= W or (bw[yy, xx] == 0):
                # Record the last in‐crack pixel (edge) at t-1
                t_prev = t - 1
                yy_p = int(round(y + perp_y * t_prev))
                xx_p = int(round(x + perp_x * t_prev))
                if 0 <= yy_p < H and 0 <= xx_p < W:
                    left_edges[yy_p, xx_p] = True
                break

        # Search along −perp direction
        for t in range(1, max_search + 1):
            yy = int(round(y - perp_y * t))
            xx = int(round(x - perp_x * t))
            if yy < 0 or yy >= H or xx < 0 or xx >= W or (bw[yy, xx] == 0):
                t_prev = t - 1
                yy_p = int(round(y - perp_y * t_prev))
                xx_p = int(round(x - perp_x * t_prev))
                if 0 <= yy_p < H and 0 <= xx_p < W:
                    right_edges[yy_p, xx_p] = True
                break

    return left_edges, right_edges


def save_binary_image(binary: np.ndarray, out_path: str):
    """
    Saves a boolean or 0/1 array as a uint8 PNG (255 for True, 0 for False).
    """
    imageio.imwrite(out_path, (binary.astype(np.uint8) * 255))


def save_comparison_plot(image_name: str, img: np.ndarray,
                         sk1: np.ndarray, sk2: np.ndarray, sk3: np.ndarray, sk4: np.ndarray,
                         save_dir: str):
    """
    Saves a side‐by‐side high‐DPI plot (Original, sk1, sk2, sk3, sk4).
    sk4 is the raw Skelite mask; the others are pruned skeletons.
    """
    plt.figure(figsize=(24, 6), dpi=200)
    titles = [
        "Original",
        "skimage.skeletonize + prune",
        "skimage.medial_axis + prune",
        "OpenCV ximgproc.thin + prune",
        "Skelite Mask (>0.5)",
    ]
    images = [img, sk1, sk2, sk3, sk4]

    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(images[i], cmap="gray", vmin=0, vmax=1)
        plt.title(titles[i], fontsize=14)
        plt.axis("off")

    plt.tight_layout()
    out_path = os.path.join(save_dir, f"{Path(image_name).stem}_comparison.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# =============================================================================
# ==  2) Main batch script: benchmarking, skeletons, metrics, edge detection
# =============================================================================

if __name__ == "__main__":
    # === User‐configurable paths ===
    image_dir    = r"D:\camerer_ml\datasets\METU_Concrete_Crack_Seg\BW_visible"
    model_path   = r"D:\camerer_ml\adaptive_pca_3dreproject\Skelite\skelite_scripted.pt"
    output_dir   = r"D:\camerer_ml\adaptive_pca_3dreproject\Skelite\skeleton_comparisons_v2"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "skel_skimage_pruned"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "skel_medial_pruned"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "skel_opencv_pruned"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "skel_skelite_thin_pruned"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "skelite_mask"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "skelite_edges"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "comparison_plots"), exist_ok=True)

    # CSV where we will log times + metrics
    csv_rows = []

    # Load Skelite JIT model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sk_model = torch.jit.load(model_path, map_location=device)
    sk_model.eval()

    all_image_paths = sorted(list(Path(image_dir).glob("*.png")))

    for img_path in all_image_paths:
        image_name = img_path.name
        print(f"---- Processing {image_name} ----")

        # 1) Load & prep
        t0 = time.time()
        img_tensor, img = load_and_prep_image(str(img_path), device)
        load_time = time.time() - t0

        # 2) Binary mask (threshold at 0.5)
        t0 = time.time()
        bw = (img > 0.5)
        mask_time = time.time() - t0

        H, W = bw.shape
        metrics_row = {
            "image": image_name,
            "load_time_s": load_time,
            "mask_time_s": mask_time
        }

        # ============================
        # A) skimage.skeletonize + prune
        # ============================
        t0 = time.time()
        sk1 = skeletonize(bw)                      # classic thinning
        prune1 = prune_branches(sk1, max_length=15)
        time_sk1 = time.time() - t0
        m1 = skeleton_metrics(prune1)
        metrics_row.update({
            "skimage_skel_time_s": time_sk1,
            "skimage_skel_components": m1["components"],
            "skimage_skel_endpoints":  m1["endpoints"],
            "skimage_skel_pixels":     m1["pixels"]
        })
        # Save pruned skeleton image
        save_binary_image(prune1, os.path.join(output_dir, "skel_skimage_pruned", f"{Path(image_name).stem}_skimage_pruned.png"))

        # ============================
        # B) skimage.medial_axis + prune
        # ============================
        t0 = time.time()
        sk2 = medial_axis(bw)                      # medial axis thinning
        prune2 = prune_branches(sk2, max_length=15)
        time_sk2 = time.time() - t0
        m2 = skeleton_metrics(prune2)
        metrics_row.update({
            "medial_time_s": time_sk2,
            "medial_components": m2["components"],
            "medial_endpoints":  m2["endpoints"],
            "medial_pixels":     m2["pixels"]
        })
        save_binary_image(prune2, os.path.join(output_dir, "skel_medial_pruned", f"{Path(image_name).stem}_medial_pruned.png"))

        # =============================================
        # C) OpenCV ximgproc.thinning + prune
        # =============================================
        t0 = time.time()
        # Ensure the mask is uint8 0/255
        bw_uint8 = (bw.astype(np.uint8) * 255)
        sk3 = cv2.ximgproc.thinning(bw_uint8, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        sk3 = (sk3 > 0)  # boolean skeleton
        prune3 = prune_branches(sk3, max_length=15)
        time_sk3 = time.time() - t0
        m3 = skeleton_metrics(prune3)
        metrics_row.update({
            "opencv_thin_time_s": time_sk3,
            "opencv_thin_components": m3["components"],
            "opencv_thin_endpoints":  m3["endpoints"],
            "opencv_thin_pixels":     m3["pixels"]
        })
        save_binary_image(prune3, os.path.join(output_dir, "skel_opencv_pruned", f"{Path(image_name).stem}_opencv_pruned.png"))

        # =============================================
        # D) Skelite inference → classic thinning + prune
        # =============================================
        # 1) Skelite inference
        t0 = time.time()
        raw_mask, sk_skelite = run_skelite(sk_model, img_tensor)  # boolean mask
        time_skelite_inf = time.time() - t0

        # Save raw Skelite mask immediately
        save_binary_image(sk_skelite, os.path.join(output_dir, "skelite_mask", f"{Path(image_name).stem}_skelite_mask.png"))

        # 2) Thin & prune
        t0 = time.time()
        sk4_raw_thin = skeletonize(sk_skelite)       # classic thinning on Skelite mask
        prune4 = prune_branches(sk4_raw_thin, max_length=15)
        time_skelite_thin = time.time() - t0

        m4 = skeleton_metrics(prune4)
        metrics_row.update({
            "skelite_inf_time_s":     time_skelite_inf,
            "skelite_thin_time_s":    time_skelite_thin,
            "skelite_components":     m4["components"],
            "skelite_endpoints":      m4["endpoints"],
            "skelite_pixels":         m4["pixels"]
        })
        save_binary_image(prune4, os.path.join(output_dir, "skel_skelite_thin_pruned", f"{Path(image_name).stem}_skelite_thin_pruned.png"))

        # =============================================
        # E) Compute edges (only for Skelite pruned skeleton)
        # =============================================
        t0 = time.time()
        # Edge detection on prune4 against original binary mask bw
        left_edges, right_edges = compute_edges(bw, prune4, max_search=50)
        time_edge = time.time() - t0

        metrics_row.update({
            "skelite_edge_time_s": time_edge
        })
        # Save left/right edges as PNG
        save_binary_image(left_edges,  os.path.join(output_dir, "skelite_edges", f"{Path(image_name).stem}_skelite_left_edges.png"))
        save_binary_image(right_edges, os.path.join(output_dir, "skelite_edges", f"{Path(image_name).stem}_skelite_right_edges.png"))

        # =============================================
        # F) Save side‐by‐side comparison plot
        # =============================================
        save_comparison_plot(
            image_name,
            img,
            prune1.astype(np.float32),
            prune2.astype(np.float32),
            prune3.astype(np.float32),
            sk_skelite.astype(np.float32),
            save_dir=os.path.join(output_dir, "comparison_plots")
        )

        # Append metrics row
        csv_rows.append(metrics_row)

        print(f"  Times (s): skimage_skel={time_sk1:.3f}, medial={time_sk2:.3f}, "
              f"opencv={time_sk3:.3f}, skelite_inf={time_skelite_inf:.3f}, skelite_thin={time_skelite_thin:.3f}, "
              f"edge_detect={time_edge:.3f}")

    # === After all images, write CSV ===
    df = pd.DataFrame(csv_rows)
    csv_out = os.path.join(output_dir, "skeleton_benchmark_metrics.csv")
    df.to_csv(csv_out, index=False)
    print(f"\nBatch processing complete! Metrics saved to:\n  {csv_out}")
