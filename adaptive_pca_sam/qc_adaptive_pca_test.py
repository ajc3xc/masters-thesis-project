import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt

# ──────────────── Quality Filtering ────────────────
def mask_quality_check(mask, min_area=20, max_area=50000, min_aspect_ratio=0.05, min_eccentricity=0.8):
    # Reject if mask is empty
    if np.count_nonzero(mask) == 0:
        return False

    # 1. Area check
    area = np.sum(mask > 0)
    if not (min_area <= area <= max_area):
        return False

    # 2. Topological hole check using contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        holes = sum(1 for h in hierarchy[0] if h[3] != -1)
        if holes > 0:
            return False

    # 3. Skeleton aspect ratio
    skeleton = skeletonize(mask > 0).astype(np.uint8)
    skeleton_length = np.sum(skeleton)
    aspect_ratio = skeleton_length / (area + 1e-6)
    if aspect_ratio < min_aspect_ratio:
        return False

    # 4. Eccentricity (cracks are usually long/thin)
    labeled = label(mask)
    props = regionprops(labeled)
    if not props:
        return False
    if props[0].eccentricity < min_eccentricity:
        return False

    return True

# ──────────────── Crack Width Estimation ────────────────
def extract_neighbors(point, window_size, shape):
    y, x = point
    half = window_size // 2
    y_min, y_max = max(0, y - half), min(shape[0], y + half + 1)
    x_min, x_max = max(0, x - half), min(shape[1], x + half + 1)
    return np.array([[yi, xi] for yi in range(y_min, y_max) for xi in range(x_min, x_max)])

def compute_curvature(neighbors):
    if len(neighbors) < 3: return 0
    dy, dx = np.gradient(neighbors[:, 0]), np.gradient(neighbors[:, 1])
    ddy, ddx = np.gradient(dy), np.gradient(dx)
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-6)**1.5
    return np.mean(curvature)

def perform_pca(neighbors):
    data = neighbors - np.mean(neighbors, axis=0)
    _, _, vh = np.linalg.svd(data)
    return vh[0], vh[1]  # tangent, normal

def sample_edges_along_normal(center, normal, mask, max_dist=20):
    y0, x0 = center
    samples = []
    for sign in [-1, 1]:
        for d in range(1, max_dist):
            dx, dy = normal[1] * d * sign, normal[0] * d * sign
            xi, yi = int(round(x0 + dx)), int(round(y0 + dy))
            if 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1]:
                if mask[yi, xi] == 0:
                    samples.append((xi, yi))
                    break
    return samples

def calculate_crack_width(edge_pts):
    if len(edge_pts) == 2:
        (x1, y1), (x2, y2) = edge_pts
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return None

def adaptive_crack_width(mask, min_window=7, max_window=15, curvature_thresh=0.1):
    skeleton = skeletonize(mask > 0).astype(np.uint8)
    points = np.argwhere(skeleton > 0)
    results = []

    for (y, x) in points:
        neighbors = extract_neighbors((y, x), max_window, mask.shape)
        skel_pts = neighbors[skeleton[neighbors[:, 0], neighbors[:, 1]] > 0]
        if len(skel_pts) < 3: continue

        curvature = compute_curvature(skel_pts)
        window_size = min_window if curvature > curvature_thresh else max_window

        sub_neighbors = extract_neighbors((y, x), window_size, mask.shape)
        active = sub_neighbors[skeleton[sub_neighbors[:, 0], sub_neighbors[:, 1]] > 0]
        if len(active) < 3: continue

        tangent, normal = perform_pca(active)
        edges = sample_edges_along_normal((y, x), normal, mask)
        width = calculate_crack_width(edges)
        if width:
            results.append(((x, y), normal, width))

    return results

def draw_overlay(mask, widths, color=(0, 255, 0)):
    overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    for (cx, cy), normal, width in widths:
        perp = np.array([-normal[1], normal[0]], dtype=np.float32)
        perp /= np.linalg.norm(perp) + 1e-6
        half = width / 2
        pt1 = (int(cx + perp[0] * half), int(cy + perp[1] * half))
        pt2 = (int(cx - perp[0] * half), int(cy - perp[1] * half))
        cv2.line(overlay, pt1, pt2, color, 1)
    return overlay

def save_to_csv(results, out_csv):
    df = pd.DataFrame([{'x': x, 'y': y, 'width_px': w} for (x, y), _, w in results])
    df.to_csv(out_csv, index=False)
    print(f"[✓] Saved CSV: {out_csv}")

# ──────────────── Entry Point ────────────────
if __name__ == "__main__":
    mask_path = "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/data/cracks/crack_segmentation_dataset/masks/CFD_001.jpg"
    overlay_out = "qc_width_output.png"
    csv_out = "qc_crack_widths.csv"

    # Load binary mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask.max() > 1:
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Run filtering
    if not mask_quality_check(mask):
        print("[✗] Mask rejected by quality filter.")
        exit()

    # Estimate width
    results = adaptive_crack_width(mask)

    if not results:
        print("[✗] No valid crack widths measured.")
    else:
        # Save visual overlay
        out_img = draw_overlay(mask, results)
        cv2.imwrite(overlay_out, out_img)
        save_to_csv(results, csv_out)

        widths = [w for _, _, w in results]
        print(f"[✓] Mean width: {np.mean(widths):.2f} px | Points: {len(widths)}")