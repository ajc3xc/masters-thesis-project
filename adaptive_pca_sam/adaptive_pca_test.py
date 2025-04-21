import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt

def get_skeleton(binary_mask):
    return skeletonize(binary_mask > 0).astype(np.uint8)

def extract_neighbors(point, window_size, image_shape):
    y, x = point
    half = window_size // 2
    y_min, y_max = max(0, y - half), min(image_shape[0], y + half + 1)
    x_min, x_max = max(0, x - half), min(image_shape[1], x + half + 1)
    return np.array([[yi, xi] for yi in range(y_min, y_max) for xi in range(x_min, x_max)])

def compute_curvature(neighbors):
    if len(neighbors) < 3:
        return 0
    dy = np.gradient(neighbors[:, 0])
    dx = np.gradient(neighbors[:, 1])
    ddy = np.gradient(dy)
    ddx = np.gradient(dx)
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-6)**1.5
    return np.mean(curvature)

def perform_pca(neighbors):
    pca_data = neighbors - np.mean(neighbors, axis=0)
    _, _, vh = np.linalg.svd(pca_data)
    tangent = vh[0]
    normal = vh[1]
    return tangent, normal

def sample_edges_along_normal(center, normal, binary_mask, max_distance=20):
    y0, x0 = center
    samples = []
    for sign in [-1, 1]:
        for d in range(1, max_distance):
            dx, dy = normal[1] * d * sign, normal[0] * d * sign
            xi, yi = int(round(x0 + dx)), int(round(y0 + dy))
            if 0 <= yi < binary_mask.shape[0] and 0 <= xi < binary_mask.shape[1]:
                if binary_mask[yi, xi] == 0:
                    samples.append((xi, yi))
                    break
    return samples

def calculate_crack_width(edge_points):
    if len(edge_points) == 2:
        (x1, y1), (x2, y2) = edge_points
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return None

def adaptive_crack_width(mask, min_window=7, max_window=15, curvature_thresh=0.1):
    skeleton = get_skeleton(mask)
    crack_widths = []

    points = np.argwhere(skeleton > 0)
    for (y, x) in points:
        neighbors = extract_neighbors((y, x), max_window, mask.shape)
        local_skel = skeleton[neighbors[:, 0], neighbors[:, 1]]
        curvature = compute_curvature(neighbors[local_skel > 0])

        window_size = min_window if curvature > curvature_thresh else max_window
        sub_neighbors = extract_neighbors((y, x), window_size, mask.shape)
        active = sub_neighbors[skeleton[sub_neighbors[:, 0], sub_neighbors[:, 1]] > 0]

        if len(active) < 3:
            continue

        tangent, normal = perform_pca(active)
        edge_pts = sample_edges_along_normal((y, x), normal, mask)
        width = calculate_crack_width(edge_pts)

        if width is not None:
            crack_widths.append(((x, y), normal, width))

    return crack_widths

def draw_width_lines(mask, crack_widths, color=(0, 255, 0)):
    overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    for (cx, cy), normal, width in crack_widths:
        perp = np.array([-normal[1], normal[0]], dtype=np.float32)
        perp /= np.linalg.norm(perp) + 1e-6
        half_width = width / 2.0
        pt1 = (int(cx + perp[0] * half_width), int(cy + perp[1] * half_width))
        pt2 = (int(cx - perp[0] * half_width), int(cy - perp[1] * half_width))
        cv2.line(overlay, pt1, pt2, color, 1)
    return overlay

def save_crack_widths_csv(crack_widths, output_csv):
    rows = []
    for (x, y), _, width in crack_widths:
        rows.append({'x': x, 'y': y, 'width_px': width})
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"[✓] Saved crack widths to: {output_csv}")

if __name__ == "__main__":
    output_img_path = "width_overlay.png"
    output_csv_path = "crack_widths.csv"
    mask_path = "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/data/cracks/crack_segmentation_dataset/masks/CFD_001.jpg"

    # Load binary crack mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask.max() > 1:
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Run adaptive PCA-based crack width estimation
    widths = adaptive_crack_width(mask)

    # Save image with overlay
    overlay = draw_width_lines(mask, widths)
    cv2.imwrite(output_img_path, overlay)
    print(f"[✓] Saved overlay image to: {output_img_path}")

    # Save CSV
    save_crack_widths_csv(widths, output_csv_path)

    # Print stats
    if widths:
        w = [w for _, _, w in widths]
        print(f"Mean width: {np.mean(w):.2f}px | Min: {np.min(w):.2f}px | Max: {np.max(w):.2f}px")
    else:
        print("No widths measured.")
