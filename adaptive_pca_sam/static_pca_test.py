import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import gaussian_filter1d

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

def sample_edges_along_normal(center, normal, confidence_map, mask, max_distance=20):
    """
    Samples along both directions of the normal vector to find edge crossings
    Uses the confidence map (or the binary mask if no logits available)
    """
    y0, x0 = center
    samples = []
    for sign in [-1, 1]:
        for d in range(1, max_distance):
            dx, dy = normal[1] * d * sign, normal[0] * d * sign
            xi, yi = int(round(x0 + dx)), int(round(y0 + dy))
            if 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1]:
                if mask[yi, xi] == 0:
                    # Optional: subpixel interpolation
                    conf = confidence_map[yi, xi] if confidence_map is not None else 1.0
                    samples.append((xi, yi, conf))
                    break
    return samples

def calculate_crack_width(edge_points):
    if len(edge_points) == 2:
        (x1, y1, _), (x2, y2, _) = edge_points
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return None

def adaptive_crack_width(mask, confidence_map=None, min_window=7, max_window=15, curvature_thresh=0.1, confidence_thresh=0.3):
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
        confidence = confidence_map[y, x] if confidence_map is not None else 1.0
        if confidence < confidence_thresh:
            continue

        edge_pts = sample_edges_along_normal((y, x), normal, confidence_map, mask)
        width = calculate_crack_width(edge_pts)

        if width is not None:
            crack_widths.append(((x, y), width))

    return crack_widths

# Example usage
if __name__ == "__main__":
    image_path = "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/data/cracks/crack_segmentation_dataset/images/CFD_001.jpg"
    mask_path = "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/data/cracks/crack_segmentation_dataset/masks/CFD_001.jpg"
    output_path = "output/pca_result.png"
    csv_path = "output/widths.csv"

    run_adaptive_pca_pipeline(image_path, mask_path, output_path, csv_path, show_labels=False)
