import cv2
import numpy as np
import os
from sklearn.decomposition import PCA

def adaptive_pca_on_mask(mask, stride=10, window_size=15, min_points=10):
    """
    Run adaptive PCA on binary mask and compute local crack widths.
    Returns list of (x1, y1, x2, y2, width).
    """
    points = np.column_stack(np.where(mask > 0))  # (y, x)
    if len(points) < min_points:
        return []

    results = []
    for i in range(0, len(points), stride):
        y0, x0 = points[i]
        y_min = max(y0 - window_size, 0)
        y_max = min(y0 + window_size, mask.shape[0])
        x_min = max(x0 - window_size, 0)
        x_max = min(x0 + window_size, mask.shape[1])
        
        local_points = np.column_stack(np.where(mask[y_min:y_max, x_min:x_max] > 0))
        if len(local_points) < min_points:
            continue

        local_points += [y_min, x_min]  # shift to global
        pca = PCA(n_components=2)
        pca.fit(local_points)

        center = np.mean(local_points, axis=0)
        v_main = pca.components_[0]
        v_perp = pca.components_[1]

        # Project all local points onto perpendicular axis
        projections = local_points @ v_perp
        crack_width = projections.max() - projections.min()

        # Flip PCA to maintain consistent direction
        #if len(results) > 0:
        #    prev_dir = np.array([results[-1][0] - results[-1][2], results[-1][1] - results[-1][3]])
        #    if np.dot(v_main, prev_dir) < 0:
        #        v_main = -v_main

        # Extend direction line for visualization
        length = 20
        x1, y1 = center + v_main * length
        x2, y2 = center - v_main * length
        results.append((int(y1), int(x1), int(y2), int(x2), crack_width))

    return results

def draw_crack_width_lines(image, results, color=(0, 255, 0), thickness=1):
    """
    Draw perpendicular crack width lines at each PCA point.
    """
    image_copy = image.copy()
    for (cx, cy, dx, dy, width) in results:
        # Perpendicular unit vector
        perp = np.array([-dy, dx], dtype=np.float32)
        perp /= np.linalg.norm(perp) + 1e-6

        half_width = width / 2.0
        pt1 = (int(cx + perp[0] * half_width), int(cy + perp[1] * half_width))
        pt2 = (int(cx - perp[0] * half_width), int(cy - perp[1] * half_width))

        cv2.line(image_copy, pt1, pt2, color, thickness)
        # Optional: draw width as text
        # mid = (int(cx), int(cy))
        # cv2.putText(image_copy, f"{width:.1f}", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    return image_copy

def load_image_and_mask(image_path, mask_path):
    """
    Load original image and binary mask (both grayscale).
    """
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask.max() > 1:
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return image, mask

def load_image_and_mask(image_path, mask_path):
    """
    Load original image and binary mask (both grayscale).
    """
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return image, mask

def run_adaptive_pca_pipeline(image_path, mask_path, output_path):
    """
    Full pipeline: load data, run PCA, export image with lines.
    """
    image, mask = load_image_and_mask(image_path, mask_path)
    lines = adaptive_pca_on_mask(mask)
    result = draw_crack_width_lines(image, lines)
    cv2.imwrite(output_path, result)
    print(f"[✓] Saved result to {output_path}")

# Example usage
if __name__ == "__main__":
    image_path = "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/data/cracks/crack_segmentation_dataset/images/CFD_001.jpg"
    mask_path = "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/data/cracks/crack_segmentation_dataset/masks/CFD_001.jpg"
    output_path = "pca_result.jpg"

   # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    run_adaptive_pca_pipeline(image_path, mask_path, output_path)
