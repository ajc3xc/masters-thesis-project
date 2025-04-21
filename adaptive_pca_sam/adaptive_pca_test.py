import cv2
import numpy as np
import os
from sklearn.decomposition import PCA

def adaptive_pca_on_mask(mask, stride=10, window_size=15, min_points=10):
    """
    Run adaptive PCA on connected regions of a binary mask to fit direction lines.
    Returns a list of lines [(x1, y1, x2, y2), ...].
    """
    points = np.column_stack(np.where(mask > 0))  # (y, x)
    if len(points) < min_points:
        return []

    lines = []
    for i in range(0, len(points), stride):
        y0, x0 = points[i]
        y_min = max(y0 - window_size, 0)
        y_max = min(y0 + window_size, mask.shape[0])
        x_min = max(x0 - window_size, 0)
        x_max = min(x0 + window_size, mask.shape[1])
        
        local_points = np.column_stack(np.where(mask[y_min:y_max, x_min:x_max] > 0))
        if len(local_points) < min_points:
            continue

        local_points += [y_min, x_min]  # shift back to global coords
        pca = PCA(n_components=2)
        pca.fit(local_points)
        center = np.mean(local_points, axis=0)
        direction = pca.components_[0]

        line_length = 20
        x1, y1 = center + direction * line_length
        x2, y2 = center - direction * line_length
        lines.append((int(y1), int(x1), int(y2), int(x2)))
    return lines

def draw_lines_on_image(image, lines, color=(0, 255, 0), thickness=2):
    """
    Draw a list of lines on an image.
    """
    image_copy = image.copy()
    for (x1, y1, x2, y2) in lines:
        cv2.line(image_copy, (x1, y1), (x2, y2), color, thickness)
    return image_copy

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
    result = draw_lines_on_image(image, lines)
    cv2.imwrite(output_path, result)
    print(f"[✓] Saved result to {output_path}")

# Example usage
if __name__ == "__main__":
    image_path = "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/data/cracks/crack_segmentation_dataset/images/CFD_001.jpg"
    mask_path = "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/data/cracks/crack_segmentation_dataset/masks/CFD_001.jpg"
    output_path = "pca_result.jpg"

   # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    run_adaptive_pca_pipeline(image_path, mask_path, output_path)
