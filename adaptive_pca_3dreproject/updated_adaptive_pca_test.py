import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, distance_transform_edt
from skimage.morphology import skeletonize
from pathlib import Path

def preprocess_mask(mask):
    """Denoise, close gaps, and slightly erode mask for cleaner skeletonization."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.erode(mask, kernel, iterations=1)
    return mask

def get_skeleton(binary_mask):
    bin_mask = (binary_mask > 0).astype(np.uint8)
    return skeletonize(bin_mask).astype(np.uint8)

def sobel_normal(mask, y, x, window=9):
    half = window // 2
    y0, y1 = max(0, y-half), min(mask.shape[0], y+half+1)
    x0, x1 = max(0, x-half), min(mask.shape[1], x+half+1)
    region = (mask[y0:y1, x0:x1] > 0).astype(np.float32)
    grad_x = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
    center = (min(half, grad_x.shape[0]-1), min(half, grad_x.shape[1]-1))
    gx = grad_x[center]
    gy = grad_y[center]
    norm = np.array([-gy, gx])
    if np.linalg.norm(norm) == 0:
        norm = np.array([0, 1])
    return norm / (np.linalg.norm(norm) + 1e-6)

def sample_edges_along_normal(center, normal, binary_mask, max_distance=20):
    y0, x0 = center
    edges = []
    for sign in [-1, 1]:
        for d in range(1, max_distance):
            dx, dy = normal[1] * d * sign, normal[0] * d * sign
            xi, yi = int(round(x0 + dx)), int(round(y0 + dy))
            if 0 <= yi < binary_mask.shape[0] and 0 <= xi < binary_mask.shape[1]:
                if binary_mask[yi, xi] == 0:
                    # Subpixel edge estimation: Linear interpolation between inside and outside pixels
                    prev_x, prev_y = x0 + normal[1] * (d-1) * sign, y0 + normal[0] * (d-1) * sign
                    alpha = 0.5  # Optionally interpolate for subpixel edge
                    edge_x = alpha * xi + (1-alpha) * prev_x
                    edge_y = alpha * yi + (1-alpha) * prev_y
                    edges.append((edge_x, edge_y))
                    break
    return edges

def calculate_crack_width(edge_points):
    if len(edge_points) == 2:
        (x1, y1), (x2, y2) = edge_points
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return None

def max_blob_width(mask, center, window=25):
    y, x = center
    half = window // 2
    y0, y1 = max(0, y-half), min(mask.shape[0], y+half+1)
    x0, x1 = max(0, x-half), min(mask.shape[1], x+half+1)
    submask = mask[y0:y1, x0:x1] > 0
    contours, _ = cv2.findContours(submask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or len(contours[0]) < 2:
        return 0
    pts = contours[0][:,0,:]
    dists = np.sqrt(np.sum((pts[None, :, :] - pts[:, None, :])**2, axis=-1))
    return dists.max()

def adaptive_crack_width(mask, min_window=7, max_window=15, junction_width_factor=1.0):
    skeleton = get_skeleton(mask)
    crack_widths = []
    points = np.argwhere(skeleton > 0)
    dist_transform = distance_transform_edt(mask > 0)
    for (y, x) in points:
        skel_neighbors = skeleton[max(0, y-1):y+2, max(0, x-1):x+2]
        num_neighbors = np.sum(skel_neighbors) - skeleton[y, x]
        # Junction/blob handling
        if num_neighbors > 2:
            width = max_blob_width(mask, (y, x), window=25) * junction_width_factor
            crack_widths.append({
                'center_x': x, 'center_y': y, 'normal_x': 0, 'normal_y': 1,
                'edge1_x': np.nan, 'edge1_y': np.nan, 'edge2_x': np.nan, 'edge2_y': np.nan,
                'width_px': width, 'junction': True
            })
            continue
        normal = sobel_normal(mask, y, x)
        edges = sample_edges_along_normal((y, x), normal, mask)
        width = calculate_crack_width(edges)
        if width is not None and len(edges) == 2:
            (x1, y1), (x2, y2) = edges
            crack_widths.append({
                'center_x': x, 'center_y': y, 'normal_x': normal[0], 'normal_y': normal[1],
                'edge1_x': x1, 'edge1_y': y1, 'edge2_x': x2, 'edge2_y': y2,
                'width_px': width, 'junction': False
            })
    return crack_widths, skeleton

def draw_width_lines(mask, crack_widths, skeleton, colormap='coolwarm_r'):
    overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    if not crack_widths:
        return overlay
    raw_widths = np.array([cw['width_px'] for cw in crack_widths])
    smoothed = gaussian_filter1d(raw_widths, sigma=2)
    norm = plt.Normalize(vmin=np.percentile(smoothed, 5), vmax=np.percentile(smoothed, 95))
    cmap = plt.get_cmap(colormap)
    for cw, width in zip(crack_widths, smoothed):
        if cw['junction']:
            color = (0, 255, 255)  # Yellow for junction/blob
        else:
            color = cmap(norm(width))[:3]
            color = tuple(int(255 * c) for c in color[::-1])
        # Draw width line
        if not cw['junction'] and not np.isnan(cw['edge1_x']) and not np.isnan(cw['edge2_x']):
            pt1 = (int(round(cw['edge1_x'])), int(round(cw['edge1_y'])))
            pt2 = (int(round(cw['edge2_x'])), int(round(cw['edge2_y'])))
            cv2.line(overlay, pt1, pt2, color, 1)
    # Overlay the skeleton in green
    ys, xs = np.where(skeleton > 0)
    for x, y in zip(xs, ys):
        cv2.circle(overlay, (x, y), 1, (0, 255, 0), -1)
    background = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(background, 0.25, overlay, 0.75, 0)
    return overlay

def save_crack_widths_csv(crack_widths, output_csv):
    df = pd.DataFrame(crack_widths)
    df.to_csv(output_csv, index=False)
    print(f"[✓] Saved crack widths to: {output_csv}")

if __name__ == "__main__":
    outputs_folder = Path(r"D:\camerer_ml\adaptive_pca_3dreproject\width_outputs")
    output_img_path = str(outputs_folder / "smoothed_norm_width_overlay.png")
    output_csv_path = str(outputs_folder / "crack_widths.csv")
    mask_path = r"D:\camerer_ml\datasets\concrete3k\concrete3k\images\001_2.jpg"  # Replace with your binary mask path
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask.max() > 1:
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    mask = preprocess_mask(mask)
    crack_widths, skeleton = adaptive_crack_width(mask)
    overlay = draw_width_lines(mask, crack_widths, skeleton)
    cv2.imwrite(output_img_path, overlay)
    print(f"[✓] Saved overlay image to: {output_img_path}")
    save_crack_widths_csv(crack_widths, output_csv_path)
    # Print stats
    if crack_widths:
        w = [cw['width_px'] for cw in crack_widths]
        print(f"Mean width: {np.mean(w):.2f}px | Min: {np.min(w):.2f}px | Max: {np.max(w):.2f}px")
    else:
        print("No widths measured.")
