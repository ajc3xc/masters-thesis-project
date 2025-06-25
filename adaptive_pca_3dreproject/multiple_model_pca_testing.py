import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, medial_axis
from scipy.ndimage import distance_transform_edt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# ----------------------- Helper Functions -----------------------

def get_skeleton(mask):
    return skeletonize(mask > 0).astype(np.uint8)

def get_medial_axis(mask):
    skel, distance = medial_axis(mask > 0, return_distance=True)
    return skel.astype(np.uint8), distance

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

def perform_pca(neighbors):
    pca_data = neighbors - np.mean(neighbors, axis=0)
    _, _, vh = np.linalg.svd(pca_data)
    tangent = vh[0]
    normal = vh[1]
    return tangent, normal

def extract_neighbors(point, window_size, shape):
    y, x = point
    half = window_size // 2
    y_min, y_max = max(0, y - half), min(shape[0], y + half + 1)
    x_min, x_max = max(0, x - half), min(shape[1], x + half + 1)
    return np.array([[yi, xi] for yi in range(y_min, y_max) for xi in range(x_min, x_max)])

def sample_edges_along_normal(center, normal, mask, max_distance=20):
    y0, x0 = center
    edges = []
    for sign in [-1, 1]:
        for d in range(1, max_distance):
            dx, dy = normal[1] * d * sign, normal[0] * d * sign
            xi, yi = int(round(x0 + dx)), int(round(y0 + dy))
            if 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1]:
                if mask[yi, xi] == 0:
                    edges.append((xi, yi))
                    break
    return edges

def calculate_crack_width(edge_points):
    if len(edge_points) == 2:
        (x1, y1), (x2, y2) = edge_points
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return None

def max_inscribed_circle(mask, center, window=25):
    # Use local distance transform to get the largest circle centered at `center`
    y, x = center
    half = window // 2
    y0, y1 = max(0, y-half), min(mask.shape[0], y+half+1)
    x0, x1 = max(0, x-half), min(mask.shape[1], x+half+1)
    submask = mask[y0:y1, x0:x1] > 0
    dist = distance_transform_edt(submask)
    max_radius = np.max(dist)
    return max_radius * 2  # Diameter

# ---------------------- Crack Width Methods ---------------------

def adaptive_pca_width(mask):
    skeleton = get_skeleton(mask)
    crack_widths = []
    points = np.argwhere(skeleton > 0)
    for (y, x) in points:
        neighbors = extract_neighbors((y, x), 15, mask.shape)
        local_skel = skeleton[neighbors[:, 0], neighbors[:, 1]]
        active = neighbors[local_skel > 0]
        if len(active) < 3:
            continue
        tangent, normal = perform_pca(active)
        edge_pts = sample_edges_along_normal((y, x), normal, mask)
        width = calculate_crack_width(edge_pts)
        if width is not None:
            crack_widths.append(((x, y), width))
    return crack_widths, skeleton

def sobel_width(mask):
    skeleton = get_skeleton(mask)
    crack_widths = []
    points = np.argwhere(skeleton > 0)
    for (y, x) in points:
        normal = sobel_normal(mask, y, x)
        edge_pts = sample_edges_along_normal((y, x), normal, mask)
        width = calculate_crack_width(edge_pts)
        if width is not None:
            crack_widths.append(((x, y), width))
    return crack_widths, skeleton

'''def medial_axis_width(mask):
    skel, dist = get_medial_axis(mask)
    crack_widths = []
    points = np.argwhere(skel > 0)
    for (y, x) in points:
        width = dist[y, x] * 2
        crack_widths.append(((x, y), width))
    return crack_widths, skel

def inscribed_circle_width(mask):
    skeleton = get_skeleton(mask)
    crack_widths = []
    points = np.argwhere(skeleton > 0)
    for (y, x) in points:
        width = max_inscribed_circle(mask, (y, x), window=25)
        crack_widths.append(((x, y), width))
    return crack_widths, skeleton'''

def medial_axis_width(mask):
    skel, dist = get_medial_axis(mask)
    crack_widths = []
    points = np.argwhere(skel > 0)
    for (y, x) in points:
        width = dist[y, x] * 2
        normal = (1, 0)  # placeholder, as above
        crack_widths.append(((x, y), normal, width))
    return crack_widths, skel

def inscribed_circle_width(mask):
    skeleton = get_skeleton(mask)
    crack_widths = []
    points = np.argwhere(skeleton > 0)
    for (y, x) in points:
        width = max_inscribed_circle(mask, (y, x), window=25)
        normal = (1, 0)  # placeholder
        crack_widths.append(((x, y), normal, width))
    return crack_widths, skeleton

# ---------------------- Visualization & Metrics -----------------

'''def draw_width_overlay(mask, crack_widths, skeleton, out_path, method_name):
    # Step 1: Overlay with color-mapped width lines
    overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    if not crack_widths:
        cv2.imwrite(str(out_path), overlay)
        return
    widths = np.array([w for _, w in crack_widths])
    norm = plt.Normalize(vmin=np.percentile(widths, 2), vmax=np.percentile(widths, 98))
    cmap = plt.get_cmap('coolwarm_r')
    for ((cx, cy), width) in crack_widths:
        color = cmap(norm(width))[:3]
        color = tuple(int(255 * c) for c in color[::-1])
        cv2.circle(overlay, (int(cx), int(cy)), 3, color, -1)

    # Step 2: Create a skeleton image (green)
    skel_img = np.zeros_like(overlay)
    ys, xs = np.where(skeleton > 0)
    for x, y in zip(xs, ys):
        cv2.circle(skel_img, (x, y), 1, (0, 255, 0), -1)

    # Step 3: Blend with alpha (0.5 for skeleton, 1.0 for overlay)
    out = cv2.addWeighted(skel_img, 0.5, overlay, 1.0, 0)
    cv2.imwrite(str(out_path), out)'''

def draw_width_overlay(mask, crack_widths, skeleton, out_path, method_name, draw_lines_every=10):
    overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    if not crack_widths:
        cv2.imwrite(str(out_path), overlay)
        return

    # Accept both 2- or 3-tuple per width entry
    if len(crack_widths[0]) == 3:
        widths = np.array([w for _, _, w in crack_widths])
    else:
        widths = np.array([w for _, w in crack_widths])

    norm = plt.Normalize(vmin=np.percentile(widths, 2), vmax=np.percentile(widths, 98))
    cmap = plt.get_cmap('coolwarm_r')

    for i, cw in enumerate(crack_widths):
        if len(cw) == 3:
            (cx, cy), normal, width = cw
        else:
            (cx, cy), width = cw
            normal = None  # Can't draw width lines

        color = cmap(norm(width))[:3]
        color = tuple(int(255 * c) for c in color[::-1])
        cv2.circle(overlay, (int(cx), int(cy)), 2, color, -1)

        # Only draw width line if normal is available
        if normal is not None and i % draw_lines_every == 0:
            perp = np.array([-normal[1], normal[0]], dtype=np.float32)
            perp /= np.linalg.norm(perp) + 1e-6
            half = width / 2
            pt1 = (int(cx + perp[0] * half), int(cy + perp[1] * half))
            pt2 = (int(cx - perp[0] * half), int(cy - perp[1] * half))
            cv2.line(overlay, pt1, pt2, color, 1)

    # Blend skeleton underneath for context
    skel_img = np.zeros_like(overlay)
    ys, xs = np.where(skeleton > 0)
    for x, y in zip(xs, ys):
        cv2.circle(skel_img, (x, y), 1, (0, 255, 0), -1)
    out = cv2.addWeighted(skel_img, 0.5, overlay, 1.0, 0)
    cv2.imwrite(str(out_path), out)


'''def save_widths_csv(crack_widths, csv_path):
    rows = []
    for (x, y), width in crack_widths:
        rows.append({'x': x, 'y': y, 'width_px': width})
    pd.DataFrame(rows).to_csv(csv_path, index=False)'''

def save_widths_csv(crack_widths, csv_path):
    rows = []
    for entry in crack_widths:
        if len(entry) == 3:
            (x, y), normal, width = entry
            nx, ny = normal
        elif len(entry) == 2:
            (x, y), width = entry
            nx, ny = None, None
        else:
            raise ValueError("Unexpected entry format in crack_widths!")
        rows.append({'x': x, 'y': y, 'normal_x': nx, 'normal_y': ny, 'width_px': width})
    pd.DataFrame(rows).to_csv(csv_path, index=False)

'''def width_stats(crack_widths):
    if crack_widths:
        w = [w for _, w in crack_widths]
        return {'mean': np.mean(w), 'min': np.min(w), 'max': np.max(w), 'std': np.std(w)}
    else:
        return {'mean': 0, 'min': 0, 'max': 0, 'std': 0}'''

def width_stats(crack_widths):
    if crack_widths:
        # Accepts both ((x, y), width) and ((x, y), normal, width)
        w = [cw[-1] for cw in crack_widths]
        return {'mean': np.mean(w), 'min': np.min(w), 'max': np.max(w), 'std': np.std(w)}
    else:
        return {'mean': 0, 'min': 0, 'max': 0, 'std': 0}


# ---------------------- Main Batch Evaluation -------------------

'''input_folder = r"D:\camerer_ml\datasets\concrete3k\concrete3k\labels_01_visible"   # Folder of .png or .jpg binary mask images
output_folder = Path("output_crack_eval")
output_folder.mkdir(exist_ok=True)

methods = {
    'adaptive_pca': adaptive_pca_width,
    'sobel': sobel_width,
    'medial_axis': medial_axis_width,
    'inscribed_circle': inscribed_circle_width
}

metrics = {}

for fname in os.listdir(input_folder):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        continue
    img_path = os.path.join(input_folder, fname)
    mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if mask.max() > 1:
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    print(f"[INFO] Processing {fname}...")

    for method, func in methods.items():
        crack_widths, skeleton = func(mask)
        # Output overlay image
        out_img = os.path.join(output_folder, f"{fname[:-4]}_{method}_overlay.png")
        draw_width_overlay(mask, crack_widths, skeleton, out_img, method)
        # Output CSV
        out_csv = os.path.join(output_folder, f"{fname[:-4]}_{method}_widths.csv")
        save_widths_csv(crack_widths, out_csv)
        # Collect stats
        stat = width_stats(crack_widths)
        metrics.setdefault(method, []).append({'image': fname, **stat})

# Save all metrics to one CSV for easy review
all_stats = []
for method, lst in metrics.items():
    for item in lst:
        item['method'] = method
        all_stats.append(item)
pd.DataFrame(all_stats).to_csv(output_folder / "all_width_stats.csv", index=False)
print("[✓] Evaluation complete. All results saved to", output_folder)'''


def process_image(fname):
    img_path = os.path.join(input_folder, fname)
    mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if mask.max() > 1:
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    image_metrics = []
    img_stem = Path(fname).stem
    img_outdir = output_folder / img_stem
    img_outdir.mkdir(exist_ok=True)
    for method, func in methods.items():
        crack_widths, skeleton = func(mask)
        # Overlay image (widths in color)
        out_img = img_outdir / f"{img_stem}_{method}_overlay.png"
        draw_width_overlay(mask, crack_widths, skeleton, out_img, method)
        # Width CSV
        out_csv = img_outdir / f"{img_stem}_{method}_widths.csv"
        save_widths_csv(crack_widths, out_csv)
        # Metrics
        stat = width_stats(crack_widths)
        stat.update({'image': fname, 'method': method})
        image_metrics.append(stat)
    return image_metrics

# --- Main script ---
input_folder = r"D:\camerer_ml\datasets\concrete3k\concrete3k\labels_01_visible"
output_folder = Path("output_crack_eval")
output_folder.mkdir(exist_ok=True)

methods = {
    'adaptive_pca': adaptive_pca_width,
    'sobel': sobel_width,
    'medial_axis': medial_axis_width,
    'inscribed_circle': inscribed_circle_width
}


image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))][:50]
if __name__ == "__main__":
    all_stats = []
    with ProcessPoolExecutor() as executor:
        for image_metrics in executor.map(process_image, image_files):
            all_stats.extend(image_metrics)

    pd.DataFrame(all_stats).to_csv(output_folder / "all_width_stats.csv", index=False)
    print("[✓] Parallel evaluation complete. Results saved to:", output_folder)