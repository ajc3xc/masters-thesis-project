import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import medial_axis
from skimage.measure import profile_line
from skimage.draw import disk
from scipy.ndimage import distance_transform_edt, binary_dilation, gaussian_filter
from time import time

def create_synthetic_crack(shape=(256, 256), width=6):
    img = np.zeros(shape, dtype=np.uint8)
    rr, cc = np.arange(30, 220), np.arange(30, 220)
    img[rr, rr + width // 2] = 1
    img[rr, rr - width // 2] = 1
    for i in range(-width // 2, width // 2 + 1):
        img[rr, rr + i] = 1
    # Add some smoothing to mimic real cracks
    img = gaussian_filter(img.astype(float), sigma=1) > 0.5
    return img

def get_skeleton_and_dist(binary_img):
    start = time()
    skeleton, dist = medial_axis(binary_img, return_distance=True)
    dt = distance_transform_edt(binary_img)
    elapsed = time() - start
    print(f"Medial axis + distance transform took {elapsed:.3f} seconds")
    return skeleton, dist, dt

def estimate_widths_medial(skeleton, dist):
    # Medial axis: width = 2 * distance at skeleton pixels
    widths = np.zeros_like(skeleton, dtype=float)
    widths[skeleton] = dist[skeleton] * 2
    return widths

def get_local_normal(skeleton, y, x, window=3):
    y0, x0 = max(y-window,0), max(x-window,0)
    y1, x1 = min(y+window+1, skeleton.shape[0]), min(x+window+1, skeleton.shape[1])
    region = skeleton[y0:y1, x0:x1]
    yy, xx = np.nonzero(region)
    coords = np.stack([yy, xx], axis=1)
    if len(coords) < 2:
        return np.array([1, 0])
    # Fit line (PCA): direction vector is first eigenvector
    coords = coords - coords.mean(axis=0)
    u, s, vh = np.linalg.svd(coords, full_matrices=False)
    direction = vh[0]
    # Normal is perpendicular to direction
    normal = np.array([-direction[1], direction[0]])
    normal /= np.linalg.norm(normal) + 1e-8
    return normal

from scipy.signal import find_peaks

def estimate_widths_subpixel(binary_img, skeleton, length=8, plot_profiles=False):
    ys, xs = np.nonzero(skeleton)
    widths_subpixel = []
    positions = []
    start = time()
    for y, x in zip(ys, xs):
        normal = get_local_normal(skeleton, y, x)
        # Sample along the normal
        p1 = (y + normal[0]*length, x + normal[1]*length)
        p2 = (y - normal[0]*length, x - normal[1]*length)
        # Sample the line as float, order=1 (subpixel)
        profile = profile_line(binary_img.astype(np.float32), p1, p2, linewidth=1, order=1, mode='constant')
        # Find where the profile crosses 0.5 (edge)
        above = profile > 0.5
        crossing = np.where(np.diff(above.astype(int)) != 0)[0]
        if len(crossing) >= 2:
            left, right = crossing[0], crossing[-1]
            width = np.abs(right - left)
            widths_subpixel.append(width)
            positions.append((y, x))
            if plot_profiles:
                plt.plot(profile)
                plt.title(f"Profile at {(y,x)}, width={width:.2f}")
                plt.show()
        # Optionally, skip points with invalid widths
    elapsed = time() - start
    print(f"Subpixel profile fitting for {len(widths_subpixel)} points took {elapsed:.3f} seconds")
    return np.array(positions), np.array(widths_subpixel)

# --- Use either a real image or synthetic crack ---
# img_path = 'your_file_here.png'
# img = imread(img_path, mode='L')
# bw = (img > 0.5)
bw = create_synthetic_crack()

skeleton, dist, dt = get_skeleton_and_dist(bw)

# Medial axis method
widths_medial = estimate_widths_medial(skeleton, dist)

# Subpixel method
positions, widths_subpixel = estimate_widths_subpixel(bw, skeleton, length=8)

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Make sure input is numeric
axes[0].imshow(bw.astype(np.uint8), cmap='gray')
axes[0].set_title('Binary Input')
axes[0].axis('off')

# Stack overlay as float32 for imshow compatibility
skel_overlay = np.stack([
    bw.astype(np.float32),                # Red channel: input
    skeleton.astype(np.float32),          # Green channel: skeleton
    np.zeros_like(bw, dtype=np.float32)   # Blue channel: zeros
], axis=2)
axes[1].imshow(skel_overlay)
axes[1].set_title('Medial Axis (Red Overlay)')
axes[1].axis('off')

# Plot widths as scatter on skeleton
ax = axes[2]
ax.imshow(bw.astype(np.uint8), cmap='gray')
ys, xs = np.nonzero(skeleton)
if np.any(skeleton):
    im1 = ax.scatter(xs, ys, c=widths_medial[skeleton], cmap='viridis', s=15, label='Medial Width')
if len(positions) > 0:
    ax.scatter([p[1] for p in positions], [p[0] for p in positions], c=widths_subpixel, cmap='plasma', marker='+', s=10, label='Subpixel Width')
ax.set_title('Widths: Medial (dots), Subpixel (crosses)')
ax.axis('off')
if np.any(skeleton):
    fig.colorbar(im1, ax=axes[2], label='Width (pixels)', orientation='vertical')

plt.tight_layout()
plt.show()

