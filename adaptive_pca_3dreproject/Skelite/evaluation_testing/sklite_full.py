import os
import time
import torch
import imageio.v2 as imageio
import numpy as np
import cv2
import matplotlib.pyplot as plt

from skimage.morphology import skeletonize, skeletonize_3d
from scipy.ndimage import convolve
from skimage.measure import label
from scipy.ndimage import distance_transform_edt
from concurrent.futures import ThreadPoolExecutor, as_completed

# =============================================================================
#  Helper functions
# =============================================================================

def load_and_prep_image(image_path: str, device: torch.device):
    """
    Load an image (RGB or grayscale), convert to float32 [0,1], return both a
    torch.Tensor (1×1×H×W) on device and the raw NumPy [0,1] array.
    """
    img = imageio.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not open {image_path}")
    if img.ndim == 3:
        img = img.mean(axis=2)  # to grayscale
    img = img.astype(np.float32) / 255.0
    H, W = img.shape
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
    return tensor, img

def run_skelite(model: torch.jit.ScriptModule, img_tensor: torch.Tensor) -> np.ndarray:
    """
    Run Skelite model on img_tensor (1×1×H×W). Return raw float mask (H, W) in [0,1].
    """
    with torch.no_grad():
        output, _ = model(img_tensor, z=None, no_iter=5, val_mode=True)
    raw_mask = output[0, 0].cpu().numpy()
    return raw_mask

def prune_branch_from_endpoint(skel_u8: np.ndarray, start_xy: tuple, max_length: int):
    """
    Walk from an endpoint (y0, x0) up to max_length. If the path has length <= max_length
    before hitting a junction or break, return that path as a list of coords to remove. Else return [].
    """
    H, W = skel_u8.shape
    y0, x0 = start_xy
    path = [(y0, x0)]
    y, x = y0, x0
    skel_copy = skel_u8  # read-only; modifications are done by caller

    for _ in range(max_length):
        # find 8-connected neighbors of (y, x)
        neighbors = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and skel_copy[ny, nx]:
                    neighbors.append((ny, nx))
        if len(neighbors) != 1:
            # reached a junction (len > 1) or dead-end (len = 0)
            break
        y, x = neighbors[0]
        path.append((y, x))
        # mark current pixel as visited
        # but do not modify the original array here—just track path
    # If we terminated because path length > max_length (i.e., we did full loop max_length times
    # and still had exactly one neighbor each step), actually path length is max_length+1
    # So if len(path) <= max_length, mark for deletion; else keep.
    if len(path) <= max_length:
        return path
    else:
        return []

def prune_branches_parallel(skel: np.ndarray, max_length: int = 10, n_workers: int = 4) -> np.ndarray:
    """
    Remove all branches ≤ max_length from a binary skeleton (bool or 0/1).
    Uses ThreadPoolExecutor to process endpoints in parallel.
    Returns a pruned binary skeleton.
    """
    sk = skel.astype(np.uint8).copy()
    H, W = sk.shape

    # 1) find endpoints: pixel has exactly 1 neighbor
    ep_kernel = np.array([[1,1,1],
                          [1,10,1],
                          [1,1,1]])
    filtered = convolve(sk, ep_kernel, mode='constant')
    endpoints = np.argwhere(filtered == 11)

    # 2) For each endpoint, walk up to max_length. If path <= max_length, mark pixels for removal.
    # Use a shared "removal set" (thread-safe accumulation).
    removal_mask = np.zeros_like(sk, dtype=bool)

    def worker(endpoint):
        return prune_branch_from_endpoint(sk, tuple(endpoint), max_length)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(worker, ep): tuple(ep) for ep in endpoints}
        for fut in as_completed(futures):
            path = fut.result()
            if path:
                for (yy, xx) in path:
                    removal_mask[yy, xx] = True

    # 3) Remove all marked pixels
    pruned = sk.copy()
    pruned[removal_mask] = 0
    return pruned.astype(bool)

def print_metrics(name: str, skel: np.ndarray):
    """
    Print connected components, endpoints, mean thickness, and pixel count of a binary skeleton.
    """
    comps = int(label(skel).max())
    # count endpoints
    ep_kernel = np.array([[1,1,1],
                          [1,10,1],
                          [1,1,1]])
    end_pts = np.sum(convolve(skel.astype(np.uint8), ep_kernel, mode='constant') == 11)
    pix = int(skel.sum())
    # mean thickness: distance from skeleton to background
    if pix > 0:
        dist = distance_transform_edt(skel)
        mean_th = float(dist[skel > 0].mean())
    else:
        mean_th = 0.0

    print(f"{name}: components={comps}, endpoints={end_pts}, mean_thickness={mean_th:.3f}, pixels={pix}")

# =============================================================================
#  Main Script
# =============================================================================

if __name__ == "__main__":
    # Paths (modify to your environment)
    image_path = r"D:\camerer_ml\datasets\METU_Concrete_Crack_Seg\BW_visible\148.png"
    model_path = r"D:\camerer_ml\adaptive_pca_3dreproject\Skelite\skelite_scripted.pt"
    output_dir = r"D:\camerer_ml\adaptive_pca_3dreproject\Skelite\parallel_prune_output"
    os.makedirs(output_dir, exist_ok=True)

    # 1) Load Skelite (TorchScript)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sk_model = torch.jit.load(model_path, map_location=device)
    sk_model.eval()

    # 2) Load & prep image
    img_tensor, img = load_and_prep_image(image_path, device)
    H, W = img.shape
    bw = (img > 0.5)  # original binary mask (for reference)

    # 3) Warm-up Skelite on GPU (5–10 iterations, no timing)
    print("Warming up Skelite GPU...")
    for _ in range(8):
        _ = run_skelite(sk_model, img_tensor)

    # 4) Actual Skelite inference timing
    t0 = time.time()
    raw_mask = run_skelite(sk_model, img_tensor)
    t_skelite = time.time() - t0
    print(f"Skelite inference time (val_mode=True, no cleanup): {t_skelite:.3f}s")

    # 5) Threshold raw_mask to get binary Skelite mask
    skel_mask = (raw_mask > 0.5)

    # 6) Pre-prune branches on skel_mask with max_length_pre = 15
    t0 = time.time()
    skel_pre_pruned = prune_branches_parallel(skel_mask, max_length=15, n_workers=4)
    t_pre_prune = time.time() - t0
    print(f"Pre-prune time (max_length=15): {t_pre_prune:.3f}s")

    # 7A) Skeletonize using original skimage.skeletonize
    t0 = time.time()
    skel_std = skeletonize(skel_pre_pruned)
    t_skel_std = time.time() - t0
    print(f"   skeletonize (skimage) time: {t_skel_std:.3f}s")

    # 7B) Skeletonize using skeletonize_3d
    t0 = time.time()
    skel_3d = skeletonize_3d(skel_pre_pruned)
    t_skel_3d = time.time() - t0
    print(f"   skeletonize_3d time: {t_skel_3d:.3f}s")

    # 8) Post-prune both skeletons with max_length_post = 10
    t0 = time.time()
    skel_std_pruned = prune_branches_parallel(skel_std, max_length=10, n_workers=4)
    t_post_prune_std = time.time() - t0
    print(f"   Post-prune (skeletonize) time (max_length=10): {t_post_prune_std:.3f}s")

    t0 = time.time()
    skel_3d_pruned = prune_branches_parallel(skel_3d, max_length=10, n_workers=4)
    t_post_prune_3d = time.time() - t0
    print(f"   Post-prune (skeletonize_3d) time (max_length=10): {t_post_prune_3d:.3f}s")

    # 9) Print metrics for each final skeleton
    print_metrics("Final (skeletonize→prune)", skel_std_pruned)
    print_metrics("Final (skeletonize_3d→prune)", skel_3d_pruned)

    # 10) Save intermediate and final images
    cv2.imwrite(os.path.join(output_dir, "01_raw_mask.png"), (raw_mask * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(output_dir, "02_skel_thresh.png"), (skel_mask.astype(np.uint8) * 255))
    cv2.imwrite(os.path.join(output_dir, "03_pre_pruned.png"), (skel_pre_pruned.astype(np.uint8) * 255))
    cv2.imwrite(os.path.join(output_dir, "04_skel_std.png"), (skel_std.astype(np.uint8) * 255))
    cv2.imwrite(os.path.join(output_dir, "05_skel_3d.png"), (skel_3d.astype(np.uint8) * 255))
    cv2.imwrite(os.path.join(output_dir, "06_std_pruned.png"), (skel_std_pruned.astype(np.uint8) * 255))
    cv2.imwrite(os.path.join(output_dir, "07_3d_pruned.png"), (skel_3d_pruned.astype(np.uint8) * 255))

    # 11) Visualization
    plt.figure(figsize=(20, 6))
    titles = [
        "Original Grayscale", 
        "Raw Skelite Mask>0.5", 
        "Pre-Pruned (≤15)", 
        "Skeletonize (std) → Pruned(≤10)", 
        "Skeletonize_3D → Pruned(≤10)"
    ]
    images = [
        img,
        skel_mask,
        skel_pre_pruned.astype(np.float32),
        skel_std_pruned.astype(np.float32),
        skel_3d_pruned.astype(np.float32)
    ]
    for i, im in enumerate(images):
        plt.subplot(1, 5, i+1)
        plt.imshow(im, cmap="gray", vmin=0, vmax=1)
        plt.title(titles[i], fontsize=12)
        plt.axis("off")
    plt.tight_layout()
    plt.show()