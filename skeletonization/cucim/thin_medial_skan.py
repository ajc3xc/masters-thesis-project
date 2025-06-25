import time
from pathlib import Path

import numpy as np
import imageio.v3 as iio
from skimage.morphology import skeletonize as cpu_skeletonize

import cupy as cp
from cucim.skimage.morphology import thin, medial_axis

from scipy.ndimage import distance_transform_edt, binary_dilation
from skan import csr

import matplotlib.pyplot as plt
import pandas as pd

# ——— CONFIG —————————————————————————
image_paths = [
    "/mnt/d/camerer_ml/datasets/METU_Concrete_Crack_Seg/BW_visible/549.png",
]
outdir = Path("./results_combined")
outdir.mkdir(exist_ok=True, parents=True)
min_branch_length = 0  # pixels: prune shorter branches
# ————————————————————————————————————

timings = []

# 0) WARM-UP: compile CUDA & Numba kernels
_ = thin(cp.zeros((10, 10), bool), max_num_iter=1)
_ = medial_axis(cp.zeros((10, 10), bool))
dummy = np.zeros((10, 10), bool)
dummy[5, 5:7] = True  # Two connected pixels
_ = csr.Skeleton(dummy)
cp.cuda.Stream.null.synchronize()

for img_path in image_paths:
    name = Path(img_path).stem

    # 1) Load & binarize
    img = iio.imread(img_path, mode='L') / 255
    bw = (img > 0.5)

    # 2) Compute original distance transform (CPU)
    edt = distance_transform_edt(bw)

    # ——— CPU PIPELINE ——————————————————————
    t0 = time.perf_counter()
    skel_cpu = cpu_skeletonize(bw)
    t_skel_cpu = time.perf_counter() - t0

    # Prune with Skan
    t0 = time.perf_counter()
    graph = csr.Skeleton(skel_cpu)  # skel_cpu is your 1-pixel skeleton
    deg = graph.degrees  # 1D array aligned with graph.coordinates

    # Build mapping from coords → index
    coords_arr = graph.coordinates.astype(int)
    coord_to_idx = {tuple(c): i for i, c in enumerate(coords_arr)}

    pruned_cpu = np.zeros_like(skel_cpu, dtype=bool)
    H, W = pruned_cpu.shape

    for path, L in zip(graph.paths_list(), graph.path_lengths()):
        path = np.array(path)
        if path.ndim != 1 or path.size < 2:
            print(path.ndim, path.size, "skipping malformed path")
            continue  # skip single-point or malformed path
        rows, cols = np.unravel_index(path, skel_cpu.shape)
        start = (rows[0], cols[0])
        end = (rows[-1], cols[-1])
        i_s = coord_to_idx.get(start)
        i_e = coord_to_idx.get(end)
        if i_s is None or i_e is None:
            print(start, end, i_s, i_e, "start or end not found in coord_to_idx")
            continue
        d_s, d_e = deg[i_s], deg[i_e]
        is_spur = ((d_s == 1 and d_e >= 3) or (d_e == 1 and d_s >= 3))
        if not (is_spur and L < min_branch_length):
            pruned_cpu[rows, cols] = True
    t_skan_cpu = time.perf_counter() - t0
    print(np.sum(pruned_cpu), "pixels in pruned CPU skeleton")
    import sys; sys.exit(0)

    # total CPU time
    t_cpu_total = t_skel_cpu + t_skan_cpu

    # ——— GPU PIPELINE (thin20 → medial_axis → skan) ————
    '''A_gpu = cp.array(bw)

    # --- GPU PIPELINE (thin20 → medial_axis) ---
    t0 = time.perf_counter()
    skel20_gpu = thin(A_gpu, max_num_iter=20)
    cp.cuda.Stream.null.synchronize()
    t_thin20 = time.perf_counter() - t0

    t0 = time.perf_counter()
    ma20_gpu = medial_axis(skel20_gpu.astype(bool))
    cp.cuda.Stream.null.synchronize()
    t_ma20 = time.perf_counter() - t0

    # bring back to host
    ma20 = ma20_gpu.get().astype(bool)

    # prune with Skan
    t0 = time.perf_counter()
    graph20 = csr.Skeleton(ma20)
    paths20 = graph20.paths_list()
    lengths20 = graph20.path_lengths()
    pruned20 = np.zeros_like(ma20)
    H, W = pruned20.shape
    for path, L in zip(paths20, lengths20):
        if L >= min_branch_length:
            coords = np.array(path).T
            in_bounds = (
                (coords[0] >= 0) & (coords[0] < H) &
                (coords[1] >= 0) & (coords[1] < W)
            )
            pruned20[coords[0][in_bounds], coords[1][in_bounds]] = True
    t_skan20 = time.perf_counter() - t0

    t_gpu_morph = t_thin20 + t_ma20
    t_gpu_total = t_gpu_morph + t_skan20'''

    # ——— BUILD WIDTH MAPS —————————————————————
    width_cpu = np.zeros_like(edt)
    width_cpu[pruned_cpu] = 2 * edt[pruned_cpu]

    #width20 = np.zeros_like(edt)
    #width20[pruned20] = 2 * edt[pruned20]

    # ——— PLOT SIDE-BY-SIDE ————————————————————
    def overlay_colored_skeleton(ax, img, mask, widthmap, title):
        ax.imshow(img, cmap='gray')
        dil = binary_dilation(mask, iterations=2)
        if mask.any():
            dist_to_skel, nearest_idx = distance_transform_edt(~mask, return_indices=True)
            vis = np.zeros_like(widthmap)
            coords = tuple(idx[dil] for idx in nearest_idx)
            vis[dil] = widthmap[coords]
        else:
            vis = np.zeros_like(widthmap)
        # Rescale width for colormap if needed
        masked = np.ma.masked_where(vis == 0, vis)
        im = ax.imshow(masked, cmap='jet', alpha=0.9, interpolation='nearest')
        ax.set_title(title)
        ax.axis('off')
        return im

    plt.figure(figsize=(8,8))
    plt.imshow(img, cmap='gray')
    skan_vis = binary_dilation(pruned_cpu, iterations=2)
    plt.imshow(np.ma.masked_where(~skan_vis, skan_vis) * 255, cmap='Greens', alpha=0.8)
    plt.title('CPU skeleton + Skan pruning (dilated)')
    plt.axis('off')
    plt.show()
    
    '''fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    im0 = overlay_colored_skeleton(axs[0], img, pruned20, width20, f"GPU thin20+MA+prune ({t_gpu_total:.3f}s)")
    im1 = overlay_colored_skeleton(axs[1], img, pruned_cpu, width_cpu, f"CPU skel+prune ({t_cpu_total:.3f}s)")

    cbar = fig.colorbar(im0, ax=axs.ravel().tolist(), fraction=0.046, pad=0.04)
    cbar.set_label("Width (pixels)")
    plt.tight_layout()
    fig.savefig(outdir / f"{name}_thin20_vs_cpu_skan.png", dpi=600,
                bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # ——— RECORD TIMINGS ——————————————————————
    timings.append({
        "image": name,
        "method": "cpu_skel+skan",
        "t_total": round(t_cpu_total, 4)
    })
    timings.append({
        "image": name,
        "method": "gpu_thin20+MA",
        "t_total": round(t_gpu_morph, 4)
    })
    timings.append({
        "image": name,
        "method": "gpu_thin20+MA+skan",
        "t_total": round(t_gpu_total, 4)
    })

# ——— SUMMARY —————————————————————————
df = pd.DataFrame(timings)
print(df.pivot_table(index="image", columns="method", values="t_total"))
df.to_csv(outdir / "timings.csv", index=False)

print(f"Results and timings saved to {outdir}")'''
