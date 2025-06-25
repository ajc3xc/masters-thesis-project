import time
from pathlib import Path

import numpy as np
import imageio.v3 as iio

from skimage.morphology import skeletonize as cpu_skeletonize

import cupy as cp
from cucim.skimage.morphology import medial_axis, thin
import imageio

# ——— CONFIG —————————————————————————
# List your binary-image paths here:
image_paths = [
    #"/mnt/d/camerer_ml/datasets/METU_Concrete_Crack_Seg/BW_visible/064.png",
    #"/mnt/d/camerer_ml/datasets/METU_Concrete_Crack_Seg/BW_visible/236.png",
    #"/mnt/d/camerer_ml/datasets/METU_Concrete_Crack_Seg/BW_visible/251.png",
    "/mnt/e/camerer_ml/datasets/METU_Concrete_Crack_Seg/BW_visible/549.png",
    #"/mnt/d/camerer_ml/datasets/METU_Concrete_Crack_Seg/BW_visible/562.png",
]

# How many thinning iterations to try (None means “full”):
#max_iters_list = [5, 10, 20, None]

# Where to dump outputs
outdir = Path("./results_combined")
outdir.mkdir(exist_ok=True, parents=True)
# ————————————————————————————————————

# Table of timings
timings = []

from scipy import ndimage
from scipy.ndimage import binary_dilation, distance_transform_edt

# --- GPU warmup for accurate timings ---
dummy = cp.zeros((3024, 3024), dtype=bool)
_ = thin(dummy, max_num_iter=1)
_ = medial_axis(dummy)
cp.cuda.Stream.null.synchronize()

for img_path in image_paths:
    import matplotlib.pyplot as plt

    name = Path(img_path).stem
    # 1) Load & binarize
    img = iio.imread(img_path, mode='L') / 255
    bw = (img > 0.5).astype(bool)

    # Compute EDT of original (for width estimation!)
    edt = distance_transform_edt(bw)

    # --- CPU: skeletonize (Zhang-Suen) ---
    t0 = time.perf_counter()
    skel_cpu = cpu_skeletonize(bw)
    t_cpu = time.perf_counter() - t0

    # --- GPU: thinning variants + medial axis after thinning ---
    #thin_iters = [5, 10, 20, 30, 40, 50, None]
    thin_iters = [20]
    skel_thins = {}
    ma_after_thin = {}
    timings_local = {}

    ma_after_thin = {}
    timings_local = {}

    A_gpu = cp.array(bw)
    for m in thin_iters:
        label = f"thin{m or 'full'}"
        t0 = time.perf_counter()
        skel_ref_gpu = thin(A_gpu, max_num_iter=m)
        cp.cuda.Stream.null.synchronize()
        t_thin = time.perf_counter() - t0
        skel_ref = skel_ref_gpu.get()
        skel_thins[label] = skel_ref
        timings_local[label] = t_thin

        # Only call medial_axis for partial thinning (not full thinning)
        if 1==1:
            t0 = time.perf_counter()
            ma_gpu = medial_axis(cp.array(skel_ref))
            cp.cuda.Stream.null.synchronize()
            t_ma = time.perf_counter() - t0
            ma_after_thin[label] = ma_gpu.get()
            timings_local[label] += t_ma
        else:
            ma_after_thin[label] = None

    # --- Build widthmaps for overlay (color MA after thinning by original width) ---
    widthmaps = {}
    width_cpu = np.zeros_like(edt)
    width_cpu[skel_cpu] = 2 * edt[skel_cpu]
    widthmaps['cpu'] = width_cpu

    for label, ma in ma_after_thin.items():
        if ma is not None:
            wm = np.zeros_like(edt)
            wm[ma.astype(bool)] = 2 * edt[ma.astype(bool)]
            widthmaps[label] = wm
        else:
            widthmaps[label] = np.zeros_like(edt)  # or skip overlay for full-thin

    # Plot overlays: each MA-after-thin colored by its true width
    methods_bin = [f"thin{m or 'full'}" for m in thin_iters] + ['cpu']
    n = len(methods_bin)
    fig, axs = plt.subplots(1, n, figsize=(4*n, 8))
    titles = [f"MA after Thin {m or 'full'}" for m in thin_iters] + ["CPU Skeleton"]

    for i, key in enumerate(methods_bin):
        ax = axs[i]
        ax.imshow(img, cmap='gray')

        if key == 'cpu':
            skel = skel_cpu
            widthmap = widthmaps[key]
        else:
            ma = ma_after_thin[key]
            skel = ma.astype(bool) if ma is not None else np.zeros_like(bw)
            widthmap = widthmaps[key]

        dilated = binary_dilation(skel, iterations=5)
        if np.any(skel):
            dist_to_skel, nearest_idx = distance_transform_edt(~skel, return_indices=True)
            vis_width = np.zeros_like(widthmap)
            vis_width[dilated] = widthmap[tuple(i[dilated] for i in nearest_idx)]
        else:
            vis_width = np.zeros_like(widthmap)

        masked = np.ma.masked_where(vis_width == 0, vis_width)
        im = ax.imshow(masked, cmap='jet', alpha=0.9, interpolation='nearest')
        ax.axis('off')
        ax.set_title(titles[i])

        if i == n-1:
            cbar = fig.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)
            cbar.set_label("Width (pixels)")

    plt.tight_layout()
    fig.savefig(outdir / f"{name}_ma_after_thin_width_overlay.png", dpi=900,
                bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # Second plot: thinning skeletons directly
    fig2, axs2 = plt.subplots(1, n, figsize=(4*n, 8))
    titles2 = [f"Thin {m or 'full'}" for m in thin_iters] + ["CPU Skeleton"]
    for i, key in enumerate(methods_bin):
        ax = axs2[i]
        ax.imshow(img, cmap='gray')

        if key == 'cpu':
            skel = skel_cpu
        else:
            skel = skel_thins[key]

        dilated = binary_dilation(skel, iterations=5)
        ax.imshow(dilated, cmap='hot', alpha=0.8)
        ax.axis('off')
        ax.set_title(titles2[i])

    plt.tight_layout()
    fig2.savefig(outdir / f"{name}_thinning_results.png", dpi=900,
                 bbox_inches='tight', pad_inches=0)
    plt.close(fig2)

    # 7) (Optional) Save individual binaries if you still want them:
    #imageio.imwrite(outdir / f"{name}_cpu_skel.png", (skel_cpu*255).astype(np.uint8))
    #imageio.imwrite(outdir / f"{name}_ma_skel.png", (skel_ma*255).astype(np.uint8))
    #for label, skel in skel_thins.items():
    #    imageio.imwrite(outdir / f"{name}_{label}.png", (skel*255).astype(np.uint8))

    # 8) Record timings
    timings.append({
        "image": name,
        "method": "skimage.cpu_skel",
        "t_total": round(t_cpu, 4)
    })
    timings.append({
        "image": name,
        "method": "cuCIM.medial_axis",
        "t_total": round(t_ma, 4)
    })
    for label, t in timings_local.items():
        timings.append({
            "image": name,
            "method": label,
            "t_total": round(t, 4)
        })

# 5) Print a summary
import pandas as pd
df = pd.DataFrame(timings)
print(df.pivot_table(index="image", columns="method", values="t_total"))

# Optionally save timings to CSV
df.to_csv(outdir / "timings.csv", index=False)

print(f"All skeletons, distance maps, and timing.csv written to {outdir}")
