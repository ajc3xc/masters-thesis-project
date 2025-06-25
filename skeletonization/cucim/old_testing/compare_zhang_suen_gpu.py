import time
import cupy as cp
import numpy as np
import imageio.v3 as iio
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize as cpu_skeletonize

def zhang_suen_thinning_gpu(binary_img, max_iters=200):
    img = binary_img.astype(cp.uint8).copy()
    prev = cp.zeros_like(img)
    iters = 0

    def neighbors(x):
        P2 = cp.roll(x, -1, axis=0)
        P3 = cp.roll(P2, -1, axis=1)
        P4 = cp.roll(x, -1, axis=1)
        P5 = cp.roll(P4, 1, axis=0)
        P6 = cp.roll(x, 1, axis=0)
        P7 = cp.roll(P6, 1, axis=1)
        P8 = cp.roll(x, 1, axis=1)
        P9 = cp.roll(P8, -1, axis=0)
        return P2, P3, P4, P5, P6, P7, P8, P9

    while not cp.all(img == prev) and iters < max_iters:
        prev = img.copy()
        P2, P3, P4, P5, P6, P7, P8, P9 = neighbors(img)
        nb_sum  = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
        trans   = ((~P2 & (P3 | P4))
                 + (~P4 & (P5 | P6))
                 + (~P6 & (P7 | P8))
                 + (~P8 & (P9 | P2))) == 1

        # sub-iteration 1
        m1 = (nb_sum >= 2) & (nb_sum <= 6)
        m2 = trans
        m3 = (P2 * P4 * P6 == 0)
        m4 = (P4 * P6 * P8 == 0)
        marker = img & m1 & m2 & m3 & m4
        img = img & ~marker

        # sub-iteration 2
        P2, P3, P4, P5, P6, P7, P8, P9 = neighbors(img)
        nb_sum  = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
        trans   = ((~P2 & (P3 | P4))
                 + (~P4 & (P5 | P6))
                 + (~P6 & (P7 | P8))
                 + (~P8 & (P9 | P2))) == 1
        m1 = (nb_sum >= 2) & (nb_sum <= 6)
        m2 = trans
        m3 = (P2 * P4 * P8 == 0)
        m4 = (P2 * P6 * P8 == 0)
        marker = img & m1 & m2 & m3 & m4
        img = img & ~marker

        iters += 1

    return img.astype(bool)

# Load and binarize
input_path = Path("/mnt/d/camerer_ml/skeletonization/skeletonide/test/images/horse.png")
bw = iio.imread(input_path) < 128

# GPU batch timing
bw_gpu = cp.array(bw)
N = 20
for _ in range(2):  # warmup
    _ = zhang_suen_thinning_gpu(bw_gpu)
cp.cuda.Stream.null.synchronize()
t0 = time.time()
for _ in range(N):
    sk_gpu = zhang_suen_thinning_gpu(bw_gpu)
    cp.cuda.Stream.null.synchronize()
t_gpu_batch = (time.time() - t0) / N
sk_gpu_np = cp.asnumpy(sk_gpu)  # Take last result for plotting
print(f"[GPU batched] avg per run: {t_gpu_batch:.4f}s")

# CPU batch timing
for _ in range(2):  # warmup
    _ = cpu_skeletonize(bw)
t0 = time.time()
for _ in range(N):
    sk_cpu = cpu_skeletonize(bw)
t_cpu_batch = (time.time() - t0) / N
print(f"[CPU batched] avg per run: {t_cpu_batch:.4f}s")

# Plot overlay
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(bw, cmap="gray")
axs[0].imshow(np.ma.masked_where(~sk_gpu_np, sk_gpu_np), cmap="jet", alpha=0.6)
axs[0].set_title(f"GPU Zhang–Suen\n{t_gpu_batch:.4f}s")
axs[0].axis("off")

axs[1].imshow(bw, cmap="gray")
axs[1].imshow(np.ma.masked_where(~sk_cpu, sk_cpu), cmap="jet", alpha=0.6)
axs[1].set_title(f"CPU Zhang–Suen\n{t_cpu_batch:.4f}s")
axs[1].axis("off")

plt.tight_layout()
plt.show()
