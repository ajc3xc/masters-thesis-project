import time
import os
import numpy as np
import cupy as cp
import imageio.v3 as iio
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize as cpu_skeletonize

# --- Binary image loader (keep as before) ---
def load_binarize(image_path: Path, thresh: float):
    import imageio
    img = imageio.imread(str(image_path))
    if img.ndim == 3:
        img = img.mean(axis=2)
    img = img.astype(np.float32) / 255.0
    bin_mask = (img < thresh).astype(np.uint8)
    return bin_mask, img

# --- Fused CUDA kernel for Zhang–Suen thinning ---
_fused_kernel = r'''
extern "C" __global__
void zs_subiter(const unsigned char* img_in, unsigned char* img_out,
                const int H, const int W, const int subiter)
{
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if (y>=H || x>=W) return;
    int idx = y*W + x;
    unsigned char P = img_in[idx];
    if (!P) { img_out[idx]=0; return; }

    int north = (y>0     ? (y-1)*W + x     : -1);
    int south = (y<H-1   ? (y+1)*W + x     : -1);
    int west  = (x>0     ? y*W + (x-1)     : -1);
    int east  = (x<W-1   ? y*W + (x+1)     : -1);
    int nw    = (y>0&&x>0     ? (y-1)*W + (x-1) : -1);
    int ne    = (y>0&&x<W-1   ? (y-1)*W + (x+1) : -1);
    int sw    = (y<H-1&&x>0   ? (y+1)*W + (x-1) : -1);
    int se    = (y<H-1&&x<W-1 ? (y+1)*W + (x+1) : -1);

    unsigned char P2 = (north>=0 ? img_in[north] : 0);
    unsigned char P3 = (north>=0&&east>=0 ? img_in[ne]   : 0);
    unsigned char P4 = (east>=0 ? img_in[east]  : 0);
    unsigned char P5 = (south>=0&&east>=0? img_in[se]   : 0);
    unsigned char P6 = (south>=0 ? img_in[south] : 0);
    unsigned char P7 = (south>=0&&west>=0? img_in[sw]   : 0);
    unsigned char P8 = (west>=0 ? img_in[west]  : 0);
    unsigned char P9 = (north>=0&&west>=0? img_in[nw]   : 0);

    int numN = P2+P3+P4+P5+P6+P7+P8+P9;
    int trans = 0;
    trans += (!P2 && (P3||P4));
    trans += (!P4 && (P5||P6));
    trans += (!P6 && (P7||P8));
    trans += (!P8 && (P9||P2));

    bool m1 = (numN>=2 && numN<=6);
    bool m2 = (trans==1);

    // Sub-iteration 1 or 2
    bool m3, m4;
    if (subiter == 0) {
        m3 = !(P2 && P4 && P6);
        m4 = !(P4 && P6 && P8);
    } else {
        m3 = !(P2 && P4 && P8);
        m4 = !(P2 && P6 && P8);
    }

    bool remove = false;
    if (m1 && m2 && m3 && m4) remove = true;
    img_out[idx] = remove ? 0 : P;
}
'''

module = cp.RawModule(code=_fused_kernel, options=('--std=c++11',), backend='nvcc')
zs_kernel = module.get_function('zs_subiter')

def fused_zhang_suen(img_gpu, max_iters=200):
    H, W = img_gpu.shape
    in_buf  = cp.array(img_gpu, dtype=cp.uint8)
    out_buf = cp.empty_like(in_buf)
    block = (32,32)
    grid  = ((W+31)//32, (H+31)//32)
    iters = 0
    prev  = cp.empty_like(in_buf)
    while iters < max_iters and not cp.all(in_buf == prev):
        prev[:] = in_buf
        zs_kernel(grid, block, (in_buf, out_buf, H, W, 0))  # subiter=0
        zs_kernel(grid, block, (out_buf, in_buf, H, W, 1))  # subiter=1
        iters += 1
    return in_buf.astype(bool)

# ----------------------------------------------------------------------------
# 2) Main pipeline: load → GPU thin vs CPU thin → overlay & time
# ----------------------------------------------------------------------------
INPUT = Path("/mnt/d/camerer_ml/skeletonization/skeletonide/test/images/horse.png")
out_dir = Path('./results')
out_dir.mkdir(exist_ok=True)
bw_mask, gray = load_binarize(INPUT, thresh=0.5)

# --- GPU fused Zhang–Suen ---
gpu_in = cp.array(bw_mask)
_ = fused_zhang_suen(gpu_in)   # warm-up
cp.cuda.Stream.null.synchronize()
t0 = time.time()
sk_gpu = fused_zhang_suen(gpu_in)
cp.cuda.Stream.null.synchronize()
t_gpu = time.time() - t0
print(f"[GPU fused ZS] {t_gpu:.4f}s")
sk_gpu_np = sk_gpu.get()

# --- CPU scikit-image skeletonize ---
t0 = time.time()
sk_cpu = cpu_skeletonize(bw_mask)
t_cpu = time.time() - t0
print(f"[CPU skimage.skeletonize] {t_cpu:.4f}s")

# --- Overlay plot and export images ---
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(gray, cmap='gray')
axs[0].imshow(np.ma.masked_where(~sk_gpu_np, sk_gpu_np), cmap='jet', alpha=0.7)
axs[0].set_title(f"GPU fused ZS\n{t_gpu:.3f}s")
axs[0].axis("off")

axs[1].imshow(gray, cmap='gray')
axs[1].imshow(np.ma.masked_where(~sk_cpu, sk_cpu), cmap='jet', alpha=0.7)
axs[1].set_title(f"CPU skimage\n{t_cpu:.3f}s")
axs[1].axis("off")

plt.tight_layout()
out_img = out_dir / "zs_gpu_vs_cpu_overlay.png"
plt.savefig(out_img, bbox_inches='tight')
plt.show()

# Also export the raw skeleton masks for further analysis if needed
iio.imwrite(out_dir / "skel_gpu.png", (sk_gpu_np * 255).astype(np.uint8))
iio.imwrite(out_dir / "skel_cpu.png", (sk_cpu * 255).astype(np.uint8))

# Export timing to txt file
with open(out_dir / "timing.txt", "w") as f:
    f.write(f"[GPU fused ZS]: {t_gpu:.4f}s\n")
    f.write(f"[CPU skimage.skeletonize]: {t_cpu:.4f}s\n")

print(f"Results saved in {out_dir}")
