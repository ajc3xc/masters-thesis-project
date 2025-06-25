import cupy as cp
import numpy as np
import imageio
import imageio.v3 as iio
from pathlib import Path
import time
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize as cpu_skeletonize

kernel_code = r'''
extern "C" __global__
void zhang_suen_iter(const unsigned char* img_in, unsigned char* img_out, int H, int W, int subiter) {
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if (y >= H || x >= W) return;
    int idx = y*W + x;
    unsigned char P = img_in[idx];
    if (P == 0) { img_out[idx] = 0; return; }
    // Neighbors (0/255 values)
    unsigned char P2 = (y > 0     ) ? img_in[(y-1)*W + x    ] : 0;
    unsigned char P3 = (y > 0 && x < W-1 ) ? img_in[(y-1)*W + (x+1)] : 0;
    unsigned char P4 = (x < W-1   ) ? img_in[y*W + (x+1)  ] : 0;
    unsigned char P5 = (y < H-1 && x < W-1) ? img_in[(y+1)*W + (x+1)] : 0;
    unsigned char P6 = (y < H-1   ) ? img_in[(y+1)*W + x    ] : 0;
    unsigned char P7 = (y < H-1 && x > 0) ? img_in[(y+1)*W + (x-1)] : 0;
    unsigned char P8 = (x > 0     ) ? img_in[y*W + (x-1)  ] : 0;
    unsigned char P9 = (y > 0 && x > 0) ? img_in[(y-1)*W + (x-1)] : 0;
    // Binary 0/1 for logic
    int b2 = P2 ? 1 : 0, b3 = P3 ? 1 : 0, b4 = P4 ? 1 : 0, b5 = P5 ? 1 : 0;
    int b6 = P6 ? 1 : 0, b7 = P7 ? 1 : 0, b8 = P8 ? 1 : 0, b9 = P9 ? 1 : 0;
    int A = (!b2 && (b3||b4)) + (!b4 && (b5||b6)) + (!b6 && (b7||b8)) + (!b8 && (b9||b2));
    int B = b2 + b3 + b4 + b5 + b6 + b7 + b8 + b9;
    int m1, m2;
    if (subiter == 0) {
        m1 = !(b2 && b4 && b6);
        m2 = !(b4 && b6 && b8);
    } else {
        m1 = !(b2 && b4 && b8);
        m2 = !(b2 && b6 && b8);
    }
    int remove = (A == 1) && (B >= 2 && B <= 6) && m1 && m2;
    img_out[idx] = remove ? 0 : 255;
}
'''

# Compile kernel
module = cp.RawModule(code=kernel_code, options=('--std=c++11',), backend='nvcc')
zs_kernel = module.get_function('zhang_suen_iter')

def zhang_suen_gpu(binary_img, max_iters=100):
    H, W = binary_img.shape
    # 0/255 uint8 convention!
    in_buf = cp.array(binary_img, dtype=cp.uint8) * 255
    out_buf = cp.empty_like(in_buf)
    prev = cp.empty_like(in_buf)
    block = (32, 32)
    grid = ((W+31)//32, (H+31)//32)
    iters = 0
    while iters < max_iters:
        prev[:] = in_buf
        zs_kernel(grid, block, (in_buf, out_buf, np.int32(H), np.int32(W), np.int32(0)))
        cp.cuda.Stream.null.synchronize()
        zs_kernel(grid, block, (out_buf, in_buf, np.int32(H), np.int32(W), np.int32(1)))
        cp.cuda.Stream.null.synchronize()
        if cp.all(in_buf == prev):
            break
        iters += 1
    return (in_buf.get() > 0).astype(np.uint8)

# --- Example usage ---

input_path = Path("/mnt/d/camerer_ml/skeletonization/skeletonide/test/images/horse.png")
img = imageio.v3.imread(str(input_path))
if img.ndim == 3:
    img = img.mean(axis=2)
img = (img < 128).astype(np.uint8)

# GPU
_ = zhang_suen_gpu(img)  # warmup

import numpy as np
from skimage.morphology import skeletonize, disk
from scipy.ndimage import binary_dilation

def bridge_gaps(skel, max_gap=2):
    # Dilate and skeletonize again to connect tiny gaps
    dilated = binary_dilation(skel, structure=disk(max_gap))
    return skeletonize(dilated)

t0 = time.time()
skel_gpu = zhang_suen_gpu(img)
skel_gpu = bridge_gaps(skel_gpu, max_gap=1)
gpu_time = time.time() - t0

# CPU
t0 = time.time()
skel_cpu = cpu_skeletonize(img)
cpu_time = time.time() - t0

# Visualize
fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs[0].imshow(img, cmap='gray')
axs[0].imshow(np.ma.masked_where(skel_gpu==0, skel_gpu), cmap='jet', alpha=0.7)
axs[0].set_title(f"GPU Zhang-Suen ({gpu_time:.4f}s)")
axs[0].axis('off')
axs[1].imshow(img, cmap='gray')
axs[1].imshow(np.ma.masked_where(skel_cpu==0, skel_cpu), cmap='jet', alpha=0.7)
axs[1].set_title(f"CPU (skimage) ({cpu_time:.4f}s)")
axs[1].axis('off')
plt.tight_layout()
plt.show()

# Save for review
import imageio
outdir = Path('./results'); outdir.mkdir(exist_ok=True)
imageio.imwrite(outdir / 'skel_gpu.png', (skel_gpu*255).astype(np.uint8))
imageio.imwrite(outdir / 'skel_cpu.png', (skel_cpu*255).astype(np.uint8))
with open(outdir / 'timing.txt', 'w') as f:
    f.write(f"GPU ZhangSuen: {gpu_time:.4f}s\nCPU skimage: {cpu_time:.4f}s\n")
print("Saved skeletons and timing to ./results/")
