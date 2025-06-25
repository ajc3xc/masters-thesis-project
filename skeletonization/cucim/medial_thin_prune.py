import cupy as cp
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
from cucim.skimage.morphology import medial_axis
from skimage.morphology import skeletonize as cpu_skeletonize

import cupy as cp
from cupyx.scipy.ndimage import convolve, distance_transform_edt
import numpy as np

def gpu_prune_spurs(skel_bool_gpu, width_map_gpu,
                    base_len=10, width_scale=2.0,
                    normal_radius=5, length_radius=50):
    """
    Prune skeleton branches whose length ≤ base + (w/width_scale)
    and whose normals never hit foreground within normal_radius.
    All on GPU.
    """
    H, W = skel_bool_gpu.shape
    sk = skel_bool_gpu.astype(cp.uint8)

    #–– 1) detect endpoints (center 10 + exactly one neighbor => sum == 11)
    K = cp.array([[1,1,1],
                  [1,10,1],
                  [1,1,1]], dtype=cp.int32)
    hits = convolve(sk, K, mode='constant', cval=0)
    endpoints = (hits == 11)

    #–– 2) approximate branch length around each endpoint by
    #    a small (!) distance transform on the skeleton mask
    #    (this actually gives distance _to_ the nearest background,
    #     but on a 1‐pixel skeleton it’s ≃ 0.5 × geodesic length)
    #    We threshold to keep only “long enough” branches.
    #    You could also run a true geodesic DT on the skeleton graph,
    #    but this is a lot simpler to code on GPU.
    small_skel_dist = distance_transform_edt(~skel_bool_gpu)  # floats
    # length_est ~= 1 / small_skel_dist  (inversely related)
    length_mask = (small_skel_dist < (1.0 / (base_len + width_map_gpu / width_scale)))

    #–– 3) normal test: for each endpoint pixel, sample +/- normal_radius
    #    along its normal.  We can do that with a tiny RawKernel if we like:
    module = cp.RawModule(code=r'''
    extern "C" __global__
    void normal_test(const unsigned char* sk, const float* wmap,
                     unsigned char* keep, int H, int W, int r) {
        int y = blockIdx.y*blockDim.y + threadIdx.y;
        int x = blockIdx.x*blockDim.x + threadIdx.x;
        if (y>=H||x>=W) return;
        int idx = y*W + x;
        if (!sk[idx]) { keep[idx]=0; return; }
        // find local gradient of width_map via sobel approx
        float dzdx = 0.0f, dzdy = 0.0f;
        if (x>0 && x<W-1) dzdx = (wmap[idx+1] - wmap[idx-1])*0.5f;
        if (y>0 && y<H-1) dzdy = (wmap[idx+W] - wmap[idx-W])*0.5f;
        // normal = ( -dzdx, -dzdy )  // points outwards
        float nx = -dzdy, ny = dzdx;
        float norm = sqrtf(nx*nx + ny*ny);
        if (norm<1e-6f) { keep[idx]=1; return; }
        nx/=norm; ny/=norm;
        // sample along normal both directions
        bool good=false;
        for (int s=-1; s<=1&&!good; s+=2) {
            for (int rr=1; rr<=r; rr++) {
                int yy = int(roundf(y + s*ny*rr));
                int xx = int(roundf(x + s*nx*rr));
                if (yy<0||yy>=H||xx<0||xx>=W || sk[yy*W+xx]) {
                    good=true;
                    break;
                }
            }
        }
        keep[idx] = good;
    }
    ''', options=('--std=c++11',), backend='nvcc')
    normal_test = module.get_function('normal_test')

    keep = cp.zeros_like(sk)
    block = (16,16)
    grid = ((W+15)//16, (H+15)//16)
    normal_test(grid, block,
        (sk, width_map_gpu.astype(cp.float32),
         keep, np.int32(H), np.int32(W), np.int32(normal_radius)))

    #–– final mask: keep skeleton pixels that
    #   a) are NOT endpoints (always keep interior pixels), OR
    #   b) are endpoints AND both length_mask _and_ normal_test passed
    final = skel_bool_gpu & (~endpoints | (length_mask & (keep.astype(bool))))
    return final

test_images = [
    '/mnt/d/camerer_ml/datasets/METU_Concrete_Crack_Seg/BW_visible/054.png',
    # ...add more images...
]

results = []

for img_path in test_images:
    # 1. Load and binarize
    img = iio.imread(img_path)
    if img.ndim == 3: img = img.mean(axis=2)
    binary = (img < 128)

    # 2. GPU medial axis (raw)
    binary_gpu = cp.array(binary)
    skel_gpu, dist_gpu = medial_axis(binary_gpu, return_distance=True)
    width_map_gpu = dist_gpu * 2

    # 3. GPU medial axis + pruning
    skel_pruned_gpu = gpu_prune_spurs(skel_gpu, width_map_gpu)  # function from above

    # 4. (optional) limited pre-thinning
    # Skipping here for speed, but you could use skimage.thin() for a few iterations if you want

    # 5. CPU Zhang–Suen (for reference)
    skel_cpu = cpu_skeletonize(binary)

    # 6. Visualization
    plt.figure(figsize=(16,4))
    plt.subplot(1,4,1)
    plt.imshow(binary, cmap='gray'); plt.title('Input')
    plt.subplot(1,4,2)
    plt.imshow(cp.asnumpy(skel_gpu), cmap='gray'); plt.title('GPU Medial axis')
    plt.subplot(1,4,3)
    plt.imshow(cp.asnumpy(skel_pruned_gpu), cmap='gray'); plt.title('GPU Medial axis + prune')
    plt.subplot(1,4,4)
    plt.imshow(skel_cpu, cmap='gray'); plt.title('CPU Zhang–Suen')
    plt.suptitle(img_path)
    plt.tight_layout(); plt.show()

    # Optionally, store results for later statistics
    results.append({
        'img': img_path,
        'raw_px': cp.asnumpy(skel_gpu).sum(),
        'pruned_px': cp.asnumpy(skel_pruned_gpu).sum(),
        'cpu_px': skel_cpu.sum(),
    })

# After running, you could plot or tabulate results['raw_px'], etc.
