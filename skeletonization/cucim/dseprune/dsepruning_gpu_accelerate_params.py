import numpy as np
import cupy as cp
from cupy import RawKernel
import importlib
import time
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.morphology import medial_axis
from scipy.ndimage import binary_dilation
from skimage.draw import line
import sknw

# Import DSE prune
from dsepruning.dsepruning import skel_pruning_DSE
import dsepruning.dse_helper as dsh

# Save original helpers to restore
orig_recnstrc = dsh.recnstrc_by_disk
orig_get_weight = dsh.get_weight

# Load image and skeletonize
img_path = '/mnt/e/camerer_ml/datasets/METU_Concrete_Crack_Seg/BW_visible/236.png'
img = imread(img_path, mode='L')
bw = img > 0.5
skeleton, dist = medial_axis(bw, return_distance=True)
H, W = skeleton.shape

# Build graph and find maximum branch length
graph = sknw.build_sknw(skeleton, multi=True)
max_len = 0
for s, e in graph.edges():
    for val in graph[s][e].values():
        max_len = max(max_len, len(val['pts']))
del graph

# Compile CUDA kernel for disk reconstruction
kernel_src = r'''
extern "C" __global__
void recn_disk(const int* pts, const unsigned char* radii, unsigned char* recn,
               int n_pts, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_pts) return;
    int y = pts[2*idx];
    int x = pts[2*idx + 1];
    int r = radii[idx];
    int r_sq = r * r;
    for (int dy = -r; dy <= r; dy++) {
        int yy = y + dy;
        if (yy < 0 || yy >= H) continue;
        int dx_max = (int) sqrtf((float)(r_sq - dy*dy));
        for (int dx = -dx_max; dx <= dx_max; dx++) {
            int xx = x + dx;
            if (xx < 0 || xx >= W) continue;
            recn[yy * W + xx] = 1;
        }
    }
}
'''
recn_disk_kernel = RawKernel(kernel_src, 'recn_disk')

# Function to patch GPU helpers
def patch_helpers(threads=128, reuse_buffers=False, uint8_radii=False, use_stream=False):
    # Pre-allocate buffers if reuse is True
    size = H * W
    if reuse_buffers:
        pts_buf = cp.empty((2 * max_len,), dtype=cp.int32)
        radii_dtype = cp.uint8 if uint8_radii else cp.int32
        rad_buf = cp.empty((max_len,), dtype=radii_dtype)
        recn_gpu = cp.zeros((size,), dtype=cp.uint8)
    if use_stream:
        stream = cp.cuda.Stream(non_blocking=True)
    else:
        stream = None

    def recnstrc_by_disk_gpu(pts_np, dist_np, branch_recn):
        n_pts = pts_np.shape[0]
        # Prepare pts_flat
        if reuse_buffers:
            buf_pts = pts_buf[:2*n_pts]
            buf_pts[:] = cp.asarray(pts_np.flatten(), dtype=cp.int32)
            buf_rad = rad_buf[:n_pts]
            radii = dist_np[pts_np[:,0], pts_np[:,1]]
            if uint8_radii:
                radii = np.minimum(radii, 255).astype(np.uint8)
            buf_rad[:] = cp.asarray(radii, dtype=buf_rad.dtype)
            recn_gpu.fill(0)
        else:
            buf_pts = cp.array(pts_np.flatten(), dtype=cp.int32)
            radii = dist_np[pts_np[:,0], pts_np[:,1]]
            if uint8_radii:
                radii = np.minimum(radii, 255).astype(np.uint8)
            buf_rad = cp.array(radii, dtype=cp.uint8 if uint8_radii else cp.int32)
            recn_gpu = cp.zeros((size,), dtype=cp.uint8)

        blocks = (n_pts + threads - 1) // threads
        if use_stream:
            recn_disk_kernel((blocks,), (threads,), (buf_pts, buf_rad, recn_gpu, n_pts, H, W), stream=stream)
            stream.synchronize()
        else:
            recn_disk_kernel((blocks,), (threads,), (buf_pts, buf_rad, recn_gpu, n_pts, H, W))

        # Copy back
        branch_recn[:, :] = cp.asnumpy(recn_gpu.reshape(H, W).astype(np.int32))

    def get_weight_gpu(recn, branch_recn):
        # simple numpy sum
        return int((recn * branch_recn).sum())

    # Patch
    dsh.recnstrc_by_disk = recnstrc_by_disk_gpu
    dsh.get_weight = get_weight_gpu

# List of configurations to test
configs = [
    {'name':'Base GPU-128',   'threads':128, 'reuse':False, 'uint8':False, 'stream':False},
    {'name':'Reuse GPU-128',  'threads':128, 'reuse':True,  'uint8':False, 'stream':False},
    {'name':'Uint8 GPU-128',  'threads':128, 'reuse':True,  'uint8':True,  'stream':False},
    {'name':'Stream GPU-128', 'threads':128, 'reuse':True,  'uint8':True,  'stream':True},
    {'name':'GPU-256',        'threads':256, 'reuse':True,  'uint8':True,  'stream':True},
    {'name':'GPU-512',        'threads':512, 'reuse':True,  'uint8':True,  'stream':True},
]

results = []

# CPU baseline (restore original)
dsh.recnstrc_by_disk = orig_recnstrc
dsh.get_weight = orig_get_weight
start = time.perf_counter()
pruned_cpu = skel_pruning_DSE(skeleton, dist, min_area_px=500)
t_cpu = time.perf_counter() - start
results.append(('CPU', t_cpu))

# GPU configs
for cfg in configs:
    # Reload module to ensure clean state
    importlib.reload(dsh)
    patch_helpers(threads=cfg['threads'],
                  reuse_buffers=cfg['reuse'],
                  uint8_radii=cfg['uint8'],
                  use_stream=cfg['stream'])
    start = time.perf_counter()
    pruned = skel_pruning_DSE(skeleton, dist, min_area_px=500)
    t = time.perf_counter() - start
    # Verify same output
    assert np.array_equal(pruned_cpu, pruned), f"Output mismatch for {cfg['name']}"
    results.append((cfg['name'], t))

# Restore original helpers
dsh.recnstrc_by_disk = orig_recnstrc
dsh.get_weight = orig_get_weight

# Plot timings
labels, times = zip(*results)
plt.figure(figsize=(8,4))
plt.bar(labels, times)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Time (s)')
plt.title('DSE Prune: CPU vs GPU Optimizations')
plt.tight_layout()
plt.show()

# Print results
for name, t in results:
    print(f"{name:15s}: {t:.3f} s")
