import numpy as np
import cupy as cp
from cupy import RawKernel
import sys, os

# Ensure the 'dsepruning' package is importable (adjust if needed)
sys.path.append(os.getcwd())

from skimage.io import imread
from skimage.morphology import medial_axis, binary_dilation
from skimage.draw import line
import sknw
from dsepruning.dsepruning import skel_pruning_DSE
import matplotlib.pyplot as plt
from time import time

# Define CUDA kernel for reconstructing disks at skeleton points
recn_disk_kernel_src = r'''
// Single CUDA kernel to compute *all* branch weights in one go
extern "C" __global__
void dse_batch_weight(const int* branch_ptrs,   // offsets into pts_flat
                      const int* pts_flat,      // y0,x0,y1,x1,...
                      const unsigned char* dist, // H×W distance map
                      int  n_branches,
                      int  H, int W,
                      float* weights_out) {     // output weight per branch
  int b = blockIdx.x;              // one block per branch
  int t = threadIdx.x;
  int start = branch_ptrs[b];
  int end   = branch_ptrs[b+1];
  float local_sum = 0.0f;
  for (int idx = start + t; idx < end; idx += blockDim.x) {
    int y = pts_flat[2*idx];
    int x = pts_flat[2*idx+1];
    local_sum += dist[y*W + x];    // or dist*(mask) logic inline
  }
  // reduce local_sum across threads in the block (e.g. via __shared__)
  // and write the final per-branch weight to weights_out[b]
}
'''
recn_disk_kernel = RawKernel(recn_disk_kernel_src, 'recn_disk')

# Monkey-patch GPU versions into dse_helper
import dsepruning.dse_helper as dsh

def gpu_recnstrc_by_disk(pts, dist, branch_recn):
    """GPU-accelerated reconstruction: paints disks for each skeleton point."""
    pts_np = np.asarray(pts, dtype=np.int32)
    H, W = dist.shape
    n_pts = pts_np.shape[0]
    # Flatten pts
    pts_flat = cp.array(pts_np.flatten(), dtype=cp.int32)
    # Radii from distance map
    radii = dist[pts_np[:, 0], pts_np[:, 1]].astype(np.int32)
    radii_flat = cp.array(radii, dtype=cp.int32)
    # GPU mask
    recn_gpu = cp.zeros(H * W, dtype=cp.uint8)
    # Launch kernel
    threads = 128
    blocks = (n_pts + threads - 1) // threads
    recn_disk_kernel((blocks,), (threads,),
                     (pts_flat, radii_flat, recn_gpu, n_pts, H, W))
    # Copy back to branch_recn (numpy int32)
    branch_recn[:, :] = cp.asnumpy(recn_gpu.reshape(H, W).astype(np.int32))

def gpu_get_weight(recn, branch_recn):
    """GPU-accelerated weight: sum of overlap of recn and branch_recn."""
    recn_gpu = cp.asarray(recn, dtype=cp.int32)
    branch_gpu = cp.asarray(branch_recn, dtype=cp.int32)
    return int(cp.sum(recn_gpu * branch_gpu).get())

# Patch the functions
dsh.recnstrc_by_disk = gpu_recnstrc_by_disk
dsh.get_weight = gpu_get_weight

# Utility to visualize graph to image
def graph2im(graph, shape):
    mask = np.zeros(shape, dtype=bool)
    for s, e in graph.edges():
        for val in graph[s][e].values():
            coords = val['pts']
            coords_1 = np.roll(coords, -1, axis=0)
            for i in range(len(coords) - 1):
                rr, cc = line(*coords[i], *coords_1[i])
                mask[rr, cc] = True
            mask[tuple(graph.nodes[s]['pts'].T.tolist())] = True
            mask[tuple(graph.nodes[e]['pts'].T.tolist())] = True
    return mask

# Main
if __name__ == "__main__":
    img_path = '/mnt/e/camerer_ml/datasets/METU_Concrete_Crack_Seg/BW_visible/236.png'
    img = imread(img_path, mode='L')
    bw = img > 0.5

    # Skeletonize
    skeleton, dist = medial_axis(bw, return_distance=True)

    # Run DSE prune with GPU-accelerated helper
    start = time()
    pruned_gpu = skel_pruning_DSE(skeleton, dist, min_area_px=1000)
    gpu_time = time() - start
    print(f"GPU-accelerated DSE prune time: {gpu_time:.2f} s")

    # For comparison, restore CPU helpers and run CPU version
    import importlib
    import dsepruning.dse_helper as dsh_cpu
    importlib.reload(dsh_cpu)  # restores original cpu Cython functions
    start = time()
    pruned_cpu = skel_pruning_DSE(skeleton, dist, min_area_px=1000)
    cpu_time = time() - start
    print(f"CPU DSE prune time:             {cpu_time:.2f} s")

    # Visualize
    from scipy.ndimage import binary_dilation
    skel_vis = binary_dilation(skeleton, iterations=2)
    pruned_gpu_vis = binary_dilation(pruned_gpu, iterations=2)
    pruned_cpu_vis = binary_dilation(pruned_cpu, iterations=2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(skel_vis, cmap='gray'); axes[0].set_title('Raw Skeleton'); axes[0].axis('off')
    axes[1].imshow(pruned_cpu_vis, cmap='gray'); axes[1].set_title(f'CPU DSE ({cpu_time:.2f}s)'); axes[1].axis('off')
    axes[2].imshow(pruned_gpu_vis, cmap='gray'); axes[2].set_title(f'GPU DSE ({gpu_time:.2f}s)'); axes[2].axis('off')
    plt.tight_layout()
    plt.show()
