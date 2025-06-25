import sknw
import numpy as np
from skimage.draw import line

# --- CPU helpers (default, import from .dse_helper) ---
from dsepruning.dse_helper import recnstrc_by_disk as cpu_recnstrc_by_disk
from dsepruning.dse_helper import get_weight as cpu_get_weight
from dsepruning.dsepruning import graph2im, skel_pruning_DSE, _remove_branch_by_DSE, _remove_mid_node, flatten

# --- GPU helpers (define here, with batched CUDA) ---
import cupy as cp
from cupy import RawKernel

# CUDA kernel for batched branch weight (as before)
batch_weight_kernel_src = r'''
extern "C" __global__
void batch_get_weight(const int* branch_ptrs,
                      const int* pts_flat,
                      const unsigned char* dist_map,
                      const int* recn_global,
                      int n_br, int H, int W,
                      unsigned int* weight_out) {
    extern __shared__ unsigned int sdat[];
    int b = blockIdx.x;
    if (b >= n_br) return;
    int t = threadIdx.x;
    int start = branch_ptrs[b];
    int end   = branch_ptrs[b+1];
    unsigned int sum = 0;
    for (int idx = start + t; idx < end; idx += blockDim.x) {
        int y = pts_flat[2*idx];
        int x = pts_flat[2*idx+1];
        int r = dist_map[y*W + x];
        int r2 = r*r;
        for (int dy=-r; dy<=r; ++dy) {
            int yy = y+dy;
            if (yy<0||yy>=H) continue;
            int dxmax = (int)sqrtf((float)(r2 - dy*dy));
            for (int dx=-dxmax; dx<=dxmax; ++dx) {
                int xx = x+dx;
                if (xx<0||xx>=W) continue;
                sum += recn_global[yy*W + xx];
            }
        }
    }
    sdat[t]=sum;
    __syncthreads();
    for (int s=blockDim.x/2; s>0; s>>=1) {
        if (t<s) sdat[t]+=sdat[t+s];
        __syncthreads();
    }
    if (t==0) weight_out[b]=sdat[0];
}
'''
batch_weight_kernel = RawKernel(batch_weight_kernel_src, 'batch_get_weight')

# ---------- Helpers to batch branches for GPU ----------
def batch_gpu_get_weights(terminals, G, recn, dist):
    H, W = dist.shape
    max_pts = int(np.sum(recn > 0))
    pts_flat = cp.empty((2*max_pts,), dtype=cp.int32)
    branch_ptrs = cp.empty((max_pts+1,), dtype=cp.int32)
    weights_out = cp.empty((len(terminals),), dtype=cp.uint32)
    dist_gpu = cp.asarray(np.clip(dist,0,255).astype(np.uint8))
    recn_gpu = cp.asarray(recn, dtype=cp.int32)
    ptr = 0
    host_ptrs = [0]
    for i, node in enumerate(terminals):
        nbr = next(G.neighbors(node))
        pts = np.asarray(G[node][nbr][0]['pts'], dtype=np.int32)
        n_pts = len(pts)
        pts_flat[2*ptr:2*(ptr+n_pts)] = cp.asarray(pts.flatten(), dtype=cp.int32)
        ptr += n_pts
        host_ptrs.append(ptr)
    host_ptrs = np.array(host_ptrs, dtype=np.int32)
    branch_ptrs[:len(terminals)+1] = cp.asarray(host_ptrs, dtype=cp.int32)
    shared = 256 * np.dtype(np.uint32).itemsize
    batch_weight_kernel(
        (len(terminals),), (256,),
        (branch_ptrs, pts_flat, dist_gpu, recn_gpu, len(terminals), H, W, weights_out),
        shared_mem=shared)
    return cp.asnumpy(weights_out)

# ------------------- DSE pruning with pluggable helpers -------------------
def skel_pruning_DSE_gpu(skel, dist, min_area_px=100, return_graph=False):
    graph = sknw.build_sknw(skel, multi=True)
    dist = dist.astype(np.int32)
    graph = _remove_mid_node(graph)
    edges = list(set(graph.edges()))
    pts = []
    for s, e in edges:
        vals = flatten([[v] for v in graph[s][e].values()])
        for ix, val in enumerate(vals):
            pts.extend(val.get('pts').tolist())
        pts.append(graph.nodes[s]['o'].astype(np.int32).tolist())
        pts.append(graph.nodes[e]['o'].astype(np.int32).tolist())
    recnstrc = np.zeros_like(dist, dtype=np.int32)
    # Use correct helper for first full recn
    cpu_recnstrc_by_disk(np.array(pts, dtype=np.int32), dist, recnstrc)
    chosen_recnstrc = cpu_recnstrc_by_disk
    chosen_get_weight = (lambda recn, branch_recn: 0)  # dummy, never called in GPU
    num_nodes = len(graph.nodes())
    checked_terminal = set()
    while True:
        graph, recnstrc = _remove_branch_by_DSE(
        graph, recnstrc, dist, min_area_px, checked_terminal=checked_terminal)
        if len(graph.nodes()) == num_nodes:
            break
        graph = _remove_mid_node(graph)
        num_nodes = len(graph.nodes())
    if return_graph:
        return graph2im(graph, skel.shape), graph
    else:
        return graph2im(graph, skel.shape)
        
def gpu_get_weight(recn, branch_recn):
    recn_gpu = cp.asarray(recn, dtype=cp.int32)
    branch_gpu = cp.asarray(branch_recn, dtype=cp.int32)
    return int(cp.sum(recn_gpu * branch_gpu).get())

import dsepruning.dse_helper as dsh

def skel_pruning_DSE_gpu_mini(skel, dist, min_area_px=100, return_graph=False):
    orig_weight = dsh.get_weight
    try:
        dsh.get_weight = gpu_get_weight
        return skel_pruning_DSE(skel, dist, min_area_px=min_area_px, return_graph=return_graph)
    finally:
        dsh.get_weight = orig_weight

# ------------------- Test: Run both CPU and GPU, check identical -------------------
if __name__ == "__main__":
    from skimage.io import imread
    from cucim.skimage.morphology import medial_axis
    from scipy.ndimage import binary_dilation
    import matplotlib.pyplot as plt
    import time
    import glob, os
    from skimage.transform import resize

    # ---- Directory and File List ----
    img_dir = '/mnt/e/camerer_ml/datasets/METU_Concrete_Crack_Seg/BW_visible/'
    all_pngs = sorted(glob.glob(os.path.join(img_dir, '*.png')))
    img_list = all_pngs[:]  # Use up to 200

    print(f"Testing {len(img_list)} images...")

    # ---- Warmup: pick first image ----
    img = imread(img_list[0], mode='L')
    bw = img > 0.5
    skel, dist = medial_axis(cp.array(bw), return_distance=True)
    skel, dist = skel.get(), dist.get()
    print(type(skel))

    for _ in range(2):
        _ = skel_pruning_DSE_gpu_mini(skel, dist, min_area_px=500)
        _ = skel_pruning_DSE_gpu(skel, dist, min_area_px=500)
        cp.cuda.Stream.null.synchronize()
        _ = skel_pruning_DSE(skel, dist, min_area_px=500)
    
    print("Warmup complete, starting benchmark...")

    # ---- Timing Variables ----
    medial_times = []
    cpu_times = []
    gpu_times = []
    gpu_mini_times = []
    all_match = True

    # ---- Benchmark Loop ----
    for idx, img_path in enumerate(img_list):
        img = imread(img_path, mode='L')
        bw = img > 0.5
        
        orig_h, orig_w = bw.shape

        # Desired size for the smallest side
        min_size = 1080

        smallest_dim = min(orig_h, orig_w)

        # Only scale down if smallest side > 1080
        if smallest_dim > min_size:
            scale = min_size / smallest_dim
            new_h = int(orig_h * scale)
            new_w = int(orig_w * scale)
        else:
            # No scaling, keep original size
            new_h = orig_h
            new_w = orig_w
            
        bw_resized = resize(
            bw, 
            (new_h, new_w), 
            order=0, 
            preserve_range=True, 
            anti_aliasing=False
        )
        print(bw_resized.shape)

        
        tm = time.time()
        skel, dist = medial_axis(cp.array(bw_resized), return_distance=True)
        cp.cuda.Stream.null.synchronize()  # Ensure GPU is ready
        medial_times.append(time.time() - tm)
        skel, dist = skel.get(), dist.get()
        
        

        t0 = time.time()
        res_cpu = skel_pruning_DSE(skel, dist, min_area_px=500)
        cpu_times.append(time.time() - t0)

        t1 = time.time()
        res_gpu = skel_pruning_DSE_gpu(skel, dist, min_area_px=500)
        cp.cuda.Stream.null.synchronize()
        gpu_times.append(time.time() - t1)

        t2 = time.time()
        res_gpu_mini = skel_pruning_DSE_gpu_mini(skel, dist, min_area_px=500)
        cp.cuda.Stream.null.synchronize()
        gpu_mini_times.append(time.time() - t2)

        if not (np.array_equal(res_cpu, res_gpu_mini)):
            all_match = False
            print(f"Mismatch at {img_path}")
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx+1} / {len(img_list)}...")

    print("------- Mean Results (over %d images) -------" % len(cpu_times))
    print(f"CPU mean      : {np.mean(cpu_times):.3f}s")
    print(f"GPU mean      : {np.mean(gpu_times):.3f}s")
    print(f"Mini GPU mean : {np.mean(gpu_mini_times):.3f}s")
    print(f"Medial Axis mean: {np.mean(medial_times):.3f}s")
    print(f"All Mini-GPU == CPU? {all_match}")
