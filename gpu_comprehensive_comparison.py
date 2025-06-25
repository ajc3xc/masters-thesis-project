import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from cucim.skimage.feature import canny
from cucim.skimage.filters import sobel_h, sobel_v
from cucim.core.operations.morphology import distance_transform_edt
from cucim.skimage.morphology import medial_axis
from sklearn.decomposition import PCA
from skimage.io import imread
from dsepruning.dsepruning import skel_pruning_DSE
from time import perf_counter

# --- User settings ---
IMG_PATH = '/mnt/e/camerer_ml/datasets/METU_Concrete_Crack_Seg/BW/236.JPG'
THRESHOLD = 0.25
MIN_AREA_PX = 1000
PATCH_SIZE = 20  # for PCA-Local

gray_cpu = imread(IMG_PATH, as_gray=True)
gray = cp.asarray(gray_cpu)
bw = (gray > THRESHOLD)

# --------- Helpers -------------
def time_method(method_func, *args, dryrun=True, name="", timings=None):
    print(f"\n{'[DRY RUN]' if dryrun else '[RUN]'} {name}...")
    t0 = perf_counter()
    out = method_func(*args)
    dt = perf_counter() - t0
    if not dryrun:
        print(f"{name} took {dt*1000:.1f} ms")
        if timings is not None:
            timings[name] = dt * 1000
    return out

def get_local_tangent_normal(y, x, skel, window=5):
    y0, y1 = max(0, y-window), min(skel.shape[0], y+window+1)
    x0, x1 = max(0, x-window), min(skel.shape[1], x+window+1)
    local_points = np.column_stack(np.nonzero(skel[y0:y1, x0:x1]))
    if len(local_points) < 3:
        return np.array([1.0, 0.0]), np.array([0.0, 1.0])
    local_points = local_points + [y0, x0]
    cov = np.cov(local_points.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    tangent = eigvecs[:, np.argmax(eigvals)]
    normal = np.array([-tangent[1], tangent[0]])
    tangent /= np.linalg.norm(tangent)
    normal /= np.linalg.norm(normal)
    return tangent, normal

# ---------- Methods --------------
def method_medial_dse(bw):
    sk_gpu, dist_gpu = medial_axis(bw, return_distance=True)
    sk, dist = sk_gpu.get(), dist_gpu.get()
    pruned = skel_pruning_DSE(sk, dist, min_area_px=MIN_AREA_PX, return_graph=False)
    width_map = np.zeros_like(sk, float)
    width_map[pruned] = dist[pruned] * 2
    return pruned, width_map

def method_canny_profile(gray, bw):
    # Ensure input is on GPU
    gray_gpu = cp.asarray(gray)
    bw_gpu = cp.asarray(bw)

    # Step 1: Canny on grayscale image
    edges_gpu = canny(gray_gpu, sigma=2.0)

    # Step 2: Distance transform from Canny edges
    edt_gpu = distance_transform_edt(edges_gpu)

    # Step 3: Medial axis on binary image (currently on CPU)
    skel, dist = medial_axis(bw_gpu, return_distance=True)
    skel_cpu, dist_cpu = skel.get(), dist.get()

    # Step 4: Prune skeleton using distance map (assumes skel_pruning_DSE is CPU-based)
    pruned = skel_pruning_DSE(skel_cpu, dist_cpu, min_area_px=MIN_AREA_PX, return_graph=False)

    # Step 5: Use GPU EDT to calculate width map
    edt_np = cp.asnumpy(edt_gpu)
    width_map = np.zeros_like(skel_cpu, dtype=float)
    width_map[pruned] = edt_np[pruned] * 2

    return pruned, width_map

def method_pca_local(bw, patch_size=PATCH_SIZE):
    sk_gpu, dist_gpu = medial_axis(bw, return_distance=True)
    sk, dist = sk_gpu.get(), dist_gpu.get()
    pruned = skel_pruning_DSE(sk, dist, min_area_px=MIN_AREA_PX, return_graph=False)
    Y, X = np.nonzero(pruned)
    width_map = np.zeros_like(sk, float)
    edge = (bw > 0.5).get().astype(np.uint8)
    for y, x in zip(Y, X):
        y0, y1 = max(0, y - patch_size), min(sk.shape[0], y + patch_size + 1)
        x0, x1 = max(0, x - patch_size), min(sk.shape[1], x + patch_size + 1)
        patch = edge[y0:y1, x0:x1]
        coords = np.column_stack(np.nonzero(patch))
        if len(coords) < 4:
            continue
        pca = PCA(n_components=2)
        pca.fit(coords)
        minor_axis = pca.components_[1]
        proj = (coords - np.array([y0, x0])) @ minor_axis
        width_map[y, x] = proj.max() - proj.min()
    return pruned, width_map

def method_profile_normal(gray, bw):
    sk_gpu, dist_gpu = medial_axis(bw, return_distance=True)
    sk = sk_gpu.get()
    pruned = skel_pruning_DSE(sk, dist_gpu.get(), min_area_px=MIN_AREA_PX, return_graph=False)
    width_map = np.zeros_like(sk, float)
    Y, X = np.nonzero(pruned)
    gray_cpu = gray.get()
    for y, x in zip(Y, X):
        window = PATCH_SIZE
        y0, y1 = max(0, y-window), min(gray_cpu.shape[0], y+window+1)
        x0, x1 = max(0, x-window), min(gray_cpu.shape[1], x+window+1)
        patch = gray_cpu[y0:y1, x0:x1]
        if patch.size < 9:
            continue
        gx = cp.asnumpy(sobel_h(cp.asarray(patch)))
        gy = cp.asnumpy(sobel_v(cp.asarray(patch)))
        gx = gx.mean()
        gy = gy.mean()
        v = np.array([gy, gx])
        nrm = np.linalg.norm(v)
        if nrm < 1e-4:
            continue
        v /= nrm
        normal = np.array([-v[1], v[0]])
        forw, back = None, None
        for i in range(1, PATCH_SIZE):
            pt = np.round([y, x] + normal * i).astype(int)
            if not (0 <= pt[0] < gray_cpu.shape[0] and 0 <= pt[1] < gray_cpu.shape[1]):
                break
            if bw.get()[pt[0], pt[1]] < 0.5:
                forw = pt
                break
        for i in range(1, PATCH_SIZE):
            pt = np.round([y, x] - normal * i).astype(int)
            if not (0 <= pt[0] < gray_cpu.shape[0] and 0 <= pt[1] < gray_cpu.shape[1]):
                break
            if bw.get()[pt[0], pt[1]] < 0.5:
                back = pt
                break
        if forw is not None and back is not None:
            width_map[y, x] = np.linalg.norm(forw - back)
    return pruned, width_map

# ----------- DRY RUNS ---------------
timings = {}
_ = time_method(method_medial_dse, bw, name="Medial+DSE", dryrun=True)
_ = time_method(method_canny_profile, gray, bw, name="Canny+Profile", dryrun=True)
_ = time_method(method_pca_local, bw, name="PCA-Local", dryrun=True)
_ = time_method(method_profile_normal, gray, bw, name="Profile-Normal", dryrun=True)

# ----------- REAL RUNS --------------
pruned_med, w_med = time_method(method_medial_dse, bw, name="Medial+DSE", dryrun=False, timings=timings)
pruned_canny, w_canny = time_method(method_canny_profile, gray, bw, name="Canny+Profile", dryrun=False, timings=timings)
pruned_pca, w_pca = time_method(method_pca_local, bw, name="PCA-Local", dryrun=False, timings=timings)
pruned_prof, w_prof = time_method(method_profile_normal, gray, bw, name="Profile-Normal", dryrun=False, timings=timings)

# ----------- STATS ------------------
def width_stats(width_map, mask):
    ws = width_map[mask]
    return dict(
        mean = float(np.nanmean(ws)),
        std = float(np.nanstd(ws)),
        min = float(np.nanmin(ws)),
        max = float(np.nanmax(ws)),
        n = int(np.sum(mask))
    )

stats = {
    "Medial+DSE": width_stats(w_med, pruned_med),
    "Canny+Profile": width_stats(w_canny, pruned_canny),
    "PCA-Local": width_stats(w_pca, pruned_pca),
    "Profile-Normal": width_stats(w_prof, pruned_prof),
}

# ----------- SAVE METRICS -----------
from pathlib import Path
Path("width_metrics").mkdir(exist_ok=True)

with open("width_metrics.txt", "w") as f:
    f.write("Width Method Timing and Statistics\n\n")
    for k in stats:
        f.write(f"Method: {k}\n")
        f.write(f"  Timing: {timings[k]:.1f} ms\n")
        f.write(f"  Mean: {stats[k]['mean']:.2f} px\n")
        f.write(f"  Std:  {stats[k]['std']:.2f} px\n")
        f.write(f"  Min:  {stats[k]['min']:.2f} px\n")
        f.write(f"  Max:  {stats[k]['max']:.2f} px\n")
        f.write(f"  Num:  {stats[k]['n']}\n\n")
print("Saved: width_metrics.txt")

# ----------- PLOTTING ---------------
def plot_ribbon(pruned, width_map, title, filename):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(np.zeros_like(pruned), cmap='gray', vmin=0, vmax=1)  # Black background
    ax.imshow(pruned, cmap='gray', vmin=0, vmax=1, alpha=1.0)      # White skeleton/crack
    ys, xs = np.nonzero(pruned)
    vmin, vmax = np.nanmin(width_map[pruned]), np.nanmax(width_map[pruned])
    lines = []
    for y, x in zip(ys, xs):
        w = width_map[y, x]
        if not np.isfinite(w) or w <= 0: continue
        _, normal = get_local_tangent_normal(y, x, pruned, window=5)
        pt1 = [x - 0.5 * w * normal[1], y - 0.5 * w * normal[0]]
        pt2 = [x + 0.5 * w * normal[1], y + 0.5 * w * normal[0]]
        lines.append((w, pt1, pt2))
    lines.sort(key=lambda t: t[0])
    for w, pt1, pt2 in lines:
        color = plt.cm.plasma((w - vmin) / (vmax - vmin))
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=color, linewidth=2)
    ax.set_title(title)
    ax.axis('off')
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin, vmax))
    fig.colorbar(sm, ax=ax, label='Width (px)')
    plt.tight_layout()
    from pathlib import Path
    Path("plot_outputs").mkdir(exist_ok=True)
    filename = f"plot_outputs/{filename}"
    plt.savefig(filename, dpi=600)
    print(f"Saved: {filename}")
    plt.close(fig)

plot_ribbon(pruned_med, w_med, "Medial+DSE", "width_medial_dse.png")
plot_ribbon(pruned_canny, w_canny, "Canny+Profile", "width_canny_profile.png")
plot_ribbon(pruned_pca, w_pca, "PCA-Local", "width_pca_local.png")
plot_ribbon(pruned_prof, w_prof, "Profile-Normal", "width_profile_normal.png")
print("\nAll methods completed and saved.")
