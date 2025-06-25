import numpy as np
from skimage.io import imread
from skimage.morphology import medial_axis, binary_dilation
from skimage.draw import line
import matplotlib.pyplot as plt
from time import time

from skan import Skeleton, summarize
import sknw
from dsepruning.dsepruning import skel_pruning_DSE

def skan_length_prune(skel, min_length=10):
    sk = Skeleton(skel)
    summary = summarize(sk)
    keep = summary['branch-distance'] > min_length
    mask = np.zeros_like(skel, bool)
    for idx, row in summary.iterrows():
        if keep[idx]:
            coords = sk.path_coordinates(idx)
            mask[tuple(coords.T)] = True
    return mask

def graph2im(graph, shape):
    mask = np.zeros(shape, dtype=bool)
    for s, e in graph.edges():
        vals = graph[s][e].values()
        for val in vals:
            coords = val['pts']
            coords_1 = np.roll(coords, -1, axis=0)
            for i in range(len(coords) - 1):
                rr, cc = line(*coords[i], *coords_1[i])
                mask[rr, cc] = True
            mask[tuple(graph.nodes[s]['pts'].T.tolist())] = True
            mask[tuple(graph.nodes[e]['pts'].T.tolist())] = True
    return mask

def count_branches(skel):
    """Rough count: endpoints + junctions, for illustration."""
    from scipy.ndimage import convolve
    kernel = np.array([[1,1,1],[1,10,1],[1,1,1]])
    vals = convolve(skel.astype(int), kernel, mode='constant')
    # Endpoints: value==11, Junctions: value>=13
    endpoints = np.sum((skel) & (vals == 11))
    junctions = np.sum((skel) & (vals >= 13))
    return endpoints, junctions

# ---------------- Main script ----------------
img_path = '/mnt/e/camerer_ml/datasets/METU_Concrete_Crack_Seg/BW_visible/236.png'  # <-- set your path here

# 1. Load and binarize
img = imread(img_path, mode='L')  # Load as grayscale
bw = (img > 0.5)

# 2. Skeletonize (medial axis)
start = time()
skeleton, dist = medial_axis(bw, return_distance=True)
time_skel = time() - start

# 3. SKAN length-prune
start = time()
pruned_skan = skan_length_prune(skeleton, min_length=10)
time_skan = time() - start

# 4. DSE prune (run on raw skeleton, or run on pruned_skan for a hybrid)
start = time()
pruned_dse = skel_pruning_DSE(skeleton, dist, min_area_px=500)
time_dse = time() - start

# --- Optional: run DSE on pruned_skan for a "hybrid" speedup ---
# start = time()
# pruned_skan_dse = skel_pruning_DSE(pruned_skan, dist, min_area_px=500)
# time_skan_dse = time() - start

# 5. Visualize
#dil = lambda x: binary_dilation(x, iterations=2)

from scipy.ndimage import distance_transform_edt, binary_dilation
def dilate(skel):
    dilated = binary_dilation(skel, iterations=2)
    return dilated

titles = ['Raw Skeleton', 'SKAN prune', 'DSE prune']
images = [dilate(skeleton), dilate(pruned_skan), dilate(pruned_dse)]
timings = [time_skel, time_skan, time_dse]
branch_counts = [count_branches(skeleton), count_branches(pruned_skan), count_branches(pruned_dse)]

plt.figure(figsize=(13,5))
for i, (img, title, t, (ep, ju)) in enumerate(zip(images, titles, timings, branch_counts)):
    plt.subplot(1, 3, i+1)
    plt.imshow(img, cmap='gray')
    plt.title(f"{title}\n{t:.2f} s, endpoints:{ep}, junctions:{ju}")
    plt.axis('off')
plt.suptitle("Skeleton Pruning: CPU vs SKAN vs DSE (timing/branch counts)")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.figure(figsize=(6,4))
plt.bar(titles, timings)
plt.ylabel("Time (s)")
plt.title("Pruning Timings")
plt.show()

print("Raw skeleton time:  {:.2f} s".format(time_skel))
print("SKAN prune time:    {:.2f} s".format(time_skan))
print("DSE prune time:     {:.2f} s".format(time_dse))
for title, (ep, ju) in zip(titles, branch_counts):
    print(f"{title}: Endpoints={ep}, Junctions={ju}")
