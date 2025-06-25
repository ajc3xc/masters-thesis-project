import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import tempfile
from ridge_detection.lineDetector import LineDetector
from ridge_detection.params import Params
from skimage.filters import gabor
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from time import perf_counter

# ---- USER SETTINGS ----
IMG_PATH = '/mnt/e/camerer_ml/datasets/METU_Concrete_Crack_Seg/BW/236.JPG'
THRESHOLD = 0.25
# Steger params (edit as needed!)
STEGER_SIGMA = 3.39
STEGER_LOWER = 0.34
STEGER_UPPER = 1.02

# ---- PREPARE CONFIG FILE AS STRING ----
config = {
    "path_to_file": IMG_PATH,
    "mandatory_parameters": {
        "Sigma": STEGER_SIGMA,
        "Lower_Threshold": STEGER_LOWER,
        "Upper_Threshold": STEGER_UPPER,
        "Maximum_Line_Length": 0,
        "Minimum_Line_Length": 0,
        "Darkline": "LIGHT",
        "Overlap_resolution": "NONE"
    },
    "optional_parameters": {
        "Line_width": 10.0,
        "High_contrast": 200,
        "Low_contrast": 80
    },
    "further_options": {
        "Correct_position": True,
        "Estimate_width": True,
        "doExtendLine": True,
        "Show_junction_points": True,
        "Show_IDs": False,
        "Display_results": False,
        "Preview": False,
        "Make_Binary": False,
        "save_on_disk": False
    }
}

# ---- WRITE CONFIG TO TEMP FILE ----
with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as f:
    json.dump(config, f)
    f.flush()
    config_path = f.name

# ---- RUN STEGER DETECTOR ----
print("Running Steger detector...")
gray = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE) / 255.0

params = Params(config_path)
ld = LineDetector(params)
# This returns: detected_lines, width_map, angle_map, confidence_map
result = ld.detectLines(gray)
skel = result[0]
width = result[1]

# ---- GABOR FILTER FOR REFERENCE ----
print("Running Gabor filter...")
filt_real, filt_imag = gabor(gray, frequency=0.1, theta=0)
mask = filt_real > np.percentile(filt_real, 90)
import cupy as cp
from cucim.skimage.morphology import medial_axis
from dsepruning.dsepruning import skel_pruning_DSE
MIN_AREA_PX = 1000  # Minimum area for DSE pruning
def width_medial_dse(bw):
    """Medial‐axis + EDT + DSE pruning."""
    # choose GPU or CPU medial axis
    if True:
        print("Using GPU for medial axis...")
        sk_gpu, dist_gpu = medial_axis(cp.array(bw), return_distance=True)
        sk, dist = sk_gpu.get(), dist_gpu.get()
    else:
        sk, dist = medial_axis(bw, return_distance=True)
    pruned = skel_pruning_DSE(sk, dist, min_area_px=MIN_AREA_PX, return_graph=False)
    w = np.zeros_like(bw, float)
    w[pruned] = dist[pruned] * 2
    return w, pruned

print("Calculating width using medial axis + DSE pruning...")
sk, dist = width_medial_dse(mask, min_area_px=MIN_AREA_PX)
w_gabor = dist * 2

# ---- VISUALIZATION ----
plt.figure(figsize=(12,4))
for i, (name, skm, wmap) in enumerate([
    ("Steger", skel > 0, width),
    ("Gabor-sketch", sk, w_gabor),
]):
    ax = plt.subplot(1,2,i+1)
    ax.imshow(gray, cmap='gray')
    ys, xs = np.nonzero(skm)
    sc = ax.scatter(xs, ys, c=wmap[ys, xs], cmap='plasma', s=10)
    ax.set_title(name)
    plt.colorbar(sc, ax=ax, label='Width (px)')
plt.tight_layout()
plt.show()
