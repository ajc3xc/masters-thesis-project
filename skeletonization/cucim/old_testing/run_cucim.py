import time
import cupy as cp
from cucim.skimage.morphology import medial_axis as gpu_thin
import imageio.v3 as iio
from pathlib import Path
from skimage.morphology import skeletonize as cpu_thin

# Load binary image as NumPy
input_path = Path("/mnt/d/camerer_ml/skeletonization/skeletonide/test/images/horse.png")
img_np = iio.imread(input_path) < 128

# GPU: "warmup" to compile kernels
img_gpu = cp.array(img_np)
_ = gpu_thin(img_gpu)
cp.cuda.Stream.null.synchronize()

# Actual GPU timing
t0 = time.time()
sk_gpu = gpu_thin(img_gpu)
cp.cuda.Stream.null.synchronize()
t_gpu = time.time() - t0
print(f"[GPU] cucim.thin: {t_gpu:.4f} s")

# CPU timing
t0 = time.time()
sk_cpu = cpu_thin(img_np)
t_cpu = time.time() - t0
print(f"[CPU] skimage.thin: {t_cpu:.4f} s")

# Convert and save both
skel_gpu = cp.asnumpy(sk_gpu).astype("uint8") * 255
skel_cpu = sk_cpu.astype("uint8") * 255
out = Path("./results")
out.mkdir(exist_ok=True)
iio.imwrite(out / "skel_gpu.png", skel_gpu)
iio.imwrite(out / "skel_cpu.png", skel_cpu)
print("Results saved to ./results/")

#/mnt/d/camerer_ml/skeletonization/skeletonide/test/images/horse.png