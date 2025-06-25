import cupy as cp
from cucim.skimage.morphology import thin
import time

# Create a simple binary cross image (as before)
arr = cp.zeros((16, 16), dtype=cp.bool_)
arr[8, 3:13] = True
arr[3:13, 8] = True

print("Input array:")
print(cp.asnumpy(arr).astype(int))

# Try GPU thinning (skeletonization)
try:
    t0 = time.time()
    skel = thin(arr)
    cp.cuda.Stream.null.synchronize()
    t1 = time.time()
    print("\nSkeletonized result:")
    print(cp.asnumpy(skel).astype(int))
    print(f"\n[GPU] cucim.thin took {t1 - t0:.4f} seconds")
except Exception as e:
    print("cuCIM test failed!")
    print(e)
