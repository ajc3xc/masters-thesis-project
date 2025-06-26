# crackpy_test.py
import numpy as np
import matplotlib.pyplot as plt
from crackpy.fracture_analysis.crack_tip import CrackTip
from skimage.io import imread

# 1) Load your pre-computed DIC displacement fields (here we fake it from mask):
RGB_PATH = '/mnt/e/camerer_ml/datasets/METU_Concrete_Crack_Seg/rgb/236.JPG'
BW_PATH  = '/mnt/e/camerer_ml/datasets/METU_Concrete_Crack_Seg/BW_visible/236.png'
img = imread(RGB_PATH, as_gray=True)
mask = imread(BW_PATH, as_gray=True) > 0.25
# Normally, CrackPy expects u_x, u_y from DIC—here we derive a synthetic opening:
ys, xs = np.nonzero(mask)
dist = np.hypot(xs - xs.min(), ys - ys.min())
u_y = dist * 0.01  # synthetic opening as a function of distance

# 2) Initialize CrackTip analysis
ctp = CrackTip(
    crack_tip=[xs.min(), ys.min()],
    crack_path=np.column_stack((xs, ys)),
    u_x=np.zeros_like(u_y),
    u_y=u_y
)

# 3) Fit K and plot COD
ctp.plot_cod_profile(show=True)
print("Estimated K_I:", ctp.K_I)
