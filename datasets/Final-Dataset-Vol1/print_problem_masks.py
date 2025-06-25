from pathlib import Path
from PIL import Image
import numpy as np

label_dir = Path("D:/camerer_ml/datasets/Final-Dataset-Vol1/labels_fixed")
allowed_values = {0, 1, 255}

bad = []

for p in label_dir.glob("*.png"):
    img = Image.open(p).convert("L")
    vals = set(np.unique(np.array(img)))
    if not vals.issubset(allowed_values):
        bad.append((p.name, sorted(vals)))

print(f"❌ Found {len(bad)} problematic masks:")
#for name, vals in bad:
#    print(f"{name}: {len(vals)}")
