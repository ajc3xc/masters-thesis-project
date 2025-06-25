import tifffile
import numpy as np
import matplotlib.pyplot as plt
from read_roi import read_roi_zip
from pathlib import Path

# ---- HARDCODED PATHS ----
input_dir = Path(r"C:\Users\ajc3xc\Downloads\krkCMd_images")  # Update this to your actual folder
output_dir = Path("roi_visualizations")
output_dir.mkdir(exist_ok=True)

# Find all .tif images in the input folder
image_paths = sorted(input_dir.rglob("*.tif"))

#print(image_paths[0].parent)
#import sys; sys.exit()

for img_path in image_paths:
    img_name = img_path.stem
    roi_name = "ROI" + img_name[3:] + ".zip"
    roi_zip = img_path.parent / roi_name
    #print(roi_zip)
    if not roi_zip.exists():
        print(f"WARNING: No ROI zip for {img_name}")
        continue

    # Load image
    img = tifffile.imread(str(img_path))
    if img.ndim == 4:
        img = img[0]
        print(f"Selected first page, new shape: {img.shape}")
    if img.ndim == 3:
        if img.shape[0] <= 4:  # (C, H, W)
            img = np.transpose(img, (1, 2, 0))
        elif img.shape[2] > 4:
            img = img[:, :, :3]
    elif img.ndim == 2:
        img = np.stack([img]*3, axis=-1)

    img_disp = img.astype(np.float32)
    img_disp -= img_disp.min()
    img_disp /= max(img_disp.max(), 1e-6)

    # Load ROIs
    rois = read_roi_zip(str(roi_zip))

    # Plot image and overlay ROIs
    plt.figure(figsize=(8, 8))
    plt.imshow(img_disp, cmap="gray")
    for roi_name, roi in rois.items():
        if 'type' in roi and roi['type'] == 'line':
            plt.plot([roi['x1'], roi['x2']], [roi['y1'], roi['y2']], 'r-', linewidth=1.5)
        elif 'x' in roi and 'y' in roi:  # Polygon, rectangle, etc.
            plt.plot(roi['x'], roi['y'], 'r-', linewidth=1.5)
        else:
            print(f"Unknown ROI type in {roi_name}")

    plt.axis('off')
    out_file = output_dir / f"{img_name}_roi.png"
    plt.savefig(out_file, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close()
    print(f"Saved: {out_file}")