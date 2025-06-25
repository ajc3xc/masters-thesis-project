from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
import numpy as np
import cv2
import shutil

label_dir = Path("labels_broken")
fixed_dir = label_dir.parent / "labels"
fixed_visible_dir = label_dir.parent / "labels_visible"
fixed_dir.mkdir(exist_ok=True)
fixed_visible_dir.mkdir(exist_ok=True)

def is_rissbilder(p: Path) -> bool:
    # Adjust this function based on how Rissbilder masks are identified
    return 'rissbilder' in p.name.lower()

def process_mask(p: Path):
    img = Image.open(p).convert("L")
    img_np = np.array(img)

    unique_vals = np.unique(img_np)
    # Save as both 0/1 and 0/255 even if already clean
    if set(unique_vals).issubset({0, 255, 1}):
        img_01 = (img_np > 0).astype(np.uint8)
    else:
        non_zero = img_np[img_np > 0]
        if len(non_zero) == 0:
            img_01 = np.zeros_like(img_np, dtype=np.uint8)
        else:
            if is_rissbilder(p):
                mean_val = np.mean(non_zero)
                threshold = max(5, mean_val * 0.5)
            elif 'cracktree200' in p.name.lower():
                threshold = 40
            else:
                threshold = 127
            img_01 = np.where(img_np > threshold, 1, 0).astype(np.uint8)
            kernel = np.ones((3, 3), np.uint8)
            img_01 = cv2.morphologyEx(img_01, cv2.MORPH_CLOSE, kernel)

    # Save PaddleSeg version (0/1)
    out_path = fixed_dir / p.name
    Image.fromarray(img_01).save(out_path)
    # Save visible version (0/255)
    out_path_visible = fixed_visible_dir / p.name
    Image.fromarray(img_01 * 255).save(out_path_visible)
    print(f"Processed {p.name}")
    return p.stem

if __name__ == "__main__":
    pngs = list(label_dir.glob("*.png"))
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_mask, pngs))

    cleaned_list = [x for x in results if x is not None]
    print(f"{len(cleaned_list)} masks processed.")
