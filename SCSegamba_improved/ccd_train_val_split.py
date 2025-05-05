import os
import shutil
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

CCD_ORIG_TRAIN = Path("/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/data/crack_segmentation_unzipped/crack_segmentation/virginia_tech_concrete_crack_congolmeration/Conglomerate Concrete Crack Detection/Conglomerate Concrete Crack Detection/Train")  # existing train dir
NEW_TRAIN_DIR  = Path("/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/data/combined_dataset/TUT_Conglomerate_Concrete/train")
VAL_DIR        = Path("/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/data/combined_dataset/TUT_Conglomerate_Concrete/val")

# Create destination folders
for d in [NEW_TRAIN_DIR, VAL_DIR]:
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "masks").mkdir(parents=True, exist_ok=True)

# Prepare image list
image_list = sorted((CCD_ORIG_TRAIN / "images").glob("*"))
random.seed(42)
random.shuffle(image_list)

# 1/9th goes to validation
val_size = len(image_list) // 9
val_images = set(image_list[:val_size])

def copy_pair(img_path: Path):
    name = img_path.name
    mask_path = CCD_ORIG_TRAIN / "masks" / name
    target_base = VAL_DIR if img_path in val_images else NEW_TRAIN_DIR
    dst_img = target_base / "images" / name
    dst_mask = target_base / "masks" / name

    # 🛑 Skip if already exists
    if dst_img.exists() and dst_mask.exists():
        return

    try:
        shutil.copy2(img_path, dst_img)
        shutil.copy2(mask_path, dst_mask)
    except Exception as e:
        print(f"[!] Error copying {name}: {e}")

# Run in parallel
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(copy_pair, p) for p in image_list]
    for f in as_completed(futures):
        f.result()  # re-raise errors if any

print(f"[✓] Split complete — Train: {len(image_list) - val_size}, Val: {val_size}")