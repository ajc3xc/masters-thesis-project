from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import shutil

label_dir = Path("D:/camerer_ml/datasets/Final-Dataset-Vol1/labels")
fixed_dir = label_dir.parent / "labels_fixed"
fixed_dir.mkdir(exist_ok=True)

def process_mask(p: Path):
    img = Image.open(p).convert("L")
    min_val, max_val = img.getextrema()

    out_path = fixed_dir / p.name

    if min_val == 0 and max_val in (0,255):
        #shutil.copy(p, out_path)
        #print(f"✅ {p.name} (unchanged)")
        return None
    else:
        #Threshold everything non-zero to 255
        img_bin = img.point(lambda x: 255 if x > 0 else 0)
        #img_bin.save(out_path)
        print(f"⚠️  {p.name} (fixed: min={min_val}, max={max_val})")
        return p.stem

with ThreadPoolExecutor() as executor:
    results = list(executor.map(process_mask, label_dir.glob("*.png")))

cleaned_list = [x for x in results if x is not None]
print(len(cleaned_list))