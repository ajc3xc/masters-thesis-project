from datasets import load_dataset
import os
from PIL import Image

root_dir = './superres_benchmarks_x2'
os.makedirs(root_dir, exist_ok=True)

dataset_names = ['Set5', 'Set14', 'BSD100', 'Urban100']
scale = 'bicubic_x2'

# Optional: specify a cache dir under your current working directory
hf_cache_dir = os.path.join(root_dir, 'hf_cache')
os.makedirs(hf_cache_dir, exist_ok=True)

for name in dataset_names:
    print(f"Downloading and saving {name}...")
    dataset = load_dataset(
        f'eugenesiow/{name}',
        scale,
        split='validation',
        cache_dir=hf_cache_dir,
        download_mode='force_redownload'  # Important
    )

    save_path = os.path.join(root_dir, name)
    hr_path = os.path.join(save_path, 'HR')
    lr_path = os.path.join(save_path, f'LR_{scale.upper()}')
    os.makedirs(hr_path, exist_ok=True)
    os.makedirs(lr_path, exist_ok=True)

    for i, sample in enumerate(dataset):
        hr_img = Image.open(sample['hr'])
        lr_img = Image.open(sample['lr'])

        hr_img.save(os.path.join(hr_path, f"{i:04d}.png"))
        lr_img.save(os.path.join(lr_path, f"{i:04d}.png"))

print("✅ All datasets downloaded and saved as images.")
