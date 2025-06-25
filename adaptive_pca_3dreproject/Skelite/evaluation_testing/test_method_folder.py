import torch
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, medial_axis
from pathlib import Path
import os

def run_skelite(model, img_tensor):
    with torch.no_grad():
        output, _ = model(img_tensor, z=None, no_iter=5)
    mask = output[0,0].cpu().numpy()
    skel_mask = mask > 0.5
    return mask, skel_mask

def load_and_prep_image(image_path, device):
    img = imageio.imread(image_path)
    if img.ndim == 3:
        img = img.mean(axis=2)
    img = img / 255.0
    tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    return tensor, img

def save_visual_comparison(image_name, img, skel_skimage, skel_medial, skel_skelite, skel_skelite_thin, save_dir):
    plt.figure(figsize=(18, 4))
    titles = [
        'Original', 
        'skimage.skeletonize', 
        'Medial Axis', 
        'Skelite Mask (>0.5)', 
        'Skelite+Thinning'
    ]
    images = [
        img, 
        skel_skimage * 1.0, 
        skel_medial * 1.0, 
        skel_skelite * 1.0, 
        skel_skelite_thin * 1.0
    ]
    for i in range(len(images)):
        plt.subplot(1, len(images), i+1)
        plt.imshow(images[i], cmap='gray', vmin=0, vmax=1)
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    out_path = os.path.join(save_dir, f"{Path(image_name).stem}_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

# --- Batch processing setup ---
image_dir = r"D:\camerer_ml\datasets\METU_Concrete_Crack_Seg\BW_visible"  # folder of images
model_path = r'D:\camerer_ml\adaptive_pca_3dreproject\Skelite\skelite_scripted.pt'
save_dir = r"D:\camerer_ml\adaptive_pca_3dreproject\Skelite\skeleton_comparisons"
os.makedirs(save_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.jit.load(model_path, map_location=device)
model.eval()

all_image_paths = sorted(list(Path(image_dir).glob("*.png")))

i = 0
for img_path in all_image_paths:
    img_tensor, img = load_and_prep_image(str(img_path), device)
    bw = img > 0.5

    skel_skimage = skeletonize(bw)
    skel_medial = medial_axis(bw)
    _, skel_skelite = run_skelite(model, img_tensor)
    skel_skelite_thin = skeletonize(skel_skelite)
    print(i)
    i += 1

    save_visual_comparison(img_path.name, img, skel_skimage, skel_medial, skel_skelite, skel_skelite_thin, save_dir)