import torch
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, medial_axis
from skimage.measure import label
from scipy.ndimage import convolve, distance_transform_edt

def count_connected_components(skel):
    lbl = label(skel)
    return lbl.max()

def count_endpoints(skel):
    kernel = np.array([[1,1,1],[1,10,1],[1,1,1]])
    filtered = convolve(skel.astype(np.uint8), kernel, mode='constant')
    return np.sum(filtered == 11)

def mean_thickness(skel_img):
    if np.sum(skel_img) == 0:
        return 0
    dist = distance_transform_edt(skel_img)
    return dist[skel_img > 0].mean()

def run_skelite(model, img_tensor):
    with torch.no_grad():
        output, _ = model(img_tensor, z=None, no_iter=-1, val_mode=True)
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

# --- Main script ---
image_path = r"D:\camerer_ml\datasets\METU_Concrete_Crack_Seg\BW_visible\561.png"
model_path = r'D:\camerer_ml\adaptive_pca_3dreproject\Skelite\skelite_scripted.pt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.jit.load(model_path, map_location=device)
model.eval()

img_tensor, img = load_and_prep_image(image_path, device)
bw = img > 0.5

# Classic thinning (strict skeleton)
skel_skimage = skeletonize(bw)
# Medial axis (centerline, sometimes with extra spurs)
#skel_medial = medial_axis(bw)
# Skelite output (as mask)
skelite_mask, skel_skelite = run_skelite(model, img_tensor)
# Thinning Skelite output
skel_skelite_thin = skeletonize(skel_skelite)

def print_metrics(name, skel):
    print(f"\n{name} Skeleton:")
    print("  Connected components:", count_connected_components(skel))
    print("  Endpoints:", count_endpoints(skel))
    print("  Mean thickness:", mean_thickness(skel))
    print("  Skeleton pixel count:", np.sum(skel))

# --- Print metrics ---
print_metrics('skimage', skel_skimage)
#print_metrics('medial axis', skel_medial)
print_metrics('Skelite mask >0.5', skel_skelite)
print_metrics('Skelite+Thinning', skel_skelite_thin)

# --- Visual comparison ---
plt.figure(figsize=(18, 4))
titles = [
    'Original', 
    'skimage.skeletonize', 
    'Skelite Mask (>0.5)', 
    'Skelite+Thinning'
]
images = [
    img, 
    skel_skimage, 
    #skel_medial, 
    skel_skelite, 
    skel_skelite_thin
]

for i in range(len(images)):
    plt.subplot(1, len(images), i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()