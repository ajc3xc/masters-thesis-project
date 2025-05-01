import os
os.environ['ORT_DISABLE_ARENA'] = '1'
# preload libstdc++ if Conda exists
print("🔧 Preloading newer libstdc++.so.6 if available...")
conda_prefix = os.environ.get('CONDA_PREFIX')
if conda_prefix:
    candidate = os.path.join(conda_prefix, 'lib', 'libstdc++.so.6')
    if os.path.exists(candidate):
        os.environ['LD_PRELOAD'] = candidate + ':' + os.environ.get('LD_PRELOAD', '')
        print(f"✅ Preloaded {candidate}")
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
#import sys
#sys.exit()
import numpy as np
from tqdm import tqdm
from PIL import Image
import onnxruntime as ort
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import to_tensor, to_pil_image

# Optional models for comparison
try:
    from torchsr.models import edsr, rcan, ninasr_b1, carn
    has_torchsr = True
except ImportError:
    has_torchsr = False
    print("⚠️ torchsr not installed. Skipping EDSR/RCAN/NinaSR/CARN.")

try:
    from torchvision.models import fsrcnn_x2
    has_fsrcnn = True
except ImportError:
    has_fsrcnn = False
    print("⚠️ torchvision fsrcnn_x2 not installed. Skipping FSRCNN.")

# === CONFIGURATION ===
dataset_dir = '/mnt/c/Users/13144/Documents/Masters_Thesis/super_resolution/Set14'
scale = 2
onnx_model_path = '/mnt/c/Users/13144/Documents/Masters_Thesis/super_resolution/superes_comparison_current_final/models/sr_block_2x_flexible.onnx'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hr_dir = os.path.join(dataset_dir, 'original')
lr_dir = os.path.join(dataset_dir, f'LRbicx{scale}')

# === LOAD MODELS ===
print(f"🧠 Loading ONNX model from {onnx_model_path}")
onnx_session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

if has_fsrcnn:
    print("🧠 Loading FSRCNN model from torchvision.models")
    fsrcnn_model = fsrcnn_x2(pretrained=True).eval().to(device)

if has_torchsr:
    print("🧠 Loading EDSR model from torchSR")
    edsr_model = edsr(scale=scale, pretrained=True).eval().to(device)

    print("🧠 Loading RCAN model from torchSR")
    rcan_model = rcan(scale=scale, pretrained=True).eval().to(device)

    print("🧠 Loading NinaSR-B1 model from torchSR")
    ninasr_model = ninasr_b1(scale=scale, pretrained=True).eval().to(device)

    print("🧠 Loading CARN model from torchSR")
    carn_model = carn(scale=scale, pretrained=True).eval().to(device)

# === METHOD WRAPPERS ===
def run_onnx_superres(lr_img: Image.Image):
    img = to_tensor(lr_img).unsqueeze(0).numpy().astype(np.float32)  # (1, 3, H, W)
    output = onnx_session.run(None, {'input': img})[0]
    out_img = np.clip(output.squeeze(0), 0, 1)
    return to_pil_image(torch.from_numpy(out_img))

def run_fsrcnn(lr_img: Image.Image):
    img = to_tensor(lr_img).unsqueeze(0).to(device)
    with torch.no_grad():
        sr_img = fsrcnn_model(img).clamp(0, 1)
    return to_pil_image(sr_img.squeeze(0).cpu())

def run_edsr(lr_img: Image.Image):
    img = to_tensor(lr_img).unsqueeze(0).to(device)
    with torch.no_grad():
        sr_img = edsr_model(img).clamp(0, 1)
    return to_pil_image(sr_img.squeeze(0).cpu())

def run_rcan(lr_img: Image.Image):
    img = to_tensor(lr_img).unsqueeze(0).to(device)
    with torch.no_grad():
        sr_img = rcan_model(img).clamp(0, 1)
    return to_pil_image(sr_img.squeeze(0).cpu())

def run_ninasr(lr_img: Image.Image):
    img = to_tensor(lr_img).unsqueeze(0).to(device)
    with torch.no_grad():
        sr_img = ninasr_model(img).clamp(0, 1)
    return to_pil_image(sr_img.squeeze(0).cpu())

def run_carn(lr_img: Image.Image):
    img = to_tensor(lr_img).unsqueeze(0).to(device)
    with torch.no_grad():
        sr_img = carn_model(img).clamp(0, 1)
    return to_pil_image(sr_img.squeeze(0).cpu())

def run_bicubic(lr_img: Image.Image, target_size):
    return lr_img.resize(target_size, Image.BICUBIC)

def run_lanczos(lr_img: Image.Image, target_size):
    return lr_img.resize(target_size, Image.LANCZOS)

# === EVALUATION LOOP ===
methods = {
    "ONNX_SR": run_onnx_superres,
    "Bicubic": run_bicubic,
    "Lanczos": run_lanczos
}
if has_fsrcnn:
    methods["FSRCNN"] = run_fsrcnn
if has_torchsr:
    methods.update({
        "EDSR": run_edsr,
        "RCAN": run_rcan,
        "NinaSR-B1": run_ninasr,
        "CARN": run_carn
    })

results = {method: {"psnr": [], "ssim": []} for method in methods.keys()}

hr_filenames = sorted(os.listdir(hr_dir))
print(f"📊 Evaluating {len(hr_filenames)} images...")

for fname in tqdm(hr_filenames):
    hr_path = os.path.join(hr_dir, fname)
    lr_path = os.path.join(lr_dir, fname)

    if not os.path.exists(lr_path):
        print(f"⚠️ Skipping missing LR file: {fname}")
        continue

    hr_img = Image.open(hr_path).convert('RGB')
    lr_img = Image.open(lr_path).convert('RGB')
    target_size = hr_img.size

    for method_name, method_func in methods.items():
        if method_name in ["Bicubic", "Lanczos"]:
            sr_img = method_func(lr_img, target_size)
        else:
            sr_img = method_func(lr_img)

        if sr_img.size != hr_img.size:
            sr_img = sr_img.resize(hr_img.size, resample=Image.BICUBIC)

        sr_np = np.array(sr_img)
        hr_np = np.array(hr_img)

        results[method_name]["psnr"].append(psnr(hr_np, sr_np, data_range=255))
        results[method_name]["ssim"].append(ssim(hr_np, sr_np, channel_axis=2, data_range=255))

# === RESULTS ===
print("\n✅ Benchmark Results:")
for method_name, scores in results.items():
    avg_psnr = np.mean(scores["psnr"])
    avg_ssim = np.mean(scores["ssim"])
    print(f"🔹 {method_name}: PSNR={avg_psnr:.2f} dB | SSIM={avg_ssim:.4f}")
