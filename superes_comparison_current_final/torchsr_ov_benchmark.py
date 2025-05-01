import os
os.environ['ORT_DISABLE_ARENA'] = '1'

from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import numpy as np
from tqdm import tqdm
from PIL import Image
from openvino.runtime import Core
import torch
from torchvision.transforms.functional import to_tensor, to_pil_image

# === CONFIGURATION ===
dataset_dir = '/mnt/c/Users/13144/Documents/Masters_Thesis/super_resolution/Set14'
scale = 2
onnx_model_path = '/mnt/c/Users/13144/Documents/Masters_Thesis/super_resolution/superes_comparison_current_final/models/wavemixsrv2_srblock_2x_fullflex.onnx'
hr_dir = os.path.join(dataset_dir, 'original')
lr_dir = os.path.join(dataset_dir, f'LRbicx{scale}')

# === OPENVINO SETUP ===
print(f"🧠 Compiling OpenVINO model from {onnx_model_path}…")
core = Core()
ov_model = core.read_model(onnx_model_path)
device_name = "GPU" if "GPU" in core.available_devices else "CPU"
compiled_model = core.compile_model(ov_model, device_name)
inp = compiled_model.input(0)
outp = compiled_model.output(0)
print(f"✅ OpenVINO model ready on {device_name}")

# === TORCH MODELS ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    from torchsr.models import edsr, rcan, ninasr_b1, carn
    edsr_model = edsr(scale=scale, pretrained=True).eval().to(device)
    rcan_model = rcan(scale=scale, pretrained=True).eval().to(device)
    ninasr_model = ninasr_b1(scale=scale, pretrained=True).eval().to(device)
    carn_model = carn(scale=scale, pretrained=True).eval().to(device)
    has_torchsr = True
except ImportError:
    has_torchsr = False
    print("⚠️ torchsr not installed. Skipping EDSR/RCAN/NinaSR/CARN.")

try:
    from torchvision.models import fsrcnn_x2
    fsrcnn_model = fsrcnn_x2(pretrained=True).eval().to(device)
    has_fsrcnn = True
except ImportError:
    has_fsrcnn = False
    print("⚠️ torchvision fsrcnn_x2 not installed. Skipping FSRCNN.")

# === HELPERS ===
def to_y_channel(img: Image.Image) -> np.ndarray:
    arr = np.array(img).astype(np.float32)
    y = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return np.clip(y, 0, 255).astype(np.uint8)

def run_openvino(lr_img: Image.Image) -> Image.Image:
    arr = np.array(lr_img).astype(np.float32) / 255.0   # HWC [0,1]
    input_tensor = np.transpose(arr, (2, 0, 1))[None]   # NCHW
    result = compiled_model([input_tensor])[outp]       # shape (1,C,H,W)
    out = np.clip(result[0], 0, 1)
    out_hwc = np.transpose(out, (1, 2, 0))              # HWC [0,1]
    return Image.fromarray((out_hwc * 255).round().astype(np.uint8))

def run_bicubic(lr_img: Image.Image, target_size) -> Image.Image:
    return lr_img.resize(target_size, Image.BICUBIC)

def run_lanczos(lr_img: Image.Image, target_size) -> Image.Image:
    return lr_img.resize(target_size, Image.LANCZOS)

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

# === BENCHMARK LOOP ===
methods = {
    "OpenVINO_SR": run_openvino,
    "Bicubic":  run_bicubic,
    "Lanczos":  run_lanczos
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

results = {name: {"psnr": [], "ssim": []} for name in methods}
files = sorted(os.listdir(hr_dir))
print(f"📊 Evaluating {len(files)} images…")

for fname in tqdm(files):
    hr_img = Image.open(os.path.join(hr_dir, fname)).convert("RGB")
    lr_img = Image.open(os.path.join(lr_dir, fname)).convert("RGB")
    target_size = hr_img.size
    hr_y = to_y_channel(hr_img)

    for name, fn in methods.items():
        if name in ("Bicubic", "Lanczos"):
            sr_img = fn(lr_img, target_size)
        else:
            sr_img = fn(lr_img)

        if sr_img.size != hr_img.size:
            sr_img = sr_img.resize(hr_img.size, Image.BICUBIC)

        sr_y = to_y_channel(sr_img)
        results[name]["psnr"].append(psnr(hr_y, sr_y, data_range=255))
        results[name]["ssim"].append(ssim(hr_y, sr_y, data_range=255))

# === RESULTS ===
print("\n✅ Benchmark Results (Y-channel only):")
for name, scores in results.items():
    print(f"🔹 {name}: PSNR={np.mean(scores['psnr']):.2f} dB | SSIM={np.mean(scores['ssim']):.4f}")