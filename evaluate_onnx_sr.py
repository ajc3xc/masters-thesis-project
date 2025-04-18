import os
os.environ['ORT_DISABLE_ARENA'] = '1'
import numpy as np
import onnxruntime as ort
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchvision.transforms.functional import to_tensor, to_pil_image
import torch
from tqdm import tqdm

# === CONFIGURATION ===
dataset_dir = '/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/superres_benchmarks/Set5/Set5'
scale = 2
onnx_model_path = '/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/models/sr_block_2x.onnx'

hr_dir = os.path.join(dataset_dir, 'original')
lr_dir = os.path.join(dataset_dir, f'LRbicx{scale}')

# === LOAD ONNX MODEL ===
print(f"🧠 Loading ONNX model from {onnx_model_path}")
session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

def pad_tensor_to_multiple(t: torch.Tensor, multiple: int = 252) -> torch.Tensor:
    _, _, h, w = t.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    return torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), mode='reflect')

def run_onnx_superres(lr_img: Image.Image):
    img = to_tensor(lr_img).unsqueeze(0)  # (1, 3, H, W)
    img = torch.nn.functional.interpolate(img, size=(256, 256), mode='bicubic', align_corners=False)

    output = session.run(None, {'input': img.numpy().astype(np.float32)})[0]
    out_img = np.clip(output.squeeze(0), 0, 1)
    return to_pil_image(torch.from_numpy(out_img))

# === EVALUATION LOOP ===
psnr_scores = []
ssim_scores = []

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

    sr_img = run_onnx_superres(lr_img)

    if sr_img.size != hr_img.size:
        sr_img = sr_img.resize(hr_img.size, resample=Image.BICUBIC)

    sr_np = np.array(sr_img)
    hr_np = np.array(hr_img)

    psnr_scores.append(psnr(hr_np, sr_np, data_range=255))
    ssim_scores.append(ssim(hr_np, sr_np, channel_axis=2, data_range=255))

# === RESULTS ===
print(f"\n✅ Results on {os.path.basename(dataset_dir)} (x{scale}):")
print(f"🔹 Average PSNR: {np.mean(psnr_scores):.2f} dB")
print(f"🔹 Average SSIM: {np.mean(ssim_scores):.4f}")