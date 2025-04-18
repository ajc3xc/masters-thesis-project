import os
import numpy as np
import torch
import onnxruntime as ort
from datasets import load_dataset
from super_image.data import EvalDataset
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

os.environ['HF_DATASETS_CACHE'] = './superres_benchmarks_x2/hf_cache'

# === CONFIGURATION ===
onnx_model_path = '/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/models/sr_block_2x.onnx'  # path to your ONNX model
scale = 2
dataset_name = 'Set5'
benchmark_name = f'eugenesiow/{dataset_name}'
hf_variant = f'bicubic_x{scale}'  # 'bicubic_x2', 'bicubic_x3', etc.

# === LOAD DATASET ===
print(f"📦 Loading {benchmark_name} at {hf_variant}")
dataset = load_dataset(
    'eugenesiow/Set5',
    'bicubic_x2',
    split='validation',
    cache_dir='./superres_benchmarks_x2/hf_cache',
    download_mode='force_redownload'
)
print(dataset[0]['lr'])  # Should be inside ./superres_benchmarks_x2/hf_cache
eval_dataset = EvalDataset(dataset)

# === LOAD ONNX MODEL ===
print(f"🧠 Loading ONNX model from {onnx_model_path}")
session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'])

# === INFERENCE FUNCTION ===
def run_onnx_superres(lr_img: Image.Image):
    # Convert to normalized tensor and NCHW format
    img = to_tensor(lr_img).unsqueeze(0).numpy().astype(np.float32)  # shape: (1, 3, H, W)
    output = session.run(None, {'input': img})[0]  # returns (1, 3, H*scale, W*scale)
    out_img = torch.from_numpy(output.squeeze(0)).clamp(0, 1)
    return to_pil_image(out_img)

# === EVALUATION LOOP ===
psnr_scores = []
ssim_scores = []

print("🚀 Running inference and computing metrics...")
for sample in tqdm(eval_dataset):
    lr_img = sample['input']
    hr_img = sample['target']

    sr_img = run_onnx_superres(lr_img)

    # Resize if dimensions mismatch (some models may produce slightly off sizes)
    if sr_img.size != hr_img.size:
        sr_img = sr_img.resize(hr_img.size, resample=Image.BICUBIC)

    sr_np = np.array(sr_img)
    hr_np = np.array(hr_img)

    psnr_scores.append(psnr(hr_np, sr_np, data_range=255))
    ssim_scores.append(ssim(hr_np, sr_np, multichannel=True, data_range=255))

# === RESULTS ===
print(f"\n📊 Results on {dataset_name} (x{scale}):")
print(f"🔹 Average PSNR: {np.mean(psnr_scores):.2f} dB")
print(f"🔹 Average SSIM: {np.mean(ssim_scores):.4f}")
