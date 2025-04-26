import os
import glob
import time
import torch
import numpy as np
import cv2
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import subprocess

# Configurations
MODELS = {
    "WaveMixSR": {
        "path": "models/WaveMixSR/WaveMixSRV2.py",
        "cmd": "python {script} --input {inpath} --output {outpath}"
    },
    "EdgeSRGAN": {
        "path": "models/EdgeSRGAN/test.py",
        "cmd": "python {script} --config models/EdgeSRGAN/config.yaml --input {inpath} --output {outpath}"
    },
    "RealESRGAN": {
        "path": "models/Real-ESRGAN/inference_realesrgan.py",
        "cmd": "python {script} -n RealESRGAN_x4plus -i {inpath} -o {outpath}"
    },
    "EDSR": {
        "path": "models/EDSR/test.py",
        "cmd": "python {script} --input {inpath} --output {outpath} --model_path models/EDSR/models/EDSR_x2.pth"
    },
    "FSRCNN": {
        "path": "models/FSRCNN/test.py",
        "cmd": "python {script} --input {inpath} --output {outpath} --model models/FSRCNN/models/FSRCNN_x2.pth"
    }
}

INPUT_DIR = "data/crack_high_res"
LR_DIR = "data/crack_low_res"
OUTPUT_DIR = "results"
os.makedirs(LR_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Generate 2x bicubic downscaled inputs
for hr_path in glob.glob(os.path.join(INPUT_DIR, "*.[jp][pn]g")):
    img = Image.open(hr_path).convert("RGB")
    lr = img.resize((img.width//2, img.height//2), Image.BICUBIC)
    lr_path = os.path.join(LR_DIR, os.path.basename(hr_path))
    lr.save(lr_path)

# 2. Benchmark loop
results = []
for name, cfg in MODELS.items():
    model_times = []
    metrics = []
    for lr_path in glob.glob(os.path.join(LR_DIR, "*.[jp][pn]g")):
        out_subdir = os.path.join(OUTPUT_DIR, name)
        os.makedirs(out_subdir, exist_ok=True)
        out_path = os.path.join(out_subdir, os.path.basename(lr_path))

        # Run model inference
        cmd = cfg["cmd"].format(script=cfg["path"], inpath=lr_path, outpath=out_path)
        start = time.time()
        subprocess.run(cmd, shell=True, check=True)
        elapsed = time.time() - start
        model_times.append(elapsed)

        # Compute PSNR & SSIM
        sr = np.array(Image.open(out_path).convert("RGB"))
        hr = np.array(Image.open(os.path.join(INPUT_DIR, os.path.basename(lr_path))).convert("RGB"))
        # Ensure same size
        sr = cv2.resize(sr, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_CUBIC)
        metrics.append({
            "psnr": psnr(hr, sr, data_range=255),
            "ssim": ssim(hr, sr, multichannel=True)
        })

    # Aggregate
    avg_time = np.mean(model_times)
    avg_psnr = np.mean([m["psnr"] for m in metrics])
    avg_ssim = np.mean([m["ssim"] for m in metrics])
    results.append((name, avg_time, avg_psnr, avg_ssim))

# 3. Print summary
print(f"{'Model':<12} {'Time(s)':>8} {'PSNR':>8} {'SSIM':>8}")
for name, t, p, s in results:
    print(f"{name:<12} {t:>8.3f} {p:>8.3f} {s:>8.3f}")