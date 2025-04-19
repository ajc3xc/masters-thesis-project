#!/usr/bin/env python3
"""
SR Benchmark Script — No CLI
Benchmarks WaveMixSR-V2 (ONNX) and BasicSR (EDSR, RCAN, RRDBNet) on a dataset folder.
"""

import os
import glob
import numpy as np
import torch
import onnxruntime as ort
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from basicsr.archs.edsr_arch import EDSR
from basicsr.archs.rcan_arch import RCAN
from basicsr.archs.rdn_arch import RRDBNet

# ================== CONFIGURATION ==================

scale = 2

dataset_dir = '/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/superres_benchmarks/Set14/Set14'
lr_dir = os.path.join(dataset_dir, f'LRbicx{scale}')
hr_dir = os.path.join(dataset_dir, 'original')
output_csv = f'sr_benchmark_set14_x{scale}.csv'

wavemix_onnx = '/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/models/sr_block_2x.onnx'

weights = {
    'EDSR':   ('EDSR_x2.pth',   'https://github.com/XPixelGroup/BasicSR/releases/download/v0.1.0/EDSR_x2.pth'),
    'RCAN':   ('RCAN_x2.pth',   'https://github.com/XPixelGroup/BasicSR/releases/download/v0.1.0/RCAN_x2.pth'),
    'RRDBNet':('RRDBNet_x2.pth','https://github.com/XPixelGroup/BasicSR/releases/download/v0.1.0/RRDBNet_x2.pth'),
}

checkpoints_dir = '/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/models'

# ============== HELPERS ==============

def download_if_missing(name, filename, url):
    path = os.path.join(checkpoints_dir, filename)
    if not os.path.exists(path):
        print(f"⬇️ Downloading {name} checkpoint...")
        urllib.request.urlretrieve(url, path)
        print(f"✅ Downloaded: {path}")
    return path

# ====================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_onnx(path):
    return ort.InferenceSession(
        path,
        providers=['CUDAExecutionProvider']
    )

def load_basic_model(cls, ckpt, scale):
    if cls is RRDBNet:
        model = RRDBNet(3, 3, 64, 23, 32, scale)
    elif cls is EDSR:
        model = EDSR(3, 3, 64, 16, scale)
    else:  # RCAN
        model = RCAN(3, 3, 64, 16, 8, scale)
    sd = torch.load(ckpt, map_location='cpu')
    model.load_state_dict(sd.get('params', sd))
    return model.eval().to(device)

def preprocess_onnx(pil):
    arr = np.array(pil).astype(np.float32)/255.0
    return arr.transpose(2,0,1)[None]

def postprocess_onnx(out):
    sr = np.clip(out.squeeze().transpose(1,2,0), 0, 1)
    return (sr*255.0).round().astype(np.uint8)

def preprocess_torch(pil):
    arr = np.array(pil).astype(np.float32)/255.0
    t = torch.from_numpy(arr.transpose(2,0,1))[None].to(device)
    return t

def postprocess_torch(tensor):
    arr = tensor.squeeze().clamp(0,1).permute(1,2,0).cpu().numpy()
    return (arr*255.0).round().astype(np.uint8)

def compute_metrics(hr_arr, sr_arr):
    return psnr(hr_arr, sr_arr, data_range=255), \
           ssim(hr_arr, sr_arr, channel_axis=2, data_range=255)

def main():
    wm_sess = load_onnx(wavemix_onnx)
    edsr = load_basic_model(EDSR, edsr_ckpt, scale)
    rcan = load_basic_model(RCAN, rcan_ckpt, scale)
    rrdb = load_basic_model(RRDBNet, rrdb_ckpt, scale)

    lr_paths = sorted(glob.glob(os.path.join(lr_dir, '*')))
    hr_map = {os.path.basename(p): p for p in glob.glob(os.path.join(hr_dir, '*'))}

    import csv
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model','Image','PSNR','SSIM'])

        for lr_path in lr_paths:
            name = os.path.basename(lr_path)
            if name not in hr_map:
                continue

            hr = Image.open(hr_map[name]).convert('RGB')
            lr = Image.open(lr_path).convert('RGB')
            hr_arr = np.array(hr)

            # WaveMixSR-V2 (ONNX)
            inp = preprocess_onnx(lr).astype(np.float32)
            out = wm_sess.run(None, {'input': inp})[0]
            sr_arr = postprocess_onnx(out)
            p, s = compute_metrics(hr_arr, sr_arr)
            writer.writerow(['WaveMixSR-V2', name, f'{p:.3f}', f'{s:.4f}'])

            # EDSR
            t = preprocess_torch(lr)
            with torch.no_grad():
                out = edsr(t)
            sr_arr = postprocess_torch(out)
            p, s = compute_metrics(hr_arr, sr_arr)
            writer.writerow(['EDSR', name, f'{p:.3f}', f'{s:.4f}'])

            # RCAN
            with torch.no_grad():
                out = rcan(t)
            sr_arr = postprocess_torch(out)
            p, s = compute_metrics(hr_arr, sr_arr)
            writer.writerow(['RCAN', name, f'{p:.3f}', f'{s:.4f}'])

            # RRDBNet
            with torch.no_grad():
                out = rrdb(t)
            sr_arr = postprocess_torch(out)
            p, s = compute_metrics(hr_arr, sr_arr)
            writer.writerow(['RRDBNet', name, f'{p:.3f}', f'{s:.4f}'])

            print(f"✓ {name} done.")

    print(f"\n✅ All results saved to {output_csv}")

if __name__ == '__main__':
    main()
