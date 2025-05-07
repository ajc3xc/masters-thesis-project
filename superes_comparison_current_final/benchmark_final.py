#!/usr/bin/env python3
import os
# ─── 1) LOCAL cuDNN INJECTION ─────────────────────────────────────────────
CUDNN_DIR = "/mnt/stor/ceph/gchen-lab/data/Adam/cudnn"
os.environ["LD_LIBRARY_PATH"] = f"{CUDNN_DIR}/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["CPATH"]         = f"{CUDNN_DIR}/include:" + os.environ.get("CPATH", "")
os.environ["LIBRARY_PATH"]  = f"{CUDNN_DIR}/lib:" + os.environ.get("LIBRARY_PATH", "")

#spit into terminal if the top fails
'''
export CUDNN_DIR=/mnt/stor/ceph/gchen-lab/data/Adam/cudnn
export LD_LIBRARY_PATH=$CUDNN_DIR/lib:$LD_LIBRARY_PATH
export CPATH=$CUDNN_DIR/include:$CPATH
export LIBRARY_PATH=$CUDNN_DIR/lib:$LIBRARY_PATH
'''

# ─── IMPORTS ────────────────────────────────────────────────────────────────
import time, csv
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torchvision.transforms.functional import to_tensor, to_pil_image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity     as compare_ssim
import onnxruntime as ort

# ─── CONFIGURATION ─────────────────────────────────────────────────────────
DATA_DIR        = '/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/superres_benchmarks/DIV2K'
HR_DIR          = os.path.join(DATA_DIR, 'DIV2K_valid_HR')
LR_DIR          = os.path.join(DATA_DIR, 'DIV2K_valid_LR_bicubic', 'X4')
SCALE           = 4
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── LOAD torchSR MODELS ────────────────────────────────────────────────────
print("🧠 Loading torchSR models…")
from torchsr.models import edsr, rcan, ninasr_b1, carn
edsr_m   = edsr(scale=SCALE,  pretrained=True).to(DEVICE).eval()
rcan_m   = rcan(scale=SCALE,  pretrained=True).to(DEVICE).eval()
ninasr_m = ninasr_b1(scale=SCALE, pretrained=True).to(DEVICE).eval()
carn_m   = carn(scale=SCALE,  pretrained=True).to(DEVICE).eval()

# ─── LOAD ONNX SESSIONS ────────────────────────────────────────────────────
# Paths for your PyTorch and ONNX models
DSCF_PTH        = '/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/superes_comparison_current_final/sr_models/DSCF-SR/model_zoo/team23_DSCF.pth'
DSCF_PATH = "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/superes_comparison_current_final/sr_models/DSCF_SR"
DSCF_ONNX       = os.path.join(DSCF_PATH, 'dscf_dynamic.onnx')
REAL_ESRGAN_ONNX= os.path.join(DSCF_PATH, 'realesrgan_x4.onnx')
print("🧠 Initializing ONNXRuntime sessions…")
dscf_sess   = ort.InferenceSession(DSCF_ONNX,        providers=['CUDAExecutionProvider'])
esrgan_sess = ort.InferenceSession(REAL_ESRGAN_ONNX,       providers=['CUDAExecutionProvider'])

# ─── INFERENCE WRAPPERS ────────────────────────────────────────────────────
def run_edsr(img):
    t = to_tensor(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad(): out = edsr_m(t).clamp(0,1)
    return to_pil_image(out.squeeze(0).cpu())

def run_rcan(img):
    t = to_tensor(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad(): out = rcan_m(t).clamp(0,1)
    return to_pil_image(out.squeeze(0).cpu())

def run_ninasr(img):
    t = to_tensor(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad(): out = ninasr_m(t).clamp(0,1)
    return to_pil_image(out.squeeze(0).cpu())

def run_carn(img):
    t = to_tensor(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad(): out = carn_m(t).clamp(0,1)
    return to_pil_image(out.squeeze(0).cpu())

def run_bicubic(img):
    return img.resize((img.width*SCALE, img.height*SCALE), Image.BICUBIC)

def run_lanczos(img):
    return img.resize((img.width*SCALE, img.height*SCALE), Image.LANCZOS)

def run_onnx(sess, img):
    x = to_tensor(img).unsqueeze(0).cpu().numpy().astype(np.float32)
    start = time.time()
    out, = sess.run(None, {'input': x})
    tm = (time.time() - start)*1000
    out = np.clip(out, 0,1)
    pil = to_pil_image(torch.from_numpy(out).squeeze(0))
    return pil, tm

# ─── METHOD DICTIONARY ─────────────────────────────────────────────────────
methods = {
    "EDSR":             run_edsr,
    "RCAN":             run_rcan,
    "NinaSR":           run_ninasr,
    "CARN":             run_carn,
    "Bicubic":          run_bicubic,
    "Lanczos":          run_lanczos,
    "DSCF_ONNX":        lambda im: run_onnx(dscf_sess, im),
    "RealESRGAN_ONNX":  lambda im: run_onnx(esrgan_sess, im),
}

# ─── BENCHMARK LOOP ─────────────────────────────────────────────────────────
results   = {m:{"psnr":[], "ssim":[], "time_ms":[]} for m in methods}
files     = sorted(os.listdir(HR_DIR))
print(f"📊 Benchmarking {len(files)} DIV2K×{SCALE}…")

for fn in tqdm(files):
    hr_path = os.path.join(HR_DIR, fn)
    lr_name = os.path.splitext(fn)[0] + f"x{SCALE}.png"
    lr_path = os.path.join(LR_DIR, lr_name)
    if not os.path.exists(lr_path):
        print(f"⚠️ Missing {lr_name}, skipping.")
        continue

    hr = Image.open(hr_path).convert('RGB')
    lr = Image.open(lr_path).convert('RGB')
    hr_np = np.array(hr)

    for name, fnc in methods.items():
        start = time.time()
        out = fnc(lr)
        if isinstance(out, tuple):
            sr, tm = out
        else:
            sr, tm = out, (time.time()-start)*1000

        if sr.size != hr.size:
            sr = sr.resize(hr.size, Image.BICUBIC)
        sr_np = np.array(sr)

        results[name]["psnr"].append(compare_psnr(hr_np, sr_np, data_range=255))
        results[name]["ssim"].append(compare_ssim(hr_np, sr_np, channel_axis=2, data_range=255))
        results[name]["time_ms"].append(tm)

# ─── SAVE & PRINT ──────────────────────────────────────────────────────────
csv_out = "full_benchmark_results.csv"
with open(csv_out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Method","Avg_PSNR","Avg_SSIM","Avg_Time_ms","FPS"])
    for name, sc in results.items():
        a_psnr = np.mean(sc["psnr"])
        a_ssim = np.mean(sc["ssim"])
        a_tm   = np.mean(sc["time_ms"])
        fps    = 1000.0/a_tm if a_tm>0 else 0
        w.writerow([name, f"{a_psnr:.4f}", f"{a_ssim:.4f}", f"{a_tm:.2f}", f"{fps:.2f}"])

print("\n✅ Summary:")
for name, sc in results.items():
    a_psnr = np.mean(sc["psnr"])
    a_ssim = np.mean(sc["ssim"])
    a_tm   = np.mean(sc["time_ms"])
    fps    = 1000.0/a_tm if a_tm>0 else 0
    print(f"🔹 {name:16s} | PSNR={a_psnr:.2f} dB | SSIM={a_ssim:.4f} | Time={a_tm:.2f} ms | FPS={fps:.2f}")

print(f"\n➡️ Results saved to ./{csv_out}")