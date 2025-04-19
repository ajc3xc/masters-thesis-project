import os
import math
import numpy as np
from PIL import Image
import torch
import onnxruntime as ort
from super_image import EdsrModel  # from eugenesiow/edsr-base :contentReference[oaicite:1]{index=1}
from torchvision.transforms.functional import to_tensor, to_pil_image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# ─── CONFIG ───────────────────────────────────────────────────────────────
dataset_dir = '/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/superres_benchmarks/Set14/Set14'
scale = 2
wavemix_onnx = '/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/models/sr_block_2x_patch.onnx'
edsr_name = 'eugenesiow/edsr-base'

hr_dir = os.path.join(dataset_dir, 'original')
lr_dir = os.path.join(dataset_dir, f'LRbicx{scale}')

# ─── LOAD MODELS ─────────────────────────────────────────────────────────
# WaveMixSR ONNX
os.environ['ORT_DISABLE_ARENA'] = '1'
wav_session = ort.InferenceSession(wavemix_onnx,
                                   providers=['CUDAExecutionProvider','CPUExecutionProvider'])

# EDSR via super_image (CPU/GPU under the hood) :contentReference[oaicite:2]{index=2}
edsr = EdsrModel.from_pretrained(edsr_name, scale=scale)
onnx
# ─── HELPERS ─────────────────────────────────────────────────────────────
def bicubic_sr(lr: Image.Image, hr_size):
    return lr.resize(hr_size, resample=Image.BICUBIC)

def onnx_sr(session, lr_patch: Image.Image):
    t = to_tensor(lr_patch).unsqueeze(0).numpy().astype(np.float32)
    out = session.run(None, {'input': t})[0]
    return to_pil_image(torch.from_numpy(np.clip(out[0],0,1)))

def edsr_sr(lr_patch: Image.Image):
    inp = lr_patch.convert('RGB')
    out = edsr([inp])[0]  # returns PIL.Image :contentReference[oaicite:3]{index=3}
    return out

def tile_and_run(lr_img, method_fn):
    w, h = lr_img.size
    pw = ph = 256  # patch size
    patches = []
    for y in range(0, h, ph):
        for x in range(0, w, pw):
            box = (x, y, min(x+pw, w), min(y+ph, h))
            patch = lr_img.crop(box)
            # pad to 256×256 if at border
            if patch.size != (pw,ph):
                patch = patch.resize((pw,ph), Image.BICUBIC)
            patches.append((box, method_fn(patch)))
    # stitch back
    out = Image.new('RGB', (w*scale, h*scale))
    for (x,y,x2,y2), p in patches:
        out.paste(p, (x*scale, y*scale))
    return out.crop((0,0, w*scale, h*scale))

# ─── EVALUATION ─────────────────────────────────────────────────────────
psnr_scores = {'bicubic':[], 'wavemix':[], 'edsr':[]}
ssim_scores = {'bicubic':[], 'wavemix':[], 'edsr':[]}

files = sorted(os.listdir(hr_dir))
print(f"Evaluating {len(files)} images on Set14…")
for fname in tqdm(files):
    hr = Image.open(os.path.join(hr_dir,fname)).convert('RGB')
    lr = Image.open(os.path.join(lr_dir,fname)).convert('RGB')

    # Run each method
    out_bic = bicubic_sr(lr, hr.size)
    out_wav = tile_and_run(lr, lambda p: onnx_sr(wav_session,p))
    out_eds = tile_and_run(lr, edsr_sr)

    hr_np, lb_np = np.array(hr), {}
    lb_np['bicubic'], lb_np['wavemix'], lb_np['edsr'] = map(np.array,(out_bic,out_wav,out_eds))

    for m in psnr_scores:
        psnr_scores[m].append(psnr(hr_np, lb_np[m], data_range=255))
        ssim_scores[m].append(ssim(hr_np, lb_np[m], channel_axis=2, data_range=255))

# ─── RESULTS ────────────────────────────────────────────────────────────
print("\nAverage PSNR / SSIM on Set14:")
for m in psnr_scores:
    print(f"{m:8s}: {np.mean(psnr_scores[m]):.2f} dB, {np.mean(ssim_scores[m]):.4f}")
