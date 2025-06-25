#!/usr/bin/env python3
import sys, os, glob, time, csv
import cv2
import torch
import numpy as np

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
# point these to your crackseg9k xception outputs:
IMG_DIR    = "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/data/crack_segmentation_unzipped/crack_segmentation/METU/rgb"        # e.g. …/dataset_data/crackseg9k/rgb
MASK_DIR   = "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/data/crack_segmentation_unzipped/crack_segmentation/METU/BW"         # e.g. …/dataset_data/crackseg9k/BW  (png/jpg)
WEIGHTS    = "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/pytorch-deeplab-xception/run/crackseg9k/crackseg9k_xception/experiment_10/checkpoint.pth.tar"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MAX_IMGS   = 100
OUTPUT_CSV = "metu_deeplabv3_xception_eval.csv"
# ────────────────────────────────────────────────────────────────────────────────

# add your project root so `modeling.deeplab` can be imported
#SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
#REPO_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))  # adjust if your code lives elsewhere
#sys.path.insert(0, REPO_ROOT)

from eval import eval_from_memory    # your existing eval script

def collect_deeplabv3_xception():
    print(f"Using device: {DEVICE}")
    device = torch.device(DEVICE)

    # ─── load the exact same DeepLab+Xception you train with ───────────────────
    from modeling.deeplab import DeepLab
    model = DeepLab(
        num_classes=2,
        backbone='xception',
        output_stride=16,
        sync_bn=False,
        freeze_bn=False
    )
    ckpt = torch.load(WEIGHTS, map_location=device)
    # saver saved a dict with 'state_dict'
    state_dict = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    # ─────────────────────────────────────────────────────────────────────────

    # gather image+mask paths
    imgs = sorted(glob.glob(os.path.join(IMG_DIR, '*.jpg')) +
                  glob.glob(os.path.join(IMG_DIR, '*.JPG')))
    imgs = imgs[:MAX_IMGS]
    masks = []
    for p in imgs:
        base = os.path.splitext(os.path.basename(p))[0]
        # support .png, .jpg, .JPG masks
        found = None
        for ext in ('.png','.jpg','.JPG'):
            m = os.path.join(MASK_DIR, base + ext)
            if os.path.exists(m):
                found = m
                break
        if found is None:
            raise FileNotFoundError(f"No mask found for image {p}")
        masks.append(found)

    preds, gts, total_time = [], [], 0.0

    # warm-up
    dummy = torch.randn(1,3,512,512, device=device)
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy)

    # inference loop
    for i, (img_p, mask_p) in enumerate(zip(imgs, masks), 1):
        print(f"[{i}/{len(imgs)}] {os.path.basename(img_p)}")
        bgr = cv2.imread(img_p)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        # resize back to orig
        im = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_CUBIC)
        tensor = torch.from_numpy(im.transpose(2,0,1))\
                      .float().unsqueeze(0).to(device)

        if device.type=='cuda': torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = model(tensor)
            if isinstance(out, dict):
                out = out.get('out', out)
        if device.type=='cuda': torch.cuda.synchronize()
        t1 = time.time()
        total_time += (t1 - t0)

        out = out.squeeze(0)
        if out.ndim==2:
            prob = torch.sigmoid(out)
            mask_pred = (prob>0.5).cpu().numpy().astype('uint8')*255
        else:
            mask_pred = out.argmax(0).cpu().numpy().astype('uint8')*255

        # debug first
        if i==1:
            od = "debug_outputs"
            os.makedirs(od, exist_ok=True)
            cv2.imwrite(os.path.join(od, "first_pred.png"), mask_pred)

        preds.append(mask_pred)
        gt = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        gt = ((gt>0).astype('uint8'))*255
        gts.append(gt)

    fps = len(preds)/total_time
    return preds, gts, fps

if __name__=='__main__':
    preds, gts, fps = collect_deeplabv3_xception()
    metrics = eval_from_memory(preds, gts)
    metrics.update({'FPS': fps})
    metrics.update({'Iters': MAX_IMGS})

    with open(OUTPUT_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(metrics.keys())
        w.writerow(metrics.values())

    print(f"✅ Saved results to {OUTPUT_CSV}")
