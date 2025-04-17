#!/usr/bin/env python3
"""
pipeline.py

End-to-end crack analysis:
  1. Super-resolve (WaveMixSR)
  2. Segment (CrackMamba)
  3. Refine (EfficientViT-SAM)
  4. Skeletonize + adaptive PCA crack-width measurement
"""

import os
import argparse
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import cv2
from skimage.morphology import skeletonize
from scipy.interpolate import interp1d

# 1. IMPORT MODELS
from wavemix import WaveMixSR
from crackmamba.model import CrackMambaModel
from efficientvit_sam import build_sam_vit_l, SamPredictor

# --- Utility functions ---

def load_models(device):
    # WaveMix-SR
    sr = WaveMixSR().to(device).eval()
    # CrackMamba
    seg = CrackMambaModel().to(device).eval()
    # EfficientViT-SAM (Level 0)
    sam_vit = build_sam_vit_l(pretrained=True).to(device).eval()
    predictor = SamPredictor(sam_vit)
    return sr, seg, predictor

def preprocess(img_path, size=512):
    img = Image.open(img_path).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return tf(img).unsqueeze(0)  # B×C×H×W

def super_resolve(lr, model, device):
    lr = lr.to(device)
    with torch.no_grad():
        hr = model(lr)
    return hr.clamp(-1,1)

def segment_crack(hr, model, device):
    with torch.no_grad():
        logits = model(hr.to(device))
        mask = torch.sigmoid(logits)
    return mask.squeeze(0).cpu().numpy()

def refine_sam(hr, mask, predictor, device):
    # hr: B×C×H×W; mask: H×W
    img = ((hr.squeeze(0).permute(1,2,0).cpu().numpy() * 0.5 + 0.5)*255).astype(np.uint8)
    predictor.set_image(img)
    # convert mask to boxes for SAM input
    h, w = mask.shape
    ys, xs = np.where(mask>0.5)
    if len(xs)==0:
        return mask, mask
    box = [xs.min(), ys.min(), xs.max(), ys.max()]
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=np.array(box)[None,:],
        multimask_output=False
    )
    refined = masks[0].astype(np.uint8)
    return refined, mask

def measure_widths(refined, conf, 
                   curvature_threshold=0.1, 
                   min_win=5, max_win=15, 
                   conf_thresh=0.5):
    skel = skeletonize(refined>0)
    coords = np.argwhere(skel)
    widths = []
    for y,x in coords:
        if conf[y,x] < conf_thresh: 
            continue
        # local window
        half = max_win//2
        y0, y1 = max(0,y-half), min(skel.shape[0], y+half+1)
        x0, x1 = max(0,x-half), min(skel.shape[1], x+half+1)
        region = np.argwhere(skel[y0:y1, x0:x1])
        if region.shape[0] < 2: 
            continue
        # curvature proxy = 0 for simplicity
        win = min_win if 0>curvature_threshold else max_win
        # PCA
        pts = region - region.mean(axis=0)
        _,_,V = np.linalg.svd(pts, full_matrices=False)
        tangent = V[0]
        normal = np.array([-tangent[1], tangent[0]])
        # sample along normal
        ds = np.linspace(-win, win, 2*win+1)
        hits = []
        for d in ds:
            yy = int(y + normal[0]*d)
            xx = int(x + normal[1]*d)
            if 0<=yy<refined.shape[0] and 0<=xx<refined.shape[1]:
                if refined[yy,xx]==0:
                    hits.append(d)
        if len(hits)>=2:
            widths.append(max(hits)-min(hits))
    return widths

# --- Main pipeline ---

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True,
                   help="folder of raw crack images")
    p.add_argument("--output-dir", required=True,
                   help="where to save masks & widths CSVs")
    p.add_argument("--device", default="cuda",
                   help="torch device")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    sr_model, seg_model, sam_pred = load_models(device)

    for fn in sorted(os.listdir(args.input_dir)):
        if not fn.lower().endswith((".png",".jpg",".jpeg")):
            continue
        fp = os.path.join(args.input_dir, fn)
        lr = preprocess(fp)
        hr = super_resolve(lr, sr_model, device)
        init_mask = segment_crack(hr, seg_model, device)
        refined, conf = refine_sam(hr, init_mask, sam_pred, device)
        widths = measure_widths(refined, conf)

        # save outputs
        base = os.path.splitext(fn)[0]
        np.save(os.path.join(args.output_dir, f"{base}_mask.npy"), refined)
        np.savetxt(os.path.join(args.output_dir, f"{base}_widths.csv"),
                   np.array(widths), delimiter=",")

        print(f"Processed {fn}: {len(widths)} width measurements")

if __name__ == "__main__":
    main()
