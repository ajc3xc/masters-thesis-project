#!/usr/bin/env python3
"""
Optimized pipeline for SCSEGAMBA with TorchScript (compiling to speed up inference)
"""

import os, time
from pathlib import Path
from typing import List
import cv2, numpy as np, torch
from PIL import Image
from torchvision import transforms
from main import get_args_parser
from models import build_model
import onnxruntime as ort

# ───────────────────────── Globals ─────────────────────────
ROOT  = Path("pipeline_trt")
SRD   = ROOT / "superres"
MSKD  = ROOT / "masks"
OVD   = ROOT / "overlays"
CSVD  = ROOT / "csv"
for d in (SRD, MSKD, OVD, CSVD): d.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORM = transforms.ToTensor()

# ───────────────────── SCSEGAMBA Optimization with TorchScript ─────────────────
def build_scseg_model(ckpt: str, h: int, w: int) -> torch.nn.Module:
    """ 
    Build the SCSEGAMBA model and optimize it for faster inference using TorchScript.
    """
    # Set up args similar to your evaluation script
    parser = get_args_parser()
    args = parser.parse_args([])
    args.load_width, args.load_height = w, h
    ck_low = ckpt.lower()

    # Set fusion and attention type
    args.fusion_mode = "original" if "original/" in ck_low or "checkpoint_tut" in ck_low else "dynamic"
    if "sebica" in ck_low: args.attention_type = "sebica"
    elif "gbc_eca" in ck_low: args.attention_type = "gbc_eca"
    elif "eca" in ck_low: args.attention_type = "eca"
    elif "sfa" in ck_low: args.attention_type = "sfa"
    else: args.attention_type = "gbc"

    # Build model and load checkpoint
    model, _ = build_model(args)
    state = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(state["model"], strict=False)
    model.to(DEVICE)
    model.eval()

    return model

    # Optimize the model using TorchScript (Tracing or Scripting)
    print("[INFO] Optimizing model with TorchScript...")
    try:
        # Tracing: useful for models without control flow
        example_input = torch.randn(1, 3, h, w).to(DEVICE)
        scripted_model = torch.jit.trace(model, example_input)  # Trace model
        print("[INFO] Traced model successfully")
    except Exception as e:
        # If tracing doesn't work, use scripting
        print(f"[INFO] Tracing failed, falling back to scripting. Error: {e}")
        scripted_model = torch.jit.script(model)  # Script model

    return scripted_model

def run_scsegamba(imgs: List[Path], ckpt: str) -> List[Path]:
    """ Run SCSEGAMBA inference with optimized TorchScript model. """
    dummy = cv2.imread(str(imgs[0]))
    H, W = dummy.shape[:2]
    model = build_scseg_model(ckpt, H, W)

    masks = []
    for p in imgs:
        img = Image.open(p).convert("RGB")
        arr = TRANSFORM(img).unsqueeze(0).to(DEVICE)  # NCHW float32

        # Inference
        with torch.no_grad():
            prob = model(arr)  # Run model
            mask = (prob[0, 0] > 0.5).cpu().numpy().astype(np.uint8) * 255
            q = MSKD / f"{p.stem}_mask.png"
            cv2.imwrite(str(q), mask)
            masks.append(q)
            print(f"[SEG] {p.name} ✓")

    return masks

# ───────────────────── Super-resolution ────────────────────
def run_superresolution(onnx: str, imgs: List[Path]) -> List[Path]:
    """ Super-resolution function using ONNX. """
    sr_paths = []
    sess = ort.InferenceSession(
        onnx, providers=['CUDAExecutionProvider','CPUExecutionProvider']
    )

    for p in imgs:
        img = Image.open(p).convert("RGB")
        arr = (np.asarray(img).astype(np.float32)/255.).transpose(2, 0, 1)[None]
        out = sess.run(None, {'input': arr})[0][0]  # Run ONNX inference
        y = (np.clip(out, 0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)
        q = SRD / p.name
        Image.fromarray(y).save(q)
        sr_paths.append(q)
        print(f"[SR ] {p.name}  →  {y.shape[0]}×{y.shape[1]}")
    return sr_paths

# ───────────────────── Adaptive-PCA (unchanged) ────────────────
from qc_adaptive_pca_test import (mask_quality_check, adaptive_crack_width,
                                  draw_overlay, save_to_csv)

def run_adaptive_pca(masks: List[Path]):
    """ Measure crack widths using Adaptive-PCA """
    for mpath in masks:
        mask = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
        if mask.max() > 1:
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        if not mask_quality_check(mask):
            print(f"[PCA] {mpath.name} rejected")
            continue
        res = adaptive_crack_width(mask)
        if not res:
            print(f"[PCA] {mpath.name} no widths")
            continue
        ov = draw_overlay(mask, res)
        ov_path = OVD / f"{mpath.stem}_ovl.png"
        csv_path = CSVD / f"{mpath.stem}.csv"
        cv2.imwrite(str(ov_path), ov)
        save_to_csv(res, str(csv_path))
        print(f"[PCA] {mpath.name}  mean={np.mean([w for *_ ,w in res]):.2f}px")

# ────────────────────────── MAIN ───────────────────────────
if __name__ == "__main__":
    TEST = [
        "/mnt/stor/gchen-lab/data/Adam/masters-thesis-project/data/crack_segmentation_unzipped/crack_segmentation/virginia_tech_concrete_crack_congolmeration/Conglomerate Concrete Crack Detection/Conglomerate Concrete Crack Detection/Test/images/CFD_001.jpg",
        "/mnt/stor/gchen-lab/data/Adam/masters-thesis-project/data/crack_segmentation_unzipped/crack_segmentation/virginia_tech_concrete_crack_congolmeration/Conglomerate Concrete Crack Detection/Conglomerate Concrete Crack Detection/Test/images/CFD_005.jpg"
        # add more JPEGs as needed
    ]
    TEST = [Path(p) for p in TEST]

    # (1) super-resolve 512² → 2048² using ONNX
    sr_imgs = run_superresolution("/mnt/stor/gchen-lab/data/Adam/masters-thesis-project/superes_comparison_current_final/dscf_dynamic.onnx", TEST)

    # (2) SCSEGAMBA (compile once ➜ TorchScript)
    CKPT = "/mnt/stor/gchen-lab/data/Adam/masters-thesis-project/SCSegamba_improved/checkpoints/weights/eca/2025_05_05_00:47:18_Dataset->TUT_dynamic/checkpoint_best.pth"
    mask_paths = run_scsegamba(TEST, CKPT)

    # (3) adaptive-PCA widths
    run_adaptive_pca(mask_paths)

    print("\n✅  All stages finished – see", ROOT.resolve())