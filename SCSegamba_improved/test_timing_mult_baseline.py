import os
import csv
import time
import torch
import numpy as np
import cv2
from skimage.filters import threshold_niblack

from datetime import datetime
from models import build_model
from datasets import create_dataset
from eval.evaluate import eval_from_memory
from main import get_args_parser

# ────────────── 1) Baseline segmentation function ──────────────
def segment_crack_fast(image, win_size=25, k=0.8, min_area=50):
    """
    Median+Niblack+CC filtering baseline (exact paper steps)  
    Returns 0/255 uint8 mask.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)

    th = threshold_niblack(blur, window_size=win_size, k=k)
    binary = (blur > th).astype(np.uint8)

    n_labels, labels = cv2.connectedComponents(binary)
    counts = np.bincount(labels.ravel())
    keep = np.isin(labels, np.where(counts >= min_area)[0])
    clean = (keep.astype(np.uint8) * 255)
    return clean

# ────────────── 2) Settings ──────────────
MAX_ITERS   = 300
RESULTS_DIR = "results_eval"
os.makedirs(RESULTS_DIR, exist_ok=True)
csv_base_name = "eval_results_dynamic"
csv_path = os.path.join(RESULTS_DIR, f"{csv_base_name}.csv")
counter = 1
while os.path.exists(csv_path):
    csv_path = os.path.join(RESULTS_DIR, f"{csv_base_name}_{counter}.csv")
    counter += 1

# ────────────── 3) Build test subset ──────────────
parser = get_args_parser()
args = parser.parse_args([])
args.phase        = 'test'
args.dataset_path = '/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/data/crack_segmentation_unzipped/crack_segmentation/virginia_tech_concrete_crack_congolmeration/Conglomerate Concrete Crack Detection/Conglomerate Concrete Crack Detection/Test'
args.batch_size   = 1
device = torch.device(args.device)

print('creating dataset')
test_dl = create_dataset(args)
print('dataset created')

test_iter   = iter(test_dl)
test_subset = []
#hotwiring doing all of them
MAX_ITERS = len(test_dl)
for i in range(MAX_ITERS):
    test_subset.append(next(test_iter))
    print(f"{i}.", end="", flush=True)
print(f"\niterated to limit of {MAX_ITERS}")

# ────────────── 4) Evaluate baseline ──────────────
print("\n🔍 Evaluating baseline segmentation…")
baseline_preds = []
baseline_gts   = []
baseline_time  = 0.0

for batch in test_subset:
    img_t   = batch["image"][0].cpu().numpy()         # (3,H,W)
    gt_t    = batch["label"][0,0].cpu().numpy()       # (H,W)
    # convert to uint8 BGR
    img_np  = (np.transpose(img_t, (1,2,0)) * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    t0 = time.time()
    pred_mask = segment_crack_fast(img_bgr)
    t1 = time.time()
    baseline_time += (t1 - t0)

    # normalize GT to 0/255
    if np.max(gt_t) > 0:
        gt_mask = (255 * (gt_t / np.max(gt_t))).astype(np.uint8)
    else:
        gt_mask = np.zeros_like(gt_t, dtype=np.uint8)

    baseline_preds.append(pred_mask)
    baseline_gts.append(gt_mask)

print("evaluating baseline metrics…")
baseline_metrics = eval_from_memory(baseline_preds, baseline_gts)
baseline_metrics["FPS"] = len(baseline_preds) / baseline_time

# write baseline row (with header if needed)
write_header = not os.path.exists(csv_path)
with open(csv_path, 'a', newline='') as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(["Model","Fusion","Attention","mIoU","ODS","OIS","F1","Precision","Recall","FPS"])
    writer.writerow([
        "baseline_fast",      # Model
        "N/A",                # Fusion
        "N/A",                # Attention
        f"{baseline_metrics['mIoU']:.4f}",
        f"{baseline_metrics['ODS']:.4f}",
        f"{baseline_metrics['OIS']:.4f}",
        f"{baseline_metrics['F1']:.4f}",
        f"{baseline_metrics['Precision']:.4f}",
        f"{baseline_metrics['Recall']:.4f}",
        f"{baseline_metrics['FPS']:.2f}",
    ])
print(f"✅ Baseline done → {csv_path}")

# ────────────── 5) Your SCSEGAMBA loop ──────────────
CHECKPOINTS = [
    "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/SCSegamba/checkpoint_TUT.pth",
    "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/SCSegamba_improved/checkpoints/weights/gbc/2025_05_05_00:43:43_Dataset->TUT_dynamic/checkpoint_best.pth",
    "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/SCSegamba_improved/checkpoints/weights/gbc/2025_05_05_01:39:47_Dataset->TUT_Crack_Conglomerate_original/checkpoint_best.pth",
    "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/SCSegamba_improved/checkpoints/weights/gbc/2025_05_05_04:16:41_Dataset->TUT_Crack_Conglomerate_dynamic/checkpoint_best.pth",
    "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/SCSegamba_improved/checkpoints/weights/gbc_eca/2025_05_05_00:47:18_Dataset->TUT_dynamic/checkpoint_best.pth",
    "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/SCSegamba_improved/checkpoints/weights/gbc_eca/2025_05_05_04:25:55_Dataset->TUT_Crack_Conglomerate_dynamic/checkpoint_best.pth",
    "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/SCSegamba_improved/checkpoints/weights/eca/2025_05_05_00:47:18_Dataset->TUT_dynamic/checkpoint_best.pth",
    "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/SCSegamba_improved/checkpoints/weights/eca/2025_05_05_04:31:15_Dataset->TUT_Crack_Conglomerate_dynamic/checkpoint_best.pth",
    "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/SCSegamba_improved/checkpoints/weights/sebica/2025_05_05_00:47:18_Dataset->TUT_dynamic/checkpoint_best.pth",
    "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/SCSegamba_improved/checkpoints/weights/sebica/2025_05_05_04:29:08_Dataset->TUT_Crack_Conglomerate_dynamic/checkpoint_best.pth",
    "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/SCSegamba_improved/checkpoints/weights/sfa/2025_05_05_00:47:18_Dataset->TUT_dynamic/checkpoint_best.pth",
    "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/SCSegamba_improved/checkpoints/weights/sfa/2025_05_05_04:31:12_Dataset->TUT_Crack_Conglomerate_dynamic/checkpoint_best.pth",
]

for ckpt_path in CHECKPOINTS:
    ckpt_lower = ckpt_path.lower()
    base_name  = os.path.splitext(os.path.basename(ckpt_path))[0]

    # infer fusion/attention…
    if "original/" in ckpt_lower or "checkpoint_tut.pth" in ckpt_lower:
        args.fusion_mode     = "original"
        args.attention_type  = None
    elif "dynamic/" in ckpt_lower:
        args.fusion_mode     = "dynamic"
    else:
        args.fusion_mode     = "weighted"

    if   "sebica"   in ckpt_lower: args.attention_type = "sebica"
    elif "gbc_eca" in ckpt_lower: args.attention_type = "gbc_eca"
    elif "eca"     in ckpt_lower: args.attention_type = "eca"
    elif "sfa"     in ckpt_lower: args.attention_type = "sfa"
    elif args.fusion_mode!="original":
        args.attention_type = "gbc"

    fusion_str = args.fusion_mode or "unknownfusion"
    attn_str   = args.attention_type or "noattn"
    model_name = f"{base_name}_F-{fusion_str}_A-{attn_str}"
    print(f"\n🔍 Evaluating {model_name}")

    # build & load
    model, _      = build_model(args)
    state_dict    = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(state_dict["model"], strict=False)
    model.to(device)
    torch.backends.cudnn.benchmark = True

    # warm‐up
    print("warming up GPU…")
    for _ in range(5):
        _ = model(torch.randn(1, 3, args.load_width, args.load_height).to(device))
    torch.cuda.synchronize()

    # inference + gather preds/gts
    preds, gts      = [], []
    total_infer_time=0.0
    with torch.no_grad():
        model.eval()
        for batch in test_subset:
            x      = batch["image"].to(device)
            target = batch["label"].to(device).long()

            torch.cuda.synchronize()
            t0 = time.time()
            out = model(x)
            torch.cuda.synchronize()
            t1 = time.time()
            total_infer_time += (t1 - t0)

            # extract & scale
            tgt_np = target[0,0].cpu().numpy()
            if np.max(tgt_np)>0:
                tgt_np = (255*(tgt_np/np.max(tgt_np))).astype(np.uint8)
            else:
                tgt_np = np.zeros_like(tgt_np, dtype=np.uint8)

            pred_np = out[0,0].cpu().numpy()
            if np.max(pred_np)>0:
                pred_np = (255*(pred_np/np.max(pred_np))).astype(np.uint8)
            else:
                pred_np = np.zeros_like(pred_np, dtype=np.uint8)

            preds.append(pred_np)
            gts.append(tgt_np)

    # compute metrics
    print("evaluating metrics…")
    m = eval_from_memory(preds, gts)
    m["FPS"] = len(preds) / total_infer_time

    # append to CSV
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            model_name,
            args.fusion_mode,
            args.attention_type or "N/A",
            f"{m['mIoU']:.4f}",
            f"{m['ODS']:.4f}",
            f"{m['OIS']:.4f}",
            f"{m['F1']:.4f}",
            f"{m['Precision']:.4f}",
            f"{m['Recall']:.4f}",
            f"{m['FPS']:.2f}",
        ])
    print(f"✅ Done: {model_name} → {csv_path}")
