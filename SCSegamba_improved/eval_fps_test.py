import os
import time
import torch
import numpy as np
import cv2
import argparse
from datasets import create_dataset
from models import build_model
from main import get_args_parser
#from eval.evaluate import eval as evaluate_metrics
import csv

from eval.evaluate import cal_prf_metrics, cal_mIoU_metrics, cal_ODS_metrics, cal_OIS_metrics

def eval_from_memory(pred_list, gt_list):
    """
    In-memory version of `eval()` that evaluates model predictions without writing to disk.

    Args:
        pred_list (List[np.ndarray]): List of predicted masks (uint8, 0 or 255)
        gt_list   (List[np.ndarray]): List of ground truth masks (uint8, 0 or 255)

    Returns:
        dict: Dictionary containing mIoU, ODS, OIS, F1, Precision, and Recall
    """
    assert len(pred_list) == len(gt_list), "Mismatched prediction and ground truth counts"

    # Precision, Recall, F1 at 0.5 threshold
    final_accuracy_all = cal_prf_metrics(pred_list, gt_list)
    final_accuracy_all = np.array(final_accuracy_all)
    Precision_list = final_accuracy_all[:, 1]
    Recall_list = final_accuracy_all[:, 2]
    F1_list = final_accuracy_all[:, 3]

    # mIoU, ODS, OIS
    mIoU = cal_mIoU_metrics(pred_list, gt_list)
    ODS = cal_ODS_metrics(pred_list, gt_list)
    OIS = cal_OIS_metrics(pred_list, gt_list)

    return {
        "mIoU": mIoU,
        "ODS": ODS,
        "OIS": OIS,
        "F1": F1_list[0],
        "Precision": Precision_list[0],
        "Recall": Recall_list[0]
    }

# -------- CONFIGURATION --------
CHECKPOINTS = [
    "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/SCSegamba/checkpoint_TUT.pth",
    "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/SCSegamba_improved/checkpoints/weights/eca/2025_04_27_11:33:22_Dataset->Crack_Conglomerate/checkpoint_best.pth",
    "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/SCSegamba_improved/checkpoints/weights/sebica/2025_04_24_17:38:27_Dataset->Crack_Conglomerate/checkpoint_best.pth",
    "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/SCSegamba_improved/checkpoints/weights/gbc_eca/2025_04_24_04:52:32_Dataset->Crack_Conglomerate/checkpoint_best.pth",
    "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/SCSegamba_improved/checkpoints/weights/sfa/2025_04_24_07:02:45_Dataset->Crack_Conglomerate/checkpoint_best.pth",
]
DATASET_PATH = "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/SCSegamba/data/TUT"
TEMP_RESULTS_ROOT = "./temp_eval_results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

CSV_PATH = "eval_results.csv"

# -------- BASE SETUP --------
parser = argparse.ArgumentParser('SCSEGAMBA Eval', parents=[get_args_parser()])
args = parser.parse_args([])
args.phase = 'test'
args.dataset_path = DATASET_PATH
args.batch_size = 1
args.device = DEVICE
args.output_dir = "./testing"
args.input_size = 512
torch.backends.cudnn.benchmark = True

# -------- EVALUATION LOOP --------
os.makedirs(TEMP_RESULTS_ROOT, exist_ok=True)
#DEVICE = torch.DEVICE(DEVICE)
args.num_threads = 6
test_dl = create_dataset(args)

for ckpt_path in CHECKPOINTS:
    # Infer attention_type from the path
    ckpt_lower = ckpt_path.lower()
    if "sebica" in ckpt_lower:
        args.attention_type = "sebica"
    elif "eca" in ckpt_lower and "gbc" not in ckpt_lower:
        args.attention_type = "eca"
    elif "gbc_eca" in ckpt_lower:
        args.attention_type = "gbc_eca"
    elif "sfa" in ckpt_lower:
        args.attention_type = "sfa"
    else:
        args.attention_type = "gbc"  # default fallback
    print(f"\n🔍 Evaluating {ckpt_path} using attention: {args.attention_type}")


    # Load model
    model, _ = build_model(args)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE)
    model.eval()
    model = torch.compile(model, mode="reduce-overhead")  # <-- Add thi

    model_name = os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))
    print(f"Evaluating model: {model_name}")

    total_time = 0.0
    os.makedirs(TEMP_RESULTS_ROOT, exist_ok=True)

    pred_list = []
    gt_list = []
    total_time = 0.0

    torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        for data in test_dl:
            x = data["image"].to(DEVICE, non_blocking=True)
            out = model(x)

    torch.cuda.synchronize()
    total_time = time.time() - start
    fps = len(test_dl) / total_time
    print(f"Overall FPS: {fps:.2f}")

    with torch.no_grad():
        for data in test_dl:
            #print(i)
            x = data["image"].to(DEVICE, non_blocking=True)
            #print(x.shape)
            target = data["label"].to(dtype=torch.int64).to(DEVICE, non_blocking=True)

            # Inference-only timing
            #torch.cuda.synchronize()
            start = time.time()
            out = model(x)
            #torch.cuda.synchronize()
            total_time += time.time() - start
            print(total_time)

            # Convert predictions and ground truth to binary uint8 masks
            pred = out[0, 0].cpu().numpy()
            gt = target[0, 0].cpu().numpy()

            pred_img = (255 * (pred / pred.max())).astype(np.uint8) if pred.max() > 0 else np.zeros_like(pred, dtype=np.uint8)
            gt_img = (255 * (gt / gt.max())).astype(np.uint8) if gt.max() > 0 else np.zeros_like(gt, dtype=np.uint8)

            pred_list.append(pred_img)
            gt_list.append(gt_img)

    # FPS based on pure inference time
    fps = len(test_dl) / total_time

    # Evaluate metrics entirely in-memory
    print("evaluating metrics")
    metrics = eval_from_memory(pred_list, gt_list)
    metrics["FPS"] = fps
    print(f"🧪 Model: {model_name}")
    print(f"  mIoU : {metrics['mIoU']:.4f}")
    print(f"  ODS  : {metrics['ODS']:.4f}")
    print(f"  OIS  : {metrics['OIS']:.4f}")
    print(f"  F1   : {metrics['F1']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall   : {metrics['Recall']:.4f}")
    print(f"  FPS  : {fps:.2f}\n")

    with open(CSV_PATH, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)

    # Clean up
    for f in os.listdir(TEMP_RESULTS_ROOT):
        os.remove(os.path.join(TEMP_RESULTS_ROOT, f))
