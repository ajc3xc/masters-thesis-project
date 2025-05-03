import os
import csv
import time
import torch
import numpy as np
from datetime import datetime
from models import build_model
from datasets import create_dataset
from eval.evaluate import cal_prf_metrics, cal_mIoU_metrics, cal_ODS_metrics, cal_OIS_metrics
from main import get_args_parser

# 🔁 Your model checkpoint paths
CHECKPOINTS = [
    "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/SCSegamba/checkpoint_TUT.pth",
    "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/SCSegamba/checkpoints/weights/2025_04_23_11:38:23_Dataset->Crack_Conglomerate/checkpoint_best.pth",
    #"/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/SCSegamba_improved/checkpoints/weights/gbc_eca/2025_04_24_04:52:32_Dataset->Crack_Conglomerate/checkpoint_best.pth",
]

# 🔧 Settings
MAX_ITERS = 50
RESULTS_DIR = "results_eval"
os.makedirs(RESULTS_DIR, exist_ok=True)
csv_base_name = "eval_results"
csv_path = os.path.join(RESULTS_DIR, f"{csv_base_name}.csv")
counter = 1
while os.path.exists(csv_path):
    csv_path = os.path.join(RESULTS_DIR, f"{csv_base_name}_{counter}.csv")
    counter += 1

# 🧠 Utility: evaluate in-memory
def eval_from_memory(pred_list, gt_list):
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

# 🌱 Initialize args
parser = get_args_parser()
args = parser.parse_args([])
args.phase = 'test'
args.dataset_path = '/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/data/crack_segmentation_unzipped/crack_segmentation/virginia_tech_concrete_crack_congolmeration/Conglomerate Concrete Crack Detection/Conglomerate Concrete Crack Detection/Test'
args.dataset_path = '/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/data/crack_segmentation_unzipped/crack_segmentation/virginia_tech_concrete_crack_congolmeration/Conglomerate Concrete Crack Detection/Conglomerate Concrete Crack Detection/Test'
#args.serial_batches = True
args.batch_size = 1
#args.num_threads = 12
device = torch.device(args.device)

# 💽 Create consistent DataLoader
print('creating dataset')
test_dl = create_dataset(args)
print('dataset created')
test_iter = iter(test_dl)
test_subset = []
for i in range(MAX_ITERS):
    batch = next(test_iter)
    test_subset.append(batch)
    print(i, end="", flush=True)  # Print dot without newline
print()
print(f'iterated to limit to {MAX_ITERS}')

# 📝 Open CSV and write header
#with open(csv_path, mode='w', newline='') as f:
#    writer = csv.writer(f)
#    writer.writerow(["Model", "Fusion", "Attention", "mIoU", "ODS", "OIS", "F1", "Precision", "Recall", "FPS"])

# 🔁 Evaluate each checkpoint
for ckpt_path in CHECKPOINTS:
    ckpt_lower = ckpt_path.lower()
    model_name = os.path.splitext(os.path.basename(ckpt_path))[0]

    # Infer fusion + attention type
    if "scsegamba/" in ckpt_lower:
        args.fusion_mode = "original"
        args.attention_type = None
    elif "dynamic" in ckpt_lower:
        args.fusion_mode = "dynamic"
    else:
        args.fusion_mode = "weighted"

    if "sebica" in ckpt_lower:
        args.attention_type = "sebica"
    elif "gbc_eca" in ckpt_lower:
        args.attention_type = "gbc_eca"
    elif "eca" in ckpt_lower:
        args.attention_type = "eca"
    elif "sfa" in ckpt_lower:
        args.attention_type = "sfa"
    elif args.fusion_mode != "original":
        args.attention_type = "gbc"

    print(f"\n🔍 Evaluating {model_name} | Fusion: {args.fusion_mode} | Attention: {args.attention_type}")
    #continue

    # Build + load model
    model, _ = build_model(args)
    #state_dict = torch.load(ckpt_path, map_location='cuda')
    #model.load_state_dict(state_dict["model"], strict=False)
    state_dict = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(state_dict["model"], strict=False)
    torch.backends.cudnn.benchmark = True
    model.to(device)

    pred_list, gt_list = [], []
    total_infer_time = 0.0

    with torch.no_grad():
        model.eval()
        for data in test_subset:
            #print(".")
            x = data["image"].cuda(non_blocking=True)
            target = data["label"].cuda(non_blocking=True).long()

            torch.cuda.synchronize()
            start = time.time()
            out = model(x)
            torch.cuda.synchronize()
            end = time.time()
            total_infer_time += end - start

            target = target[0, 0, ...].cpu().numpy()
            out = out[0, 0, ...].cpu().numpy()
            #target = 255 * (target / np.max(target))
            #out = 255 * (out / np.max(out))
            if np.max(target) > 0:
                target = (255 * (target / np.max(target))).astype(np.uint8)
            else:
                target = np.zeros_like(target, dtype=np.uint8)

            if np.max(out) > 0:
                out = (255 * (out / np.max(out))).astype(np.uint8)
            else:
                out = np.zeros_like(out, dtype=np.uint8)
            pred_list.append(target)
            gt_list.append(out)

    print("evaluating metrics")
    metrics = eval_from_memory(pred_list, gt_list)
    metrics["FPS"] = len(test_subset) / total_infer_time

    # Write result row
    # Check if file exists to decide whether to write header
    write_header = not os.path.exists(csv_path)
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Model", "Fusion", "Attention", "mIoU", "ODS", "OIS", "F1", "Precision", "Recall", "FPS"])
        writer.writerow([
            model_name,
            args.fusion_mode,
            args.attention_type or "N/A",
            f"{metrics['mIoU']:.4f}",
            f"{metrics['ODS']:.4f}",
            f"{metrics['OIS']:.4f}",
            f"{metrics['F1']:.4f}",
            f"{metrics['Precision']:.4f}",
            f"{metrics['Recall']:.4f}",
            f"{metrics['FPS']:.2f}"
        ])
    print(f"✅ Done: {model_name} → {csv_path}")