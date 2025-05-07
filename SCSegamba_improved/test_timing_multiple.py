import os
import csv
import time
import torch
import numpy as np
from datetime import datetime
from models import build_model
from datasets import create_dataset
from eval.evaluate import eval_from_memory
from main import get_args_parser

# 🔁 Your model checkpoint paths
'''CHECKPOINTS = [
    "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/SCSegamba/checkpoint_TUT.pth",
    "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/SCSegamba/checkpoints/weights/2025_04_23_11:38:23_Dataset->Crack_Conglomerate/checkpoint_best.pth",
    "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/SCSegamba_improved/checkpoints/weights/eca/2025_04_27_11:33:22_Dataset->Crack_Conglomerate/checkpoint_best.pth",
    "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/SCSegamba_improved/checkpoints/weights/gbc_eca/2025_04_24_04:52:32_Dataset->Crack_Conglomerate/checkpoint_best.pth",
    "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/SCSegamba_improved/checkpoints/weights/sebica/2025_04_24_17:38:27_Dataset->Crack_Conglomerate/checkpoint_best.pth",
    "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/SCSegamba_improved/checkpoints/weights/sfa/2025_04_24_07:02:45_Dataset->Crack_Conglomerate/checkpoint_best.pth",
]

CHECKPOINTS = [
    "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/SCSegamba/checkpoint_TUT.pth",
    "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/SCSegamba/checkpoints/TUT/checkpoint_best.pth",
    "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/SCSegamba/checkpoints/weights/2025_04_23_11:38:23_Dataset->Crack_Conglomerate/checkpoint_best.pth",
]'''

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

# 🔧 Settings
MAX_ITERS = 300
RESULTS_DIR = "results_eval"
os.makedirs(RESULTS_DIR, exist_ok=True)
csv_base_name = "eval_results_dynamic"
csv_path = os.path.join(RESULTS_DIR, f"{csv_base_name}.csv")
counter = 1
while os.path.exists(csv_path):
    csv_path = os.path.join(RESULTS_DIR, f"{csv_base_name}_{counter}.csv")
    counter += 1

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
#print(len(test_dl))
#import sys
#sys.exit()
for i in range(MAX_ITERS):
    batch = next(test_iter)
    test_subset.append(batch)
    print(f"{i}.", end="", flush=True)  # Print dot without newline
print()
print(f'iterated to limit to {MAX_ITERS}')

# 📝 Open CSV and write header
#with open(csv_path, mode='w', newline='') as f:
#    writer = csv.writer(f)
#    writer.writerow(["Model", "Fusion", "Attention", "mIoU", "ODS", "OIS", "F1", "Precision", "Recall", "FPS"])

# 🔁 Evaluate each checkpoint
for ckpt_path in CHECKPOINTS:
    ckpt_lower = ckpt_path.lower()
    base_name = os.path.splitext(os.path.basename(ckpt_path))[0]

    # Infer fusion + attention type
    if "original/" in ckpt_lower or "checkpoint_tut.pth" in ckpt_lower:
        args.fusion_mode = "original"
        args.attention_type = None
    elif "dynamic/" in ckpt_lower:
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

    fusion_str = args.fusion_mode or "unknownfusion"
    attn_str = args.attention_type or "noattn"
    model_name = f"{base_name}_F-{fusion_str}_A-{attn_str}"
    print(f"\n🔍 Evaluating {ckpt_lower} {model_name} | Fusion: {args.fusion_mode} | Attention: {args.attention_type}")
    #continue

    # Build + load model
    model, _ = build_model(args)
    #state_dict = torch.load(ckpt_path, map_location='cuda')
    #model.load_state_dict(state_dict["model"], strict=False)
    state_dict = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(state_dict["model"], strict=False)
    torch.backends.cudnn.benchmark = True
    model.to(device)

    print("warming up gpu")
    for _ in range(5):
        _ = model(torch.randn(1, 3, args.load_width, args.load_height).cuda())
    torch.cuda.synchronize()
    print("gpu warmed up")

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
            pred_list.append(out)
            gt_list.append(target)

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