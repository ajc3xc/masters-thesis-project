import multiprocessing as mp
try:
    mp.set_start_method('forkserver')
except RuntimeError:
    pass  # Already set

import os
import time
import glob
import random
import _codecs
import argparse
from argparse import Namespace
from pathlib import Path

# ─── Third-party libraries ────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import f1_score, jaccard_score
import torch
import torch.backends.cudnn as cudnn
from torch.amp import GradScaler, autocast
from torch.serialization import add_safe_globals
from tqdm import tqdm
import cv2
from mmengine.optim.scheduler.lr_scheduler import PolyLR

# ─── Local application imports ────────────────────────────────────────────
from datasets import create_dataset
from models import build_model
from engine import train_one_epoch
from eval.evaluate import eval
from util.logger import get_logger
add_safe_globals([argparse.Namespace, np.core.multiarray.scalar, np.dtype, _codecs.encode])

DATASET_LIST = [
    {
        "name": "Crack_Conglomerate",
        "train": "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/data/crack_segmentation_unzipped/crack_segmentation/virginia_tech_concrete_crack_congolmeration/Conglomerate Concrete Crack Detection/Conglomerate Concrete Crack Detection/Train",
        "test":  "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/data/crack_segmentation_unzipped/crack_segmentation/virginia_tech_concrete_crack_congolmeration/Conglomerate Concrete Crack Detection/Conglomerate Concrete Crack Detection/Test",
    }
]

def get_args_parser():
    parser = argparse.ArgumentParser('SCSEGAMBA FOR CRACK', add_help=False)
    parser.add_argument('--BCELoss_ratio', default=0.87, type=float)
    parser.add_argument('--DiceLoss_ratio', default=0.13, type=float)
    parser.add_argument('--Norm_Type', default='GN', type=str)
    parser.add_argument('--batch_size_train', default=8, type=int)
    parser.add_argument('--batch_size_test', default=8, type=int)
    parser.add_argument('--lr_scheduler', default='PolyLR', type=str)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--attention_type', default='gbc_eca', choices=['gbc_eca','eca','sfa','sebica'], help="MFS block attention module", type=str)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--lr_drop', default=30, type=int)
    parser.add_argument('--sgd', action='store_true')
    parser.add_argument('--output_dir', default='./checkpoints/weights', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--serial_batches', action='store_true')
    parser.add_argument('--num_threads', default=8, type=int)
    parser.add_argument('--input_size', default=512, type=int)
    return parser


def export_to_onnx(model, args, dataset_name):
    onnx_dir = Path("onnx_exports")
    onnx_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy_input = torch.randn(1, 1, 512, 512).to(args.device)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_dir / f"{dataset_name}_best.onnx",
        input_names=["input"],
        output_names=["output"],
        export_params=True,
        do_constant_folding=True,
        opset_version=15,
        dynamic_axes={'input': {2: 'height', 3: 'width'}, 'output': {2: 'height', 3: 'width'}}
    )
    print(f"[✓] Exported ONNX for {dataset_name}.")

def evaluate_segmentation(preds, labels):
    flat_preds = np.concatenate([p.flatten() for p in preds])
    flat_labels = np.concatenate([l.flatten() for l in labels])
    return {
        'F1': f1_score(flat_labels, flat_preds, average='binary'),
        'mIoU': jaccard_score(flat_labels, flat_preds, average='binary')
    }


def train_on_dataset(dataset_cfg, args):
    dataset_name = dataset_cfg['name']
    onnx_file = Path("onnx_exports") / f"{dataset_name}_best.onnx"
    if onnx_file.exists():
        print(f"[✓] Skipping {dataset_name} (ONNX already exists)")
        return

    # === Resume or create new run directory ===
    base_path = Path(args.output_dir) / args.attention_type
    results_base = Path("results") / args.attention_type

    existing = sorted(glob.glob(str(base_path / f"*Dataset->{dataset_name}")), reverse=True)
    if existing and getattr(args, 'resume', True):
        process_folder = Path(existing[0])
        cur_time = process_folder.name.split('_')[0]
        print(f"[✓] Resuming from: {process_folder}")
    else:
        cur_time = time.strftime('%Y_%m_%d_%H:%M:%S', time.localtime())
        process_folder = base_path / f"{cur_time}_Dataset->{dataset_name}"
        process_folder.mkdir(parents=True, exist_ok=True)

    results_root = results_base / f"{cur_time}_Dataset->{dataset_name}"
    results_root.mkdir(parents=True, exist_ok=True)

    checkpoint_file = process_folder / 'checkpoint_best.pth'

    # === Loggers ===
    log_train = get_logger(str(process_folder), 'train')
    log_test  = get_logger(str(process_folder), 'test')
    log_eval  = get_logger(str(process_folder), 'eval')
    log_train.info("Args: " + str(args))

    # === Reproducibility ===
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    # === Build model and optimizer ===
    model, criterion = build_model(args)
    model.to(device)
    scaler = GradScaler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = PolyLR(optimizer, eta_min=args.min_lr, begin=args.start_epoch, end=args.epochs)

    # === Resume from latest epoch if available ===
    def extract_epoch(path):
        return int(path.stem.split("checkpoint_epoch")[-1])

    checkpoints = sorted(
        process_folder.glob("checkpoint_epoch*.pth"),
        key=extract_epoch,
        reverse=True
    )

    if checkpoints:
        latest_ckpt = checkpoints[0]
        print(f"[✓] Resuming from: {latest_ckpt.name}")
        checkpoint = torch.load(latest_ckpt, weights_only=False)

        def remap_gbc_to_attention_keys(state_dict):
            new_state = {}
            for k, v in state_dict.items():
                if "MFS.GBC_C." in k:
                    new_key = k.replace("MFS.GBC_C", "MFS.attention")
                    new_state[new_key] = v
                else:
                    new_state[k] = v
            return new_state

        checkpoint['model'] = remap_gbc_to_attention_keys(checkpoint['model'])
        missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)
        print("⚠️ [Patched Load] Missing:", missing)
        print("⚠️ [Patched Load] Unexpected:", unexpected)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        args.start_epoch = checkpoint['epoch'] + 1
        best_mIoU = checkpoint.get('best_mIoU', 0)
    else:
        print("[!] No checkpoint found — starting from scratch")
        args.start_epoch = 0
        best_mIoU = 0

    # === Load training data ===
    args.phase = 'train'
    args.dataset_path = dataset_cfg['train']
    args.batch_size = args.batch_size_train
    train_loader = create_dataset(args)

    # === Training Loop ===
    for epoch in range(args.start_epoch, args.epochs):
        log_train.info(f"Epoch {epoch+1}/{args.epochs}")
        train_one_epoch(model, criterion, train_loader, optimizer, epoch, args, log_train)
        scheduler.step()

        # Save per-epoch checkpoint
        safe_checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'epoch': epoch,
            'best_mIoU': float(best_mIoU),
        }

        # Optional: only save args if you're confident it's simple
        if hasattr(args, '__dict__'):
            safe_checkpoint['args'] = {
                k: (float(v) if isinstance(v, np.generic) else v)
                for k, v in vars(args).items()
            }

        epoch_ckpt_path = process_folder / f'checkpoint_epoch{epoch}.pth'
        torch.save(safe_checkpoint, epoch_ckpt_path)


        # === Validation ===
        args.phase = 'test'
        args.dataset_path = dataset_cfg['test']
        args.batch_size = args.batch_size_test
        test_loader = create_dataset(args)
        model.eval()

        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                x = batch["image"].to(device, non_blocking=True)
                target = batch["label"].to(dtype=torch.int64, device=device)
                out = model(x)
                out_np = out[0, 0].cpu().numpy()
                target_np = target[0, 0].cpu().numpy()

                # Save only in the last epoch
                if epoch == args.epochs - 1:
                    out_img = (255 * (out_np / out_np.max())).astype(np.uint8)
                    target_img = (255 * (target_np / target_np.max())).astype(np.uint8) if target_np.max() > 0 else np.zeros_like(target_np, dtype=np.uint8)
                    
                    name = Path(batch["A_paths"][0]).stem
                    final_output_dir = results_root / "final_outputs"
                    final_output_dir.mkdir(parents=True, exist_ok=True)

                    cv2.imwrite(str(final_output_dir / f"{name}_pre.png"), out_img)
                    cv2.imwrite(str(final_output_dir / f"{name}_lab.png"), target_img)

                # Still needed for in-memory evaluation
                all_preds.append((out_np > 0.5).astype(np.uint8))
                all_labels.append((target_np > 0.5).astype(np.uint8))

        metrics = evaluate_segmentation(all_preds, all_labels)
        print(f"[{dataset_name}] Epoch {epoch}: mIoU = {metrics['mIoU']:.4f}, F1 = {metrics['F1']:.4f}")
        if metrics['mIoU'] > best_mIoU:
            best_mIoU = metrics['mIoU']
            #torch.save(model.state_dict(), checkpoint_file)
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'epoch': epoch,
                'args': args,
                'best_mIoU': best_mIoU
            }, checkpoint_file)
            log_train.info(f"New best model at epoch {epoch}: mIoU = {best_mIoU:.4f}")

    # === ONNX Export ===
    checkpoint = torch.load(checkpoint_file, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    #dummy_input = torch.randn(1, 3, args.input_size, args.input_size).to(device)
    dummy_input = torch.randn(1, 1, args.input_size, args.input_size).to(device)
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)  # (B,1,H,W) → (B,3,H,W)
            return self.model(x)

    model_to_export = ModelWrapper(model)

    torch.onnx.export(
        model_to_export,
        dummy_input,
        onnx_file,
        input_names=["input"],
        output_names=["output"],
        export_params=True,
        do_constant_folding=True,
        opset_version=16,
        dynamic_axes={'input': {2: 'height', 3: 'width'}, 'output': {2: 'height', 3: 'width'}}
    )
    print(f"[✓] Finished {dataset_name} — Best mIoU: {best_mIoU:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SCSEGAMBA FOR CRACK', parents=[get_args_parser()])
    parser.add_argument('--dataset_mode', default='crack', type=str,
                        help='Dataset mode selector (required by create_dataset)')
    args = parser.parse_args()
    for dataset_cfg in DATASET_LIST:
        train_on_dataset(dataset_cfg, args)