#!/usr/bin/env python3
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils.utils import load_image
from utils.model_utils import get_model, build_model, load_model_checkpoint
from imageio.v2 import imread
import numpy as np

class PairListDataset(Dataset):
    """
    Dataset that loads (mask, skeleton) pairs from a .txt file.
    Each line in <list_file> should be:
        <relative_path_to_mask> <relative_path_to_skeleton>
    where paths are relative to `root_dir`.
    """
    def __init__(self, root_dir, list_file):
        self.root_dir = root_dir
        self.pairs = []
        with open(list_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                mask_rel, skel_rel = line.split()
                self.pairs.append((mask_rel, skel_rel))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        mask_rel, skel_rel = self.pairs[idx]
        mask_path = os.path.join(self.root_dir, mask_rel)
        skel_path = os.path.join(self.root_dir, skel_rel)

        #mask = load_image(mask_path)   # returns a FloatTensor of shape [1, 1, H, W]
        mask = imread(mask_path)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        skel = imread(skel_path)
        skel = torch.tensor(skel, dtype=torch.float32).unsqueeze(0)

        #print(np.unique(mask), np.unique(skel))

        return mask, skel
        #skel = load_image(skel_path)   # same

        #return mask, skel

def dice_loss(pred, target, eps=1e-6):
    """
    Computes Dice loss for binary maps. Assumes pred and target are [B,1,H,W], float in [0,1].
    """
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()

def main():
    # ───────────────────────────────────────────────────────────────────────────────
    # Hardcoded paths / settings for convenience:
    # (Adjust these paths before running)
    #config_path = r"D:\camerer_ml\adaptive_pca_3dreproject\Skelite\ft_cfg_cs9k.yml"
    config_path = r"D:\camerer_ml\adaptive_pca_3dreproject\Skelite\ft_cfg_concrete3k.yml"
    save_dir = "fine_tuned_checkpoints"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # ───────────────────────────────────────────────────────────────────────────────

    # 1) Load YAML config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 2) Create save directory if needed
    os.makedirs(save_dir, exist_ok=True)
    dataset_dir = os.path.join(save_dir, config["dataset_name"])
    os.makedirs(dataset_dir, exist_ok=True)

    # 3) Device setup (overrides YAML if needed)
    print(f"[INFO] Using device: {device}")

    # 4) Build model using Skelite utilities
    model_module = get_model(config["net_type"])
    model = build_model(model_module, config, device).to(device)

    # 5) If resume_from/pretrained is set in YAML, load weights
    if "resume_from" in config and config["resume_from"]:
        ckpt_path = config["resume_from"]
        if os.path.isfile(ckpt_path):
            print(f"[INFO] Loading pretrained weights from {ckpt_path}")
            model = load_model_checkpoint(model, ckpt_path, device)
        else:
            print(f"[WARNING] resume_from path not found: {ckpt_path}")

    # 6) Prepare datasets and dataloaders (hardcoded via YAML’s train_list / val_list)
    train_dataset = PairListDataset(
        root_dir=config["data_root"],
        list_file=config["train_list"]
    )
    val_dataset = PairListDataset(
        root_dir=config["data_root"],
        list_file=config["val_list"]
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["val_batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 2),
        pin_memory=True
    )

    # 7) Define loss functions
    bce_loss = nn.BCELoss()

    # 8) Optimizer (Adam) + optional scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["lr"]),
        betas=(float(config.get("beta1", 0.9)), float(config.get("beta2", 0.999))),
        weight_decay=float(config.get("weight_decay", 0))
    )
    # Uncomment below to use a StepLR scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer,
    #     step_size=int(config.get("lr_step", 20)),
    #     gamma=float(config.get("lr_gamma", 0.5))
    # )

    # 9) Training loop parameters
    max_epoch = int(config["max_epoch"])
    log_iter = int(config.get("log_iter", 10))
    val_every = int(config.get("val_epoch", 1))
    skel_iters = int(config["net"]["skel_num_iter"])

    for epoch in range(1, max_epoch + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, (mask, true_skel) in enumerate(train_loader, start=1):
            # mask, true_skel: [B,1, H, W]
            mask = mask.to(device)
            true_skel = true_skel.to(device)

            optimizer.zero_grad()
            # Forward: model returns (pred_skel, _) as in demo.py
            pred_skel, _ = model(mask, z=None, no_iter=skel_iters)
            pred_skel = torch.clamp(pred_skel, 0.0, 1.0)

            # Compute losses: BCE + Dice
            loss_bce = bce_loss(pred_skel, true_skel)
            loss_dice = dice_loss(pred_skel, true_skel)
            loss = loss_bce + loss_dice

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch_idx % log_iter == 0:
                avg = running_loss / log_iter
                print(
                    f"[Epoch {epoch:03d}/{max_epoch:03d}] "
                    f"[Batch {batch_idx:04d}/{len(train_loader):04d}] "
                    f"Loss: {avg:.6f}"
                )
                running_loss = 0.0

        # Optional: scheduler.step()
        # scheduler.step()

        # 10) Validation
        if (epoch % val_every) == 0 or epoch == max_epoch:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for mask, true_skel in val_loader:
                    mask = mask.to(device)
                    true_skel = true_skel.to(device)

                    pred_skel, _ = model(mask, z=None, no_iter=skel_iters)
                    pred_skel = torch.clamp(pred_skel, 0.0, 1.0)

                    loss_bce = bce_loss(pred_skel, true_skel)
                    loss_dice = dice_loss(pred_skel, true_skel)
                    val_loss += (loss_bce + loss_dice).item()

            avg_val = val_loss / len(val_loader)
            print(f"--- [Epoch {epoch:03d}] Validation Loss: {avg_val:.6f}")

        # 11) Save a checkpoint every epoch
        ckpt_name = f"skelite_epoch{epoch:03d}.pth"
        ckpt_path = os.path.join(dataset_dir, ckpt_name)
        torch.save({"net": model.state_dict()}, ckpt_path)
        print(f"[INFO] Saved checkpoint: {ckpt_path}")

    print("[TRAINING COMPLETE]")


if __name__ == "__main__":
    main()