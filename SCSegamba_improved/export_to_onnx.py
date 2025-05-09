#!/usr/bin/env python3
import torch
from pathlib import Path
from argparse import Namespace

# ─── Update these two paths ───────────────────────────────────────────────
CHECKPOINT_PATH = Path("/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/SCSegamba_improved/checkpoints/weights/eca/2025_05_05_00:47:18_Dataset->TUT_dynamic/checkpoint_best.pth")
OUTPUT_ONNX_PATH = Path("onnx_exports/TUT_eca_dynamic_1792.onnx")
# ──────────────────────────────────────────────────────────────────────────

# Make sure the output dir exists
OUTPUT_ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)

# ─── Build the model exactly as during training ────────────────────────────
# You only need those args that build_model actually uses:
args = Namespace(
    # (these defaults match your training script)
    BCELoss_ratio=0.87,
    DiceLoss_ratio=0.13,
    Norm_Type='GN',
    attention_type='eca',
    fusion_mode='dynamic',
    # unused but safe to include
    batch_size_train=1,
    batch_size_test=1,
    lr_scheduler='PolyLR',
    lr=5e-4,
    min_lr=1e-6,
    weight_decay=0.01,
    epochs=1,
    start_epoch=0,
    lr_drop=30,
    sgd=False,
    output_dir='.',
    device='cuda' if torch.cuda.is_available() else 'cpu',
    seed=42,
    serial_batches=False,
    num_threads=1,
    input_size=1792,
)
from models import build_model
model, _ = build_model(args)
model.eval()

# ─── Load & remap your checkpoint ─────────────────────────────────────────
ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
state_dict = ckpt.get('model', ckpt)

def remap_gbc_to_attention_keys(sd):
    new_sd = {}
    for k, v in sd.items():
        if "MFS.GBC_C." in k:
            new_key = k.replace("MFS.GBC_C", "MFS.attention")
            new_sd[new_key] = v
        else:
            new_sd[k] = v
    return new_sd

state_dict = remap_gbc_to_attention_keys(state_dict)
missing, unexpected = model.load_state_dict(state_dict, strict=False)
if missing or unexpected:
    print("⚠️  Warning: missing keys:", missing)
    print("⚠️  Warning: unexpected keys:", unexpected)

# ─── Wrap for single-channel expansion & ONNX export ────────────────────────
class ModelWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, x):
        # if grayscale, expand to 3-channel
        #if x.shape[1] == 1:
        #    B, C, H, W = x.shape
        #    x = x.expand(B, 3, H, W)
        B, C, H, W = x.shape
        x = x.expand(B, 3, H, W)
        return self.m(x)

wrapper = ModelWrapper(model).to(args.device)

# dummy input: channel=1 (grayscale), H and W can be *anything* at inference
dummy_input = torch.randn(1, 1, 1792, 1792, device=args.device)

with torch.no_grad():
    torch.onnx.export(
        wrapper,
        dummy_input,                     # shape [1,1,512,512]
        OUTPUT_ONNX_PATH,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {2: "height",  3: "width"},
            "output": {2: "height",  3: "width"},
        },
        opset_version=17,
        do_constant_folding=True,
        export_params=True,
    )

print(f"[✓] Exported ONNX to {OUTPUT_ONNX_PATH}")
