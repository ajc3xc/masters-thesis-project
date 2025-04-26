from models.WaveMixSR.WaveMixSRV2 import SR_Block
import torch
import torch.nn as nn
import torch.nn.functional as F


class SafeInterp(nn.Module):
    def __init__(self, mode="bicubic"):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        h, w = x.shape[-2:]
        return F.interpolate(x, size=(h * 2, w * 2), mode=self.mode, align_corners=False)


class WaveMixPatchWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === FIX 1: Correct depth
model = SR_Block(depth=2, mult=1, final_dim=144, ff_channel=144, dropout=0.3).to(device)

# === FIX 2: Patch BOTH upsample layers
if hasattr(model, "path1") and isinstance(model.path1[0], nn.Upsample):
    model.path1[0] = SafeInterp(mode="bicubic")
if hasattr(model, "path2") and isinstance(model.path2, nn.Upsample):
    model.path2 = SafeInterp(mode="bicubic")

# Load weights
state_dict = torch.load("models/WaveMixSR/saved_model_weights/bsd100_2x_y_df2k_33.2.pth", map_location=device)
model.load_state_dict(state_dict, strict=False)
model.eval()

# Wrap
wrapped_model = WaveMixPatchWrapper(model).to(device)

# Dummy flexible input
dummy_input = torch.randn(1, 3, 256, 320).to(device)

# Export ONNX
torch.onnx.export(
    wrapped_model,
    dummy_input,
    "models/wavemixsrv2_srblock_2x_fullflex.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=15,
    dynamic_axes={
        "input": {0: "batch", 2: "height", 3: "width"},
        "output": {0: "batch", 2: "height", 3: "width"},
    }
)

print("✅ Exported FINAL corrected WaveMixSR 2x ONNX model with full flexibility!")