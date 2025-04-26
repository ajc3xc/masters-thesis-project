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
        # 2x upscaling with flexible size
        return F.interpolate(x, size=(h * 2, w * 2), mode=self.mode, align_corners=False)


class WaveMixPatchWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Accept any [B, 3, H, W]
        return self.model(x)


# Load base model and patch the interpolate layer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SR_Block(depth=4, mult=1, final_dim=144, ff_channel=144, dropout=0.3).to(device)

# Fix: properly replace the first interp layer in path1
if hasattr(model, "path1") and isinstance(model.path1[0], nn.Upsample):
    model.path1[0] = SafeInterp(mode="bicubic")
else:
    raise ValueError("Expected model.path1[0] to be an Upsample layer.")

# Load weights
state_dict = torch.load("models/WaveMixSR/saved_model_weights/bsd100_2x_y_df2k_33.2.pth", map_location=device)
model.load_state_dict(state_dict, strict=False)  # loose loading just in case
model.eval()

# Wrap model
wrapped_model = WaveMixPatchWrapper(model).to(device)

# Dummy input (flexible size)
dummy_input = torch.randn(1, 3, 256, 320).to(device)  # <-- Now, random size (256x320)

# Export to ONNX
torch.onnx.export(
    wrapped_model,
    dummy_input,
    "models/sr_block_2x_flexible.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=15,
    dynamic_axes={
        "input": {0: "batch", 2: "height", 3: "width"},
        "output": {0: "batch", 2: "height", 3: "width"},
    }
)

print("✅ Exported ONNX model with flexible width and height!")