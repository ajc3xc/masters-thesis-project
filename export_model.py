import torch
import torch.nn as nn
import torch.nn.functional as F
from WaveMixSR.WaveMixSRV2 import SR_Block


class SafeInterp(nn.Module):
    def __init__(self, mode="bicubic", scale_factor=None):
        super().__init__()
        self.mode = mode
        self.scale_factor = scale_factor

    def forward(self, x):
        # Expect 4D: [B, C, H, W]
        return F.interpolate(x, mode=self.mode, scale_factor=self.scale_factor, align_corners=False)


class WaveMixPatchWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # x: [B, 3, 252, 252]
        return self.model(x)


# Load base model and patch the interpolate layer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SR_Block(depth=4, mult=1, final_dim=144, ff_channel=144, dropout=0.3).to(device)
model.path1[0] = SafeInterp(mode="bicubic", scale_factor=2)

# Load weights
state_dict = torch.load("models/bsd100_2x_y_df2k_33.2.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()

# Wrap model
wrapped_model = WaveMixPatchWrapper(model).to(device)

# Use a single 252x252 patch as dummy input
dummy_input = torch.randn(1, 3, 252, 252).to(device)

# Export to ONNX
torch.onnx.export(
    wrapped_model, dummy_input, "models/sr_block_2x_patch.onnx",
    input_names=["input"], output_names=["output"],
    opset_version=15,
    dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"},
                  "output": {0: "batch", 2: "height", 3: "width"}}
)

print("✅ Exported ONNX model for single 252x252 patch.")
