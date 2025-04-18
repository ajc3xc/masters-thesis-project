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
        B, C, H, W = x.shape
        return F.interpolate(x, mode=self.mode, scale_factor=self.scale_factor, align_corners=False)


class FlattenPatchesWrapper(nn.Module):
    def __init__(self, model, patch_size=252):
        super().__init__()
        self.model = model
        self.patch_size = patch_size

    def forward(self, x):
        # x: [1, 3, H, W]
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"H and W must be divisible by patch size ({self.patch_size})"
        Hn, Wn = H // self.patch_size, W // self.patch_size

        # Split into patches: [1, 3, H, W] → [B*N, 3, 252, 252]
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, C, Hn, Wn, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, C, self.patch_size, self.patch_size)

        # Run model
        out = self.model(patches)  # [B*N, 3, H', W']
        return out


# Load model and patch interpolate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SR_Block(depth=4, mult=1, final_dim=144, ff_channel=144, dropout=0.3).to(device)
model.path1[0] = SafeInterp(mode="bicubic", scale_factor=2)

# Load weights
state_dict = torch.load("models/bsd100_2x_y_df2k_33.2.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()

# Wrap with flattening
wrapped_model = FlattenPatchesWrapper(model).to(device)

# Dummy input: [1, 3, 504, 504]
dummy_input = torch.randn(1, 3, 504, 504).to(device)

# Export to ONNX
torch.onnx.export(
    wrapped_model, dummy_input, "models/sr_block_2x_flattened.onnx",
    input_names=["input"], output_names=["output"],
    opset_version=15,
    dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"},
                  "output": {0: "batch", 2: "height", 3: "width"}}
)

print("✅ Exported ONNX to models/sr_block_2x_flattened.onnx")
