import torch
import os
from sr_models.DSCF_SR.models.team23_DSCF import DSCF

# Hardcoded paths
CKPT_PATH = "sr_models/DSCF_SR/model_zoo/team23_DSCF.pth"
ONNX_PATH = "dscf_dynamic.onnx"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize and load model
model = DSCF(3, 3, feature_channels=26, upscale=4).to(DEVICE).eval()
state = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(state, strict=False)

# Dummy input with dynamic spatial dims
dummy = torch.randn(1, 3, 256, 256, device=DEVICE)

# Export
torch.onnx.export(
    model, dummy, ONNX_PATH,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {2: 'height', 3: 'width'},
        'output': {2: 'height', 3: 'width'}
    },
    opset_version=11
)

print(f"✅ Exported to {ONNX_PATH}")
