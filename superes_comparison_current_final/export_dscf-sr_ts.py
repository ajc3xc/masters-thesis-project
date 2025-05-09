import torch
from sr_models.DSCF_SR.models.team23_DSCF import DSCF

# === Hardcoded paths ===
CKPT_PATH = "sr_models/DSCF_SR/model_zoo/team23_DSCF.pth"
TS_PATH = "dcsf_dynamic_ts.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Initialize and load model ===
model = DSCF(3, 3, feature_channels=26, upscale=4).to(DEVICE).eval()
state = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(state, strict=False)

# === Dummy input for tracing ===
dummy_input = torch.randn(1, 3, 256, 256, device=DEVICE)

# === Export to TorchScript ===
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save(TS_PATH)

print(f"✅ Exported TorchScript model to {TS_PATH}")
