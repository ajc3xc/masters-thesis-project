import torch
from WaveMixSR.WaveMixSRV2 import SR_Block

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Match architecture used in pretrained model
model = SR_Block(
    depth=4,          # usually 4; try 3 or 2 if mismatch
    mult=1,
    final_dim=144,
    dropout=0.3
).to(device)

# Load pretrained weights
state_dict = torch.load("models/bsd100_2x_y_df2k_33.2.pth", map_location=device)
model.load_state_dict(state_dict)

model.eval()
print("✅ Pretrained SR_Block loaded successfully!")
