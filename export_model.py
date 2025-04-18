import torch
from WaveMixSR.WaveMixSRV2 import SR_Block

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SR_Block(
    depth=4,
    mult=1,
    final_dim=144,
    ff_channel=144,
    dropout=0.3
).to(device)

# Load pretrained weights
state_dict = torch.load("models/bsd100_2x_y_df2k_33.2.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 3, 256, 256).to(device)
torch.onnx.export(
    model, dummy_input, "models/sr_block_2x.onnx",
    input_names=["input"], output_names=["output"],
    opset_version=15,  # or 13 for Jetson 4.x
    dynamic_axes={
        "input": {0: "batch", 2: "height", 3: "width"},
        "output": {0: "batch", 2: "height", 3: "width"}
    }
)
print("✅ Exported to ONNX")

# Export to TorchScript (.pt)
traced = torch.jit.trace(model, dummy_input)
traced.save("models/sr_block_2x.pt")
print("✅ Exported to TorchScript")