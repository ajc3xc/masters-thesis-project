import torch
import imageio

# --- Set paths ---
model_path = "skelite_scripted.pt"
image_path = "demo_data/drive_sample.png"

# --- Load the TorchScript model ---
model = torch.jit.load(model_path)
model.eval()

# --- Load test input image ---
img = imageio.imread(image_path) / 255.
img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
print(f"Image shape: {img.shape}")

# --- Run inference ---
with torch.no_grad():
    output, _ = model(img, z=None, no_iter=5)  # adjust args as needed
    print(f"Output shape: {output.shape}")

# --- Optionally visualize or assert something ---
import matplotlib.pyplot as plt
plt.imshow(output[0,0].cpu().numpy() > 0.5, cmap="gray")
plt.title("Output Skeleton Prediction")
plt.axis('off')
plt.show()
