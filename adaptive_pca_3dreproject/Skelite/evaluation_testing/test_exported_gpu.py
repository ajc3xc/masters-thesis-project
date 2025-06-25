import torch
import imageio
import time

def load_image(image_path, device):
    img = imageio.imread(image_path)
    if img.ndim == 3:
        img = img.mean(axis=2)
    img = img / 255.0
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    return img

def measure_inference_time(model_path, image_path, device, warmup=True):
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    img = load_image(image_path, device)

    if warmup:
        for _ in range(5):
            with torch.no_grad():
                _ = model(img, z=None, no_iter=5)

    start = time.time()
    with torch.no_grad():
        _ = model(img, z=None, no_iter=5)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end = time.time()

    return 1000 * (end - start)  # ms

# --- Set paths
model_path = r"D:\camerer_ml\adaptive_pca_3dreproject\Skelite\skelite_scripted.pt"
#image_path = r"D:\camerer_ml\adaptive_pca_3dreproject\Skelite\demo_data\drive_sample.png"
image_path = r"D:\camerer_ml\datasets\METU_Concrete_Crack_Seg\BW_visible\024.png"

# --- CPU timing
cpu_time = measure_inference_time(model_path, image_path, torch.device("cpu"))
print(f"CPU Inference Time: {cpu_time:.2f} ms")

# --- GPU timing (if available)
if torch.cuda.is_available():
    gpu_time = measure_inference_time(model_path, image_path, torch.device("cuda"))
    print(f"GPU Inference Time: {gpu_time:.2f} ms")
else:
    print("GPU not available.")