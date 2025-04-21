import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import onnxruntime as ort
import matplotlib.pyplot as plt
import os
from skimage.morphology import skeletonize

# === USER CONFIGURATION ===
ENCODER_MODEL_PATH = "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/adaptive_pca_sam/efficientvit-sam/onnx/l0_encoder.onnx"
DECODER_MODEL_PATH = "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/adaptive_pca_sam/efficientvit-sam/onnx/l0_decoder.onnx"
IMAGE_PATH = "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/data/cracks/crack_segmentation_dataset/images/CFD_001.jpg"
MASK_PATH = "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/data/cracks/crack_segmentation_dataset/masks/CFD_001.jpg"  # Optional mask for prompts
MODEL_VARIANT = "l0"  # or "xl0"
OUT_PATH = f"crack_sam_output_{MODEL_VARIANT}.png"
NUM_PROMPTS = 6       # number of skeleton points to sample

# === IMAGE PREPROCESSING ===
def preprocess_image(img: np.ndarray, target_size: int = 512):
    pixel_mean = [123.675 / 255, 116.28 / 255, 103.53 / 255]
    pixel_std  = [58.395 / 255, 57.12 / 255, 57.375 / 255]

    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    newh, neww = int(h * scale + 0.5), int(w * scale + 0.5)
    img_resized = cv2.resize(img, (neww, newh), interpolation=cv2.INTER_LINEAR)

    tensor = torch.tensor(img_resized).permute(2, 0, 1).float() / 255.0
    tensor = T.Normalize(mean=pixel_mean, std=pixel_std)(tensor)
    padded = torch.zeros((3, target_size, target_size))
    padded[:, :newh, :neww] = tensor
    return padded.unsqueeze(0).numpy(), (h, w), (newh, neww)

# === PROMPT FROM VERIFIED MASK ===
def get_skeleton_prompts_from_array(mask: np.ndarray, num_points=5):
    bin_mask = (mask > 127).astype(np.uint8)
    skeleton = skeletonize(bin_mask).astype(np.uint8)
    ys, xs = np.where(skeleton)
    if len(xs) == 0:
        raise ValueError("No skeleton pixels found.")
    idx = np.linspace(0, len(xs) - 1, num=num_points, dtype=int)
    coords = np.stack([xs[idx], ys[idx]], axis=-1).astype(np.float32)
    coords = coords[None, :, :]  # (1, N, 2)
    labels = np.ones((1, coords.shape[1]), dtype=np.float32)
    return coords, labels

# === SAM INFERENCE ===
def run_sam_onnx(encoder_path, decoder_path, image_input, orig_size, point_coords, point_labels, target_size):
    encoder = ort.InferenceSession(encoder_path, providers=["CUDAExecutionProvider"])
    decoder = ort.InferenceSession(decoder_path, providers=["CUDAExecutionProvider"])

    embedding = encoder.run(None, {encoder.get_inputs()[0].name: image_input})[0]

    oldh, oldw = orig_size
    scale = target_size / max(oldh, oldw)
    newh, neww = int(oldh * scale + 0.5), int(oldw * scale + 0.5)
    point_coords = point_coords.copy()
    point_coords[..., 0] *= (neww / oldw)
    point_coords[..., 1] *= (newh / oldh)

    output = decoder.run(None, {
        "image_embeddings": embedding,
        "point_coords": point_coords,
        "point_labels": point_labels,
    })[0]

    masks = torch.tensor(output)
    masks = F.interpolate(masks, size=(newh, neww), mode="bilinear", align_corners=False)
    masks = masks[..., :newh, :neww]
    masks = F.interpolate(masks, size=(oldh, oldw), mode="bilinear", align_corners=False)
    return masks.squeeze().numpy()

# === MAIN EXECUTION ===
if __name__ == "__main__":
    #os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    image_bgr = cv2.imread(IMAGE_PATH)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    crack_mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)

    assert image_rgb is not None, "Image failed to load"
    assert crack_mask is not None, "Mask failed to load"

    if MODEL_VARIANT == "l0":
        target_size = 512
    elif MODEL_VARIANT == "xl0":
        target_size = 1024
    else:
        raise ValueError("Invalid model variant")
    print(target_size)

    preprocessed, orig_size, _ = preprocess_image(image_rgb, target_size)
    point_coords, point_labels = get_skeleton_prompts_from_array(crack_mask, num_points=NUM_PROMPTS)

    mask = run_sam_onnx(
        ENCODER_MODEL_PATH,
        DECODER_MODEL_PATH,
        preprocessed,
        orig_size,
        point_coords,
        point_labels,
        target_size
    )

    # === VISUALIZE RESULT ===
    plt.figure(figsize=(8, 8))
    plt.imshow(image_rgb)
    plt.imshow(mask > 0.5, alpha=0.5, cmap="Blues")

    # Show prompt points
    for (x, y) in point_coords[0]:
        plt.scatter([x], [y], color='lime', edgecolors='black', s=100, marker='*', linewidths=1.5)

    plt.axis("off")
    plt.savefig(OUT_PATH, bbox_inches="tight", dpi=300)
    print(f"✅ Saved SAM result to {OUT_PATH}")