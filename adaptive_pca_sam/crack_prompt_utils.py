import numpy as np
import cv2

'''
ENCODER_PATH = "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/adaptive_pca_sam/efficientvit-sam/onnx/l0_encoder.onnx"
DECODER_PATH = "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/adaptive_pca_sam/efficientvit-sam/onnx/l0_decoder.onnx"
IMAGE_PATH = "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/data/cracks/crack_segmentation_dataset/images/CFD_001.jpg"
MASK_PATH = "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/data/cracks/crack_segmentation_dataset/masks/CFD_001.jpg"  # Optional mask for prompts
OUT_PATH = "crack_sam_output.png"
'''

def generate_prompt_points_from_mask(mask, num_points=5):
    """
    Convert a binary crack mask into N prompt points for SAM.
    """
    if np.max(mask) > 1:
        mask = mask / 255.0
    mask = (mask > 0.5).astype(np.uint8)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("No crack pixels found in mask.")

    idx = np.random.choice(len(xs), size=min(num_points, len(xs)), replace=False)
    coords = np.stack([xs[idx], ys[idx]], axis=-1).astype(np.float32)
    labels = np.ones(len(coords), dtype=np.float32)  # all positive points

    coords = coords[None, :, :]  # (1, N, 2)
    labels = labels[None, :]     # (1, N)
    return coords, labels
