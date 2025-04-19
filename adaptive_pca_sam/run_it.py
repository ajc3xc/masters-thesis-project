#!/usr/bin/env python3
"""
Full script to:
 1. Load a binary crack mask (e.g., from CrackMamba)
 2. Optionally refine it using EfficientViT-SAM L0
 3. Estimate crack widths via adaptive PCA
 4. Save the refined mask and overlay visualization

Usage:
  python efficient_sam_pca_width.py \
    --image path/to/image.jpg \
    --mask path/to/input_mask.png \
    --output_mask path/to/refined_mask.png \
    --output_overlay path/to/overlay.png

Requirements:
  pip install torch torchvision pillow scikit-image scikit-learn opencv-python
"""

import argparse
import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
from skimage.morphology import skeletonize
from sklearn.decomposition import PCA
import cv2
import os

try:
    from efficientvit.sam.model_zoo import create_sam_model
    from efficientvit.sam.utils import load_image
    from efficientvit.sam.predictor import SamPredictor
except ImportError:
    create_sam_model = None
    SamPredictor = None
    print("⚠️ EfficientViT-SAM not available. Refinement will be skipped.")

def adaptive_crack_width_estimation(refined_mask, curvature_threshold=0.1, min_window_size=5, max_window_size=21):
    skeleton = skeletonize(refined_mask > 0).astype(np.uint8)
    points = np.argwhere(skeleton > 0)
    widths = []

    def extract_neighborhood(mask, center, radius):
        x, y = center
        x1, x2 = max(x - radius, 0), min(x + radius + 1, mask.shape[0])
        y1, y2 = max(y - radius, 0), min(y + radius + 1, mask.shape[1])
        return mask[x1:x2, y1:y2], (x1, y1)

    def compute_curvature(neighborhood):
        ys, xs = np.nonzero(neighborhood)
        coords = np.stack((xs, ys), axis=1)
        if len(coords) < 3:
            return 0.0
        pca = PCA(n_components=2).fit(coords)
        return pca.explained_variance_ratio_[1]

    def sample_normal_direction(center, normal_vector, binary_mask, max_distance=20):
        x0, y0 = center
        edge_points = []
        for sign in [-1, 1]:
            for d in range(1, max_distance):
                dx = int(round(sign * normal_vector[0] * d))
                dy = int(round(sign * normal_vector[1] * d))
                x, y = x0 + dx, y0 + dy
                if 0 <= x < binary_mask.shape[0] and 0 <= y < binary_mask.shape[1]:
                    if binary_mask[x, y] == 0:
                        edge_points.append((x, y))
                        break
        return edge_points if len(edge_points) == 2 else None

    for (x, y) in points:
        patch, _ = extract_neighborhood(skeleton, (x, y), max_window_size // 2)
        curvature = compute_curvature(patch)

        radius = min_window_size // 2 if curvature > curvature_threshold else max_window_size // 2
        patch, (x0, y0) = extract_neighborhood(skeleton, (x, y), radius)
        ys, xs = np.nonzero(patch)
        coords = np.stack((xs, ys), axis=1)

        if len(coords) < 3:
            continue

        pca = PCA(n_components=2).fit(coords)
        normal = pca.components_[1]

        edge_pts = sample_normal_direction((x, y), normal, refined_mask)
        if edge_pts is not None:
            d = np.linalg.norm(np.array(edge_pts[0]) - np.array(edge_pts[1]))
            widths.append((x, y, d))

    return widths

def refine_mask_with_sam(image_path, mask_array):
    if create_sam_model is None:
        return mask_array

    model = create_sam_model("l0", pretrained=True).eval().cuda()
    image_pil, image_np = load_image(image_path)
    predictor = SamPredictor(model)
    predictor.set_image(image_np)

    mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0).float()
    input_boxes = predictor.get_boxes_from_mask(mask_tensor)

    masks, _, _ = predictor.predict_torch(boxes=input_boxes, multimask_output=False)
    combined = masks.squeeze().cpu().numpy().astype(np.uint8)
    return combined

def main():
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--image", type=str, help="Optional original image path")
    #parser.add_argument("--mask", type=str, required=True, help="Input crack mask (from CrackMamba)")
    #parser.add_argument("--output_mask", type=str, default="refined_mask.png")
    #parser.add_argument("--output_overlay", type=str, default="overlay.png")
    image = "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/data/cracks/crack_segmentation_dataset/images/CFD_001.jpg"
    mask = "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/data/cracks/crack_segmentation_dataset/masks/CFD_001.jpg"
    output_mask = "output_mask.png"
    output_overlay = "output_overlay.png"

    mask = np.array(Image.open(mask).convert("1"), dtype=np.uint8)
    if image:
        refined_mask = refine_mask_with_sam(image, mask)
    else:
        refined_mask = mask

    Image.fromarray((refined_mask * 255).astype(np.uint8)).save(output_mask)

    widths = adaptive_crack_width_estimation(refined_mask)

    if image:
        base = Image.open(image).convert("RGB").resize((refined_mask.shape[1], refined_mask.shape[0]))
    else:
        base = Image.fromarray((refined_mask * 255).astype(np.uint8)).convert("RGB")

    draw = ImageDraw.Draw(base)
    for x, y, d in widths:
        r = d / 2
        draw.ellipse([(y - r, x - r), (y + r, x + r)], outline="red", width=1)

    base.save(output_overlay)
    print(f"✅ Saved overlay to {output_overlay} with {len(widths)} widths estimated")

if __name__ == '__main__':
    main()