#!/usr/bin/env python3
import onnxruntime as ort
import numpy as np
from PIL import Image
import os

def preprocess_image(path):
    img = Image.open(path).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0  # Normalize to [0,1]
    img_np = np.transpose(img_np, (2, 0, 1))  # Convert to CHW
    img_np = np.expand_dims(img_np, axis=0)  # Add batch dimension
    return img_np.astype(np.float32), img.size  # size is (W, H)

def test_images(onnx_path, image_paths):
    sess = ort.InferenceSession(
        onnx_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    for path in image_paths:
        inp, (W, H) = preprocess_image(path)
        out = sess.run(None, {'input': inp})
        print(f"{os.path.basename(path)} ({H}×{W}) → Output shape {out[0].shape}")

if __name__ == "__main__":
    image_paths = [
        "/mnt/stor/gchen-lab/data/Adam/masters-thesis-project/data/crack_segmentation_unzipped/crack_segmentation/virginia_tech_labeled_cracks_wild/LCW Concrete Crack Detection/LCW Concrete Crack Detection/512x512/Test/images/0.jpeg",
        "/mnt/stor/gchen-lab/data/Adam/masters-thesis-project/data/crack_segmentation_unzipped/crack_segmentation/virginia_tech_labeled_cracks_wild/LCW Concrete Crack Detection/LCW Concrete Crack Detection/512x512/Test/images/1.jpeg",
        "/mnt/stor/gchen-lab/data/Adam/masters-thesis-project/data/crack_segmentation_unzipped/crack_segmentation/virginia_tech_labeled_cracks_wild/LCW Concrete Crack Detection/LCW Concrete Crack Detection/512x512/Test/images/2.jpeg",
        "/mnt/stor/gchen-lab/data/Adam/masters-thesis-project/data/crack_segmentation_unzipped/crack_segmentation/virginia_tech_labeled_cracks_wild/LCW Concrete Crack Detection/LCW Concrete Crack Detection/512x512/Test/images/3.jpeg",
        "/mnt/stor/gchen-lab/data/Adam/masters-thesis-project/data/crack_segmentation_unzipped/crack_segmentation/virginia_tech_labeled_cracks_wild/LCW Concrete Crack Detection/LCW Concrete Crack Detection/512x512/Test/images/4.jpeg",
        # Add more image paths here
    ]
    test_images("dscf_dynamic.onnx", image_paths)
