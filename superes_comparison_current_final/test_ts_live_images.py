#!/usr/bin/env python3
import torch
import torchvision.transforms as T
from PIL import Image
import os

device = torch.device(args.device)

def load_model(pt_path, device):
    model = torch.load(pt_path, map_location=device)
    model.eval()
    return model

def preprocess_image(img_path, device):
    image = Image.open(img_path).convert("RGB")
    tensor = T.ToTensor()(image).unsqueeze(0).to(device)  # Shape: [1, 3, H, W]
    return tensor, image.size  # image.size is (W, H)

def test_images(pt_path, image_paths):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(pt_path, device)

    for path in image_paths:
        input_tensor, (W, H) = preprocess_image(path, device)
        with torch.no_grad():
            output = model(input_tensor)
        print(f"{os.path.basename(path)} ({H}×{W}) → Output shape {tuple(output.shape)}")

if __name__ == "__main__":
    image_paths = [
        "/mnt/stor/gchen-lab/data/Adam/masters-thesis-project/data/crack_segmentation_unzipped/crack_segmentation/virginia_tech_labeled_cracks_wild/LCW Concrete Crack Detection/LCW Concrete Crack Detection/512x512/Test/images/0.jpeg",
        "/mnt/stor/gchen-lab/data/Adam/masters-thesis-project/data/crack_segmentation_unzipped/crack_segmentation/virginia_tech_labeled_cracks_wild/LCW Concrete Crack Detection/LCW Concrete Crack Detection/512x512/Test/images/1.jpeg",
        "/mnt/stor/gchen-lab/data/Adam/masters-thesis-project/data/crack_segmentation_unzipped/crack_segmentation/virginia_tech_labeled_cracks_wild/LCW Concrete Crack Detection/LCW Concrete Crack Detection/512x512/Test/images/2.jpeg",
        "/mnt/stor/gchen-lab/data/Adam/masters-thesis-project/data/crack_segmentation_unzipped/crack_segmentation/virginia_tech_labeled_cracks_wild/LCW Concrete Crack Detection/LCW Concrete Crack Detection/512x512/Test/images/3.jpeg",
        "/mnt/stor/gchen-lab/data/Adam/masters-thesis-project/data/crack_segmentation_unzipped/crack_segmentation/virginia_tech_labeled_cracks_wild/LCW Concrete Crack Detection/LCW Concrete Crack Detection/512x512/Test/images/4.jpeg",
        # Add more image paths here
    ]
    test_images("/mnt/stor/gchen-lab/data/Adam/masters-thesis-project/superes_comparison_current_final/dcsf_dynamic_ts.pt", image_paths)
