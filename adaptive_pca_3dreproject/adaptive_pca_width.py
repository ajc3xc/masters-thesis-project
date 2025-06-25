import os
import cv2
import numpy as np
from adaptive_pca_test import adaptive_crack_width, draw_width_lines, save_crack_widths_csv

def main(mask_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask.max() > 1:
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.dilate(mask_clean, kernel, iterations=1)  # Optional

    widths = adaptive_crack_width(mask_clean)
    overlay = draw_width_lines(mask_clean, widths)
    base = os.path.splitext(os.path.basename(mask_path))[0]

    overlay_path = os.path.join(output_dir, f"{base}_width_overlay.png")
    csv_path = os.path.join(output_dir, f"{base}_crack_widths.csv")

    cv2.imwrite(overlay_path, overlay)
    save_crack_widths_csv(widths, csv_path)

    print(f"Overlay saved: {overlay_path}")
    print(f"Widths CSV saved: {csv_path}")

if __name__ == "__main__":
    # Change these as needed:
    mask_path = r"D:\camerer_ml\datasets\concrete3k\concrete3k\labels_01_visible\001_2.png"  # From PaddleSeg, or use a GT mask
    output_dir = "width_outputs"
    main(mask_path, output_dir)
