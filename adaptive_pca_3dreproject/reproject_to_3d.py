import os
import numpy as np
import pandas as pd
import cv2
from backprojection import backproject_point  # You will add this function, see below

def backproject_point(x, y, depth, K, T_cw):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z = depth[int(y), int(x)]
    if z <= 0:
        return None
    x_cam = (x - cx) * z / fx
    y_cam = (y - cy) * z / fy
    pt_cam = np.array([x_cam, y_cam, z])
    # World coordinates
    R_wc = T_cw[:3, :3].T
    t_wc = -R_wc @ T_cw[:3, 3]
    pt_world = R_wc @ pt_cam + t_wc
    return pt_world

def main(widths_csv, depth_path, K_path, Tcw_path, out_csv):
    # Load data
    df = pd.read_csv(widths_csv)
    depth = np.load(depth_path)
    K = np.load(K_path)
    T_cw = np.load(Tcw_path)

    output_rows = []
    for i, row in df.iterrows():
        x, y, width = row['x'], row['y'], row['width_px']
        pt_center = backproject_point(x, y, depth, K, T_cw)
        # For edge points (project width along normal; here we just estimate ±width/2 in x)
        # If you have normal vectors, add them to the CSV in script 1 and use them here!
        # Otherwise, just output center for now.
        if pt_center is not None:
            output_rows.append({'x_2d': x, 'y_2d': y, 'width_px': width,
                                'X': pt_center[0], 'Y': pt_center[1], 'Z': pt_center[2]})

    out_df = pd.DataFrame(output_rows)
    out_df.to_csv(out_csv, index=False)
    print(f"3D crack points saved: {out_csv}")

if __name__ == "__main__":
    # Paths to your files
    widths_csv = "width_outputs/test_mask_crack_widths.csv"
    depth_path = "projection_data00/depth_00.npy"
    K_path = "projection_data00/K.npy"
    Tcw_path = "projection_data00/Tcw_00.npy"
    out_csv = "crack_points_3d.csv"
    main(widths_csv, depth_path, K_path, Tcw_path, out_csv)
