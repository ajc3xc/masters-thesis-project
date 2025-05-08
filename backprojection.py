import numpy as np

import numpy as np

def backproject_surface_and_crack(K, depth_map, mask, T_cw):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    R_wc = T_cw[:3, :3].T
    t_wc = -R_wc @ T_cw[:3, 3]

    H, W = depth_map.shape
    points = []
    crack_flags = []

    for v in range(H):
        for u in range(W):
            z = depth_map[v, u]
            x_cam = (u - cx) * z / fx
            y_cam = (v - cy) * z / fy
            pt_cam = np.array([x_cam, y_cam, z])
            pt_world = R_wc @ pt_cam + t_wc
            points.append(pt_world)
            crack_flags.append(mask[v, u] > 0)

    return np.array(points), np.array(crack_flags, dtype=bool)