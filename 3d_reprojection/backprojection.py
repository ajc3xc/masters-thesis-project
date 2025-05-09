import numpy as np
from scipy.spatial.transform import Rotation as R

def sample_perturbed_pose(T_cw, rotation_std=0.01, translation_std=0.001):
    R_cw = T_cw[:3, :3]
    t_cw = T_cw[:3, 3]

    # Small-angle rotation noise using axis-angle representation
    rot_noise = R.from_rotvec(np.random.normal(0, rotation_std, size=3)).as_matrix()
    R_cw_perturbed = rot_noise @ R_cw

    # Small translation noise
    t_cw_perturbed = t_cw + np.random.normal(0, translation_std, size=3)

    # Compose perturbed pose
    T_perturbed = np.eye(4)
    T_perturbed[:3, :3] = R_cw_perturbed
    T_perturbed[:3, 3] = t_cw_perturbed
    return T_perturbed

def backproject_surface_and_crack(K, depth_map, mask, T_cw, perturbation_samples=25, include_noncrack=False):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    if include_noncrack:
        ys, xs = np.nonzero(depth_map > 0)
    else:
        ys, xs = np.where(mask > 0)

    points_mean = []
    points_var = []
    crack_flags = []

    for u, v in zip(xs, ys):
        z = depth_map[v, u]
        if z <= 0:
            continue

        x_cam = (u - cx) * z / fx
        y_cam = (v - cy) * z / fy
        pt_cam = np.array([x_cam, y_cam, z])

        samples = []
        for _ in range(perturbation_samples):
            T_j = sample_perturbed_pose(T_cw)
            R_wc = T_j[:3, :3].T
            t_wc = -R_wc @ T_j[:3, 3]
            pt_world = R_wc @ pt_cam + t_wc
            samples.append(pt_world)

        samples = np.stack(samples)
        points_mean.append(np.mean(samples, axis=0))
        points_var.append(np.var(samples, axis=0))
        crack_flags.append(mask[v, u] > 0)

    return (
        np.array(points_mean),
        np.array(points_var),
        np.array(crack_flags, dtype=bool)
    )