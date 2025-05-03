import os
import glob
import time
import cv2
import numpy as np
import onnxruntime as ort
from eval.evaluate import eval as scsegamba_eval  # assumes PYTHONPATH includes eval/

def preprocess_color(image_path, input_size):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # May return None or grayscale fallback
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected 3-channel image but got shape {img.shape} from: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    pad = max(h, w)
    square = np.zeros((pad, pad, 3), dtype=img.dtype)
    square[:h, :w, :] = img
    square = cv2.resize(square, (input_size, input_size))
    inp = square.astype(np.float32) / 255.0
    inp = np.transpose(inp, (2, 0, 1))  # (C, H, W)
    return inp[np.newaxis, :, :, :]  # (1, 3, H, W)

def preprocess_grayscale(image_path, input_size):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    h, w = img.shape
    pad = max(h, w)
    square = np.zeros((pad, pad), dtype=img.dtype)
    square[:h, :w] = img
    square = cv2.resize(square, (input_size, input_size))
    inp = square.astype(np.float32) / 255.0
    return inp[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)


def run_model(session, input_tensor):
    session.run(None, {"input": input_tensor})  # warm-up
    start = time.time()
    out = session.run(None, {"input": input_tensor})
    end = time.time()
    return out[0], end - start

def evaluate_model_onnx(model_path, dataset_path, input_size):
    session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])

    model_tag = os.path.splitext(os.path.basename(model_path))[0]
    pred_dir = os.path.join("onnx_eval_results", model_tag)
    os.makedirs(pred_dir, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(dataset_path, 'images', '*.jpg')))
    total_time = 0.0
    pred_count = 0

    for img_path in image_paths:
        x = preprocess_grayscale(img_path, input_size)
        mask, t = run_model(session, x)
        total_time += t
        pred_mask = mask[0, 0] if mask.ndim == 4 else mask[0]
        pred_mask = (pred_mask / np.max(pred_mask)) * 255.0
        pred_mask = pred_mask.astype(np.uint8)

        base = os.path.splitext(os.path.basename(img_path))[0]
        gt_path = os.path.join(dataset_path, 'masks', base + '.jpg')
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        if gt is None:
            print(f"Missing GT mask: {gt_path}")
            continue

        gt = cv2.resize(gt, (input_size, input_size))
        pred_mask = cv2.resize(pred_mask, (input_size, input_size))

        #cv2.imwrite(os.path.join(pred_dir, f"{base}_pre.png"), pred_mask)
        #cv2.imwrite(os.path.join(pred_dir, f"{base}_lab.png"), gt)
        pred_count += 1

    fps = pred_count / total_time if total_time > 0 else 0
    return pred_dir, fps

def main():
    models = [
        "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/SCSegamba_improved/onnx_exports/Crack_Conglomerate_eca_best.onnx",
        #"/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/SCSegamba_improved/onnx_exports/Crack_Conglomerate_sebica_best.onnx",
        #"/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/SCSegamba_improved/onnx_exports/Crack_Conglomerate_gbc_eca_best.onnx",
        #"/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/SCSegamba_improved/onnx_exports/Crack_Conglomerate_sfa_best.onnx",
        #"/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/SCSegamba/onnx_exports/Crack_Conglomerate_best.onnx",

    ]

    datasets = {
        "crack_conglomerate_test": "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/data/crack_segmentation_unzipped/crack_segmentation/virginia_tech_concrete_crack_congolmeration/Conglomerate Concrete Crack Detection/Conglomerate Concrete Crack Detection/Test"
    }

    for model_path in models:
        for ds_name, ds_dir in datasets.items():
            sample = glob.glob(os.path.join(ds_dir, 'images', '*.jpg'))[0]
            input_size = max(cv2.imread(sample).shape[:2])
            pred_dir, fps = evaluate_model_onnx(model_path, ds_dir, input_size)

            print(f"\n🔍 Evaluating {os.path.basename(model_path)} on {ds_name}")
            metrics = scsegamba_eval(None, pred_dir, epoch=0)
            metrics['FPS'] = fps

            for k, v in metrics.items():
                print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
        print(f"{model_path} evaluated")

if __name__ == "__main__":
    main()
