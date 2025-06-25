import paddle
import numpy as np
from pathlib import Path
import cv2
import time
import csv

# === Import all models you want to test here ===
from models.hrsegnet_b16 import HrSegNetB16
from models.hrsegnet_b32 import HrSegNetB32
from models.crackscopenet import CrackScopeNet_Tiny, CrackScopeNet_Small, CrackScopeNet, CrackScopeNet_Large
from models.crackscopenet_v2 import (
    CrackScopeNet_Vortex, CrackScopeNet_MSCASPP,
    CrackScopeNet_Large_Vortex, CrackScopeNet_Large_MSCASPP,
)
from eval import eval_from_memory  # Use your preferred metrics implementation

def get_image_mask_pairs(base_dir, txt_path, max_imgs=None):
    image_paths, mask_paths = [], []
    with open(txt_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            img_rel, mask_rel = line.strip().split()
            image_paths.append((base_dir / Path(img_rel)).resolve())
            mask_paths.append((base_dir / Path(mask_rel)).resolve())
            if max_imgs and len(image_paths) >= max_imgs:
                break
    return image_paths, mask_paths

def preprocess_image(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    return img

def load_model(model_class, weights_path, num_classes=2):
    model = model_class(num_classes=num_classes)
    if weights_path and Path(weights_path).exists():
        state_dict = paddle.load(str(weights_path))
        model.set_state_dict(state_dict)
    model.eval()
    return model

def evaluate_model(model, image_paths, mask_paths, device='gpu'):
    preds, gts, total_time = [], [], 0.0

    print("Warming up the model...")
    dummy_img = preprocess_image(image_paths[0])
    dummy_tensor = paddle.to_tensor(dummy_img)
    if device == 'gpu':
        dummy_tensor = dummy_tensor.cuda()
    with paddle.no_grad():
        for _ in range(4):
            _ = model(dummy_tensor)

    for idx, (img_p, mask_p) in enumerate(zip(image_paths, mask_paths), 1):
        print(f"   [{idx}/{len(image_paths)}] {img_p.name}")
        img = preprocess_image(img_p)
        mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Could not read mask: {mask_p}. Skipping.")
            continue
        gt = (mask > 0).astype('uint8') * 255
        img_tensor = paddle.to_tensor(img)
        if device == 'gpu':
            img_tensor = img_tensor.cuda()
        start = time.time()
        with paddle.no_grad():
            out = model(img_tensor)
        end = time.time()
        total_time += (end - start)
        if isinstance(out, (list, tuple)):
            out = out[0]
        elif isinstance(out, dict):
            out = out.get('out', out)
        out = paddle.nn.functional.sigmoid(out)
        out = out.numpy().squeeze()
        if out.ndim == 3:
            out = out[0] if out.shape[0] == 1 else np.argmax(out, axis=0)
        pred_mask = (out > 0.5).astype('uint8') * 255
        preds.append(pred_mask)
        gts.append(gt)
    fps = len(preds) / total_time if total_time > 0 else 0
    return preds, gts, fps

if __name__ == "__main__":
    paddle.set_device('gpu' if paddle.is_compiled_with_cuda() else 'cpu')
    MAX_IMGS = None  # e.g. 10 for fast debugging, None for full eval

    # Define your models, classes and checkpoints (18 total for your project)
    MODELS = [
        # Original
        {"label": "csn_tiny", "class": CrackScopeNet_Tiny, "ckpt": r"D:\camerer_ml\outputs_crackscopenet_tiny_crackseg9k\best_model\model.pdparams"},
        {"label": "csn_small", "class": CrackScopeNet_Small, "ckpt": r"D:\camerer_ml\outputs_crackscopenet_small_crackseg9k\best_model\model.pdparams"},
        {"label": "hrsegnetb16_orig", "class": HrSegNetB16, "ckpt": r"D:\camerer_ml\hrsegnet\hrsegnet_models\hrsegnetb16\best_model\model.pdparams"},
        {"label": "hrsegnetb32_orig", "class": HrSegNetB32, "ckpt": r"D:\camerer_ml\hrsegnet\hrsegnet_models\hrsegnetb32\best_model\model.pdparams"},
        {"label": "csn_orig", "class": CrackScopeNet, "ckpt": r"D:\camerer_ml\outputs_crackscopenet_crackseg9k\best_model\model.pdparams"},
        {"label": "csn_large_orig", "class": CrackScopeNet_Large, "ckpt": r"D:\camerer_ml\outputs_crackscopenet_large_crackseg9k\best_model\model.pdparams"},
        {"label": "csn_large_vortex_orig", "class": CrackScopeNet_Large_Vortex, "ckpt": r"E:\camerer_ml\finished_models\crackscopenet_large_vortex_9k\best_model\model.pdparams"},
        {"label": "csn_large_mscaspp_orig", "class": CrackScopeNet_Large_MSCASPP, "ckpt": r"E:\camerer_ml\finished_models\crackscopenet_large_mscaspp_9k\best_model\model.pdparams"},
        # Transfer 1
        #{"label": "hrsegnetb16_trans1", "class": HrSegNetB16, "ckpt": r"E:\camerer_ml\finished_models\hrsegnetb16_tl\model.pdparams"},
        #{"label": "hrsegnetb32_trans1", "class": HrSegNetB32, "ckpt": r"E:\camerer_ml\finished_models\hrsegnetb32_tl\model.pdparams"},
        #{"label": "csn_trans1", "class": CrackScopeNet, "ckpt": r"D:\camerer_ml\outputs_crackscopenet_tl_concrete3k\best_model\model.pdparams"},
        #{"label": "csn_large_trans1", "class": CrackScopeNet_Large, "ckpt": r"D:\camerer_ml\outputs_crackscopenet_large_tl_concrete3k\best_model\model.pdparams"},
        #{"label": "csn_large_vortex_trans1", "class": CrackScopeNet_Large_Vortex, "ckpt": r"E:\camerer_ml\finished_models\crackscopenet_large_vortex_tl\best_model\model.pdparams"},
        #{"label": "csn_large_mscaspp_trans1", "class": CrackScopeNet_Large_MSCASPP, "ckpt": r"E:\camerer_ml\finished_models\crackscopenet_large_mscaspp_tl\best_model\model.pdparams"},
        # Fine-tune 2
        {"label": "hrsegnetb16_ft2", "class": HrSegNetB16, "ckpt": r"E:\camerer_ml\finished_models\hrsegnetb16_ft\model.pdparams"},
        {"label": "hrsegnetb32_ft2", "class": HrSegNetB32, "ckpt": r"E:\camerer_ml\finished_models\hrsegnetb32_ft\model.pdparams"},
        {"label": "csn_ft2", "class": CrackScopeNet, "ckpt": r"D:\camerer_ml\outputs_crackscopenet_ft_concrete3k\best_model\model.pdparams"},
        {"label": "csn_large_ft2", "class": CrackScopeNet_Large, "ckpt": r"D:\camerer_ml\outputs_crackscopenet_large_ft_concrete3k\best_model\model.pdparams"},
        {"label": "csn_large_vortex_ft2", "class": CrackScopeNet_Large_Vortex, "ckpt": r"E:\camerer_ml\finished_models\crackscopenet_large_vortex_ft\best_model\model.pdparams"},
        {"label": "csn_large_mscaspp_ft2", "class": CrackScopeNet_Large_MSCASPP, "ckpt": r"E:\camerer_ml\finished_models\crackscopenet_large_mscaspp_ft\best_model\model.pdparams"},
        {"label": "csn_tiny_ft2", "class": CrackScopeNet_Tiny, "ckpt": r"D:\camerer_ml\outputs_crackscopenet_tiny_ft_concrete3k\best_model\model.pdparams"},
        {"label": "csn_small_ft2", "class": CrackScopeNet_Small, "ckpt": r"D:\camerer_ml\outputs_crackscopenet_small_ft_concrete3k\best_model\model.pdparams"},
    ]

    # Datasets: fill in as needed
    DATASETS = {
        "metu": {
            "base_dir": Path(r"D:\camerer_ml\datasets\METU_Concrete_Crack_Seg"),
            "txt": Path(r"D:\camerer_ml\datasets\METU_Concrete_Crack_Seg\test_metu.txt"),
            "csv": "eval_metu.csv",
        },
        "concrete3k": {
            "base_dir": Path(r"D:\camerer_ml\datasets\concrete3k\concrete3k"),
            "txt": Path(r"D:\camerer_ml\datasets\concrete3k\concrete3k\test_whole.txt"),
            "csv": "eval_concrete3k.csv",
        }
    }

    output_folder = Path("D:\camerer_ml\outputs_eval_small")
    output_folder.mkdir(exist_ok=True)

    for dsname, cfg in DATASETS.items():
        image_paths, mask_paths = get_image_mask_pairs(cfg["base_dir"], cfg["txt"], max_imgs=MAX_IMGS)
        with open(output_folder / cfg["csv"], "w", newline='') as f:
            writer = csv.writer(f)
            header_written = False
            for m in MODELS:
                print(f"▶ Evaluating {m['label']} on {dsname} ...")
                model = load_model(m["class"], m["ckpt"], num_classes=2)
                preds, gts, fps = evaluate_model(model, image_paths, mask_paths)
                metrics = eval_from_memory(preds, gts)
                metrics.update({"FPS": fps, "N": len(preds)})
                if not header_written:
                    writer.writerow(["Model", "Label"] + list(metrics.keys()))
                    header_written = True
                writer.writerow([m["class"].__name__, m["label"]] + list(metrics.values()))
                f.flush()
                print(f"✅ Done {m['label']} on {dsname}")
