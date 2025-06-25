import csv
import time
import torch
import cv2
from pathlib import Path
import numpy as np

from eval import eval_from_memory

def load_xception_model(weights_path, device):
    from modeling.deeplab import DeepLab
    model = DeepLab(num_classes=2, backbone='xception', output_stride=16, sync_bn=False, freeze_bn=False)
    ckpt = torch.load(weights_path, map_location=device)
    state_dict = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model

def get_image_mask_pairs(base_dir, txt_path, max_imgs=None):
    """
    Reads a txt file where each line is:
        images\458_10.jpg labels_01\458_10.png
    Returns two lists: image_paths, mask_paths (both absolute Path objects)
    """
    image_paths = []
    mask_paths = []
    with open(txt_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            img_rel, mask_rel = line.strip().split()
            img_path = (base_dir / Path(img_rel)).resolve()
            mask_path = (base_dir / Path(mask_rel)).resolve()
            image_paths.append(img_path)
            mask_paths.append(mask_path)
            if max_imgs and len(image_paths) >= max_imgs:
                break
    return image_paths, mask_paths

def collect_deeplabv3_xception(model, image_paths, mask_paths, device):
    preds, gts, total_time = [], [], 0.0

    dummy = torch.randn(1,3,512,512, device=device)
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy)
    
    for i, (img_p, mask_p) in enumerate(zip(image_paths, mask_paths), 1):
        print(f"[{i}/{len(image_paths)}] {img_p.name}")
        bgr = cv2.imread(str(img_p))
        if bgr is None:
            print(f"  Skipping: Could not read {img_p}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        im = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_CUBIC)
        #im = im.astype(np.float32)
        tensor = torch.from_numpy(im.transpose(2,0,1)).float().unsqueeze(0).to(device)
        if device.type=='cuda': torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = model(tensor)
            if isinstance(out, dict): out = out.get('out', out)
        if device.type=='cuda': torch.cuda.synchronize()
        t1 = time.time()
        total_time += (t1 - t0)
        out = out.squeeze(0)
        #print(out.ndim, out.shape)
        if out.ndim == 2:
            prob = torch.sigmoid(out)
            mask_pred = (prob > 0.5).cpu().numpy().astype('uint8')
        elif out.ndim == 3 and out.shape[0] == 2:
            mask_pred = out.argmax(0).cpu().numpy().astype('uint8')
        else:
            raise ValueError(f"Unexpected output shape: {out.shape}")
        #else:
        #    mask_pred = out.argmax(0).cpu().numpy().astype('uint8')
        preds.append(mask_pred)
        gt = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            print(f"  Skipping: Could not read {mask_p}")
            continue
        gt = (gt > 0).astype('uint8')
        gts.append(gt)

        # === Add these print statements, for the first image only ===
        if i == 1:
            print("!!!!!!")
            print("Unique values in prediction:", np.unique(mask_pred))
            print("Unique values in ground truth:", np.unique(gt))
            cv2.imwrite(f"pred.png", mask_pred * 255)
            cv2.imwrite(f"gt.png", gt * 255)
            print("out shape:", out.shape)
            print("out min/max/mean:", out.min().item(), out.max().item(), out.mean().item())
            import sys
            sys.exit()
        # ===========================================================
    fps = len(preds)/total_time if total_time > 0 else 0
    return preds, gts, fps


if __name__=="__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_IMGS = None  # Or an integer to limit
    MODELS = {
        "imagenet_crackseg9k": r"D:\camerer_ml\pytorch_deeplab_xception\pytorch-deeplab-xception\run\crackseg9k\crackseg9k_pretrain_imagenet\experiment_0\checkpoint.pth.tar",
        "nopretrain_crackseg9k": r"E:\camerer_ml\finished_models\deeplabv3p_xception\nopretrain_crackseg9k\checkpoint.pth.tar",
        "imagenet_ft_concrete3k": r"D:\camerer_ml\pytorch_deeplab_xception\pytorch-deeplab-xception\run\concrete3k\concrete3k_xception_ft\model_best.pth.tar",
        "nopretrain_ft_concrete3k": r"E:\camerer_ml\finished_models\deeplabv3p_xception\nopretrain_ft_concrete3k\checkpoint.pth.tar",
    }
    DATASETS = {
        "concrete3k": {
            "base_dir": Path(r"D:\camerer_ml\datasets\concrete3k\concrete3k"),
            "txt": Path(r"D:\camerer_ml\datasets\concrete3k\concrete3k\test_whole.txt"),
            "csv": "eval_concrete3k_xception_normalized.csv",
        },
        "metu": {
            "base_dir": Path(r"D:\camerer_ml\datasets\METU_Concrete_Crack_Seg"),
            "txt": Path(r"D:\camerer_ml\datasets\METU_Concrete_Crack_Seg\test_metu.txt"),
            "csv": "eval_metu_xception_normalized.csv",
        }
    }

    output_folder = Path(r"D:\camerer_ml\outputs_eval")
    for dataset, cfg in DATASETS.items():
        out_csv = output_folder / cfg["csv"]
        print(f"\n=== Evaluating on dataset: {dataset} ===")
        image_paths, mask_paths = get_image_mask_pairs(cfg["base_dir"], cfg["txt"], max_imgs=MAX_IMGS)
        with open(out_csv, 'w', newline='') as f:
            w = csv.writer(f)
            first = True
            for model_name, weights in MODELS.items():
                print(f"▶ Evaluating model: {model_name} ...")
                model = load_xception_model(weights, DEVICE)
                preds, gts, fps = collect_deeplabv3_xception(
                    model, image_paths, mask_paths, device=DEVICE
                )
                metrics = eval_from_memory(preds, gts)
                metrics.update({'FPS': fps, 'Iters': len(preds)})
                if first:
                    w.writerow(['Model'] + list(metrics.keys()))
                    first = False
                w.writerow([model_name] + list(metrics.values()))
                f.flush()
                print(f"✅ {model_name} results added to {out_csv}")
