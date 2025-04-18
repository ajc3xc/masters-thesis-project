import os
import onnxruntime as ort
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
from torchvision import transforms
import torch

def load_onnx_model(path):
    return ort.InferenceSession(path)

def preprocess_ycbcr(image, size=256):
    image = image.convert('YCbCr').resize((size, size))
    y, cb, cr = image.split()
    return y, cb, cr

def run_sr(sess, y_tensor):
    y_np = y_tensor.unsqueeze(0).numpy().astype(np.float32)
    out = sess.run(['output'], {'input': y_np})[0]
    return torch.tensor(out).squeeze(0)

def merge_ycbcr(y_sr, cb, cr):
    cb_up = torch.nn.functional.interpolate(cb.unsqueeze(0), scale_factor=2, mode='bicubic')
    cr_up = torch.nn.functional.interpolate(cr.unsqueeze(0), scale_factor=2, mode='bicubic')
    img = torch.cat([y_sr, cb_up.squeeze(0), cr_up.squeeze(0)], dim=0).clamp(0,1).permute(1,2,0).cpu().numpy()
    return Image.fromarray((img * 255).astype('uint8'), mode='YCbCr').convert('RGB')

def evaluate_image(sr_img, hr_img):
    sr_np = np.array(sr_img.resize(hr_img.size)) / 255.0
    hr_np = np.array(hr_img) / 255.0
    psnr = peak_signal_noise_ratio(hr_np, sr_np, data_range=1.0)
    ssim = structural_similarity(hr_np, sr_np, multichannel=True, data_range=1.0)
    return psnr, ssim

def evaluate_masks(pred_mask, true_mask):
    y_pred = np.array(pred_mask).astype(bool).flatten()
    y_true = np.array(true_mask).astype(bool).flatten()
    return {
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'iou': jaccard_score(y_true, y_pred)
    }

def dummy_segmentation(img):
    # replace with real segmentation
    gray = img.convert('L')
    return gray.point(lambda p: p > 128 and 255)

def run_evaluation(image_dir, mask_dir, sr_model_path, n=5):
    sess = load_onnx_model(sr_model_path)
    tf = transforms.ToTensor()

    files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))][:n]

    for fname in files:
        img_path = os.path.join(image_dir, fname)
        mask_path = os.path.join(mask_dir, os.path.splitext(fname)[0] + '.png')
        if not os.path.exists(mask_path):
            print(f"⚠️ Mask missing for {fname}, skipping")
            continue

        img = Image.open(img_path).convert('RGB')
        y, cb, cr = preprocess_ycbcr(img)

        y_tensor = tf(y)
        cb_tensor = tf(cb)
        cr_tensor = tf(cr)

        y_sr = run_sr(sess, y_tensor)
        sr_img = merge_ycbcr(y_sr, cb_tensor, cr_tensor)

        psnr, ssim = evaluate_image(sr_img, img)
        pred_mask = dummy_segmentation(sr_img)
        true_mask = Image.open(mask_path).resize(pred_mask.size).convert('1')
        metrics = evaluate_masks(pred_mask, true_mask)

        print(f"\n📁 {fname}")
        print(f"  PSNR: {psnr:.2f}, SSIM: {ssim:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}, IoU: {metrics['iou']:.4f}")

if __name__ == '__main__':
    run_evaluation(
        image_dir='/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/data/cracks/crack_segmentation_dataset/train/images',
        mask_dir='/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/data/cracks/crack_segmentation_dataset/train/masks',
        sr_model_path='models/sr_block_2x.onnx',
        n=10  # evaluate 10 images
    )