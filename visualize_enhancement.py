import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from WaveMixSR import WaveMixSR_V2

def load_model(weights_path, device):
    model = WaveMixSR_V2(
        sr=2, blocks=2, mult=1,
        final_dim=144, ff_channel=144, dropout=0.3
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

def enhance(img, model, device):
    # YCbCr SR: only upscale the Y channel
    ycbcr = img.convert('YCbCr').split()
    y = transforms.ToTensor()(ycbcr[0]).unsqueeze(0).to(device)
    with torch.no_grad():
        sr_y = model(y)
    cb = transforms.ToTensor()(ycbcr[1]).unsqueeze(0).to(device)
    cr = transforms.ToTensor()(ycbcr[2]).unsqueeze(0).to(device)
    cb_up = torch.nn.functional.interpolate(cb, scale_factor=2, mode='bicubic')
    cr_up = torch.nn.functional.interpolate(cr, scale_factor=2, mode='bicubic')
    # merge channels & return
    import numpy as np
    y_np = sr_y.squeeze().cpu().clamp(0,1).numpy()
    cb_np = cb_up.squeeze().cpu().numpy()
    cr_np = cr_up.squeeze().cpu().numpy()
    up = np.stack([y_np, cb_np, cr_np], axis=2)
    return Image.fromarray((up*255).astype('uint8'), 'YCbCr').convert('RGB')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model('models/bsd100_2x_y_df2k_33.2.pth', device)
    src = 'data/CRACK500/images'
    files = [f for f in os.listdir(src) if f.endswith('.jpg')][:4]

    fig, axes = plt.subplots(2,4, figsize=(16,8))
    for col, fname in enumerate(files):
        img = Image.open(os.path.join(src,fname)).convert('RGB')
        img = img.resize((256,256))
        sr = enhance(img, model, device)
        axes[0,col].imshow(img);   axes[0,col].set_title('Original')
        axes[1,col].imshow(sr);    axes[1,col].set_title('SR Enhanced')
        for ax in axes[:,col]: ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    main()
