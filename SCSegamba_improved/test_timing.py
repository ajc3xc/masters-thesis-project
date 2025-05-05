'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org
'''

import numpy as np
import torch
import argparse
import os
import cv2
from datasets import create_dataset
from models import build_model
from main import get_args_parser
import time

#from torch.cuda.amp import autocast

from eval.evaluate import eval_from_memory

parser = argparse.ArgumentParser('SCSEGAMBA FOR CRACK', parents=[get_args_parser()])
args = parser.parse_args()
args.phase = 'test'
#args.dataset_path = '/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/SCSegamba/data/TUT'
args.dataset_path = '/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/data/crack_segmentation_unzipped/crack_segmentation/virginia_tech_concrete_crack_congolmeration/Conglomerate Concrete Crack Detection/Conglomerate Concrete Crack Detection/Test'


if __name__ == '__main__':
    args.batch_size = 10
    args.attention_type = 'gbc'
    args.fusion_mode = 'original'
    t_all = []
    device = torch.device(args.device)
    print(device)
    test_dl = create_dataset(args)
    #load_model_file = "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/SCSegamba/checkpoint_TUT.pth"
    load_model_file = "/mnt/stor/ceph/gchen-lab/data/Adam/masters-thesis-project/SCSegamba/checkpoints/TUT/checkpoint_best.pth"
    data_size = len(test_dl)
    model, criterion = build_model(args)
    #state_dict = torch.load(load_model_file)
    import argparse
    #from torch.serialization import add_safe_globals

    #add_safe_globals([argparse.Namespace])
    state_dict = torch.load(load_model_file, weights_only=False)
    model.load_state_dict(state_dict["model"], strict=False)
    #model = torch.compile(model)
    torch.backends.cudnn.benchmark = True
    model.to(device)
    print("Load Model Successful!")
    suffix = load_model_file.split('/')[-2]
    save_root = "./results/results_test/" + suffix
    if not os.path.isdir(save_root):
        os.makedirs(save_root)
    
    max_iters = 100
    with torch.no_grad():
        model.eval()

        pred_list = []
        gt_list = []
        total_infer_time = 0.0
        torch.cuda.synchronize()
        for batch_idx, (data) in enumerate(test_dl):
            if max_iters is not None and batch_idx * args.batch_size >= max_iters:
                break
            print(f"{(batch_idx*args.batch_size)+1}-{(batch_idx+1)*args.batch_size}")
            x = data["image"]
            target = data["label"]
            #if device != 'cpu':
            #    print('using cuda')
            x, target = x.cuda(non_blocking=True), target.cuda(non_blocking=True).long()

            torch.cuda.synchronize()
            start = time.time()
            out = model(x)
            torch.cuda.synchronize()
            end = time.time()
            total_infer_time += end - start

            target = target[0, 0, ...].cpu().numpy()
            out = out[0, 0, ...].cpu().numpy()
            #root_name = data["A_paths"][0].split("/")[-1][0:-4]
            target = 255 * (target / np.max(target))
            out = 255 * (out / np.max(out))

            # out[out >= 0.5] = 255
            # out[out < 0.5] = 0

            pred_list.append(out)
            gt_list.append(target)

        fps = len(test_dl) / total_infer_time
        metrics = eval_from_memory(pred_list, gt_list)
        metrics["FPS"] = fps
        #print(f"🧪 Model: {model_name}")
        print(f'FPS: {fps}')
        print(f"  mIoU : {metrics['mIoU']:.4f}")
        print(f"  ODS  : {metrics['ODS']:.4f}")
        print(f"  OIS  : {metrics['OIS']:.4f}")
        print(f"  F1   : {metrics['F1']:.4f}")
        print(f"  Precision: {metrics['Precision']:.4f}")
        print(f"  Recall   : {metrics['Recall']:.4f}")
        #print(f"  FPS  : {fps:.2f}\n")

            #print('----------------------------------------------------------------------------------------------')
            #print(os.path.join(save_root, "{}_lab.png".format(root_name)))
            #print(os.path.join(save_root, "{}_pre.png".format(root_name)))
            #print('----------------------------------------------------------------------------------------------')
            #cv2.imwrite(os.path.join(save_root, "{}_lab.png".format(root_name)), target)
            #cv2.imwrite(os.path.join(save_root, "{}_pre.png".format(root_name)), out)
    print("Finished!")
