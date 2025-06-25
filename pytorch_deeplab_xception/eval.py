import numpy as np
import os
import logging
import glob
import cv2
import numpy as np

def cal_global_acc(pred, gt):
    h,w = gt.shape
    return [np.sum(pred==gt), float(h*w)]

def get_statistics_seg(pred, gt, num_cls=2):
    h,w = gt.shape
    statistics = []
    for i in range(num_cls):
        tp = np.sum((pred==i)&(gt==i))
        fp = np.sum((pred==i)&(gt!=i))
        fn = np.sum((pred!=i)&(gt==i))
        statistics.append([tp, fp, fn])
    return statistics

def get_statistics_prf(pred, gt):
    tp = np.sum((pred==1)&(gt==1))
    fp = np.sum((pred==1)&(gt==0))
    fn = np.sum((pred==0)&(gt==1))
    return [tp, fp, fn]

def segment_metrics(pred_list, gt_list, num_cls = 2):
    global_accuracy_cur = []
    statistics = []

    for pred, gt in zip(pred_list, gt_list):
        gt_img = (gt / 255).astype('uint8')
        pred_img = (pred / 255).astype('uint8')
        global_accuracy_cur.append(cal_global_acc(pred_img, gt_img))
        statistics.append(get_statistics_seg(pred_img, gt_img, num_cls))


    global_acc = np.sum([v[0] for v in global_accuracy_cur]) / np.sum([v[1] for v in global_accuracy_cur])
    counts = []
    for i in range(num_cls):
        tp = np.sum([v[i][0] for v in statistics])
        fp = np.sum([v[i][1] for v in statistics])
        fn = np.sum([v[i][2] for v in statistics])

        counts.append([tp, fp, fn])

    mean_acc = np.sum([v[0] / (v[0] + v[2]) for v in counts]) / num_cls
    mean_iou_acc = np.sum([v[0] / (np.sum(v)) for v in counts]) / num_cls

    return global_acc, mean_acc, mean_iou_acc

def prf_metrics(pred_list, gt_list):
    statistics = []

    for pred, gt in zip(pred_list, gt_list):
        gt_img = (gt / 255).astype('uint8')
        pred_img = (((pred / np.max(pred))>0.5)).astype('uint8')
        statistics.append(get_statistics_prf(pred_img, gt_img))

    tp = np.sum([v[0] for v in statistics])
    fp = np.sum([v[1] for v in statistics])
    fn = np.sum([v[2] for v in statistics])
    print("tp:{}, fp:{}, fn:{}".format(tp,fp,fn))
    p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
    r_acc = tp / (tp + fn)
    f_acc = 2 * p_acc * r_acc / (p_acc + r_acc)
    return p_acc,r_acc,f_acc


def cal_prf_metrics(pred_list, gt_list, thresh_step=0.01):
    final_accuracy_all = []
    for thresh in np.arange(0.0, 1.0, thresh_step):
        statistics = []
        for pred, gt in zip(pred_list, gt_list):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            statistics.append(get_statistics(pred_img, gt_img))
        tp = np.sum([v[0] for v in statistics])
        fp = np.sum([v[1] for v in statistics])
        fn = np.sum([v[2] for v in statistics])

        p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
        #r_acc = tp / (tp + fn)
        r_acc = tp / (tp + fn) if (tp + fn) != 0 else 0.0
        final_accuracy_all.append([thresh, p_acc, r_acc, 2 * p_acc * r_acc / (p_acc + r_acc)])

    return final_accuracy_all

def thred_half(src_img_list, tgt_img_list):
    Precision, Recall, F_score = prf_metrics(src_img_list, tgt_img_list)
    Global_Accuracy, Class_Average_Accuracy, Mean_IOU = segment_metrics(src_img_list, tgt_img_list)
    print("Global Accuracy:{}, Class Average Accuracy:{}, Mean IOU:{}, Precision:{}, Recall:{}, F score:{}".format(
        Global_Accuracy, Class_Average_Accuracy, Mean_IOU, Precision, Recall, F_score))

def get_statistics(pred, gt):
    tp = np.sum((pred==1)&(gt==1))
    fp = np.sum((pred==1)&(gt==0))
    fn = np.sum((pred==0)&(gt==1))
    return [tp, fp, fn]

def cal_OIS_metrics(pred_list, gt_list, thresh_step=0.01):
    final_F1_list = []
    for pred, gt in zip(pred_list, gt_list):
        p_acc_list = []
        r_acc_list = []
        F1_list = []
        for thresh in np.arange(0.0, 1.0, thresh_step):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            tp, fp, fn = get_statistics(pred_img, gt_img)
            p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
            if tp + fn == 0:
                r_acc=0
            else:
                r_acc = tp / (tp + fn)
            if p_acc + r_acc==0:
                F1 = 0
            else:
                F1 = 2 * p_acc * r_acc / (p_acc + r_acc)

            p_acc_list.append(p_acc)
            r_acc_list.append(r_acc)
            F1_list.append(F1)

        assert len(p_acc_list)==100, "p_acc_list is not 100"
        assert len(r_acc_list)==100, "r_acc_list is not 100"
        assert len(F1_list)==100, "F1_list is not 100"

        max_F1 = np.max(np.array(F1_list))
        final_F1_list.append(max_F1)

    final_F1 = np.sum(np.array(final_F1_list))/len(final_F1_list)
    return final_F1

def cal_ODS_metrics(pred_list, gt_list, thresh_step=0.01):
    save_data = {
        "ODS": [],
    }
    final_ODS = []
    for thresh in np.arange(0.0, 1.0, thresh_step):
        ODS_list = []
        for pred, gt in zip(pred_list, gt_list):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            tp, fp, fn = get_statistics(pred_img, gt_img)
            # calculate precision
            p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
            if tp + fn == 0:
                r_acc=0
            else:
                r_acc = tp / (tp + fn)
            if p_acc + r_acc==0:
                F1 = 0
            else:
                F1 = 2 * p_acc * r_acc / (p_acc + r_acc)
            ODS_list.append(F1)

        ave_F1 = np.mean(np.array(ODS_list))
        final_ODS.append(ave_F1)
    ODS = np.max(np.array(final_ODS))
    return ODS

def cal_mIoU_metrics(pred_list, gt_list, thresh_step=0.01, pred_imgs_names=None, gt_imgs_names=None):
    final_iou = []
    for thresh in np.arange(0.0, 1.0, thresh_step):
        iou_list = []
        for i, (pred, gt) in enumerate(zip(pred_list, gt_list)):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            TP = np.sum((pred_img == 1) & (gt_img == 1))
            TN = np.sum((pred_img == 0) & (gt_img == 0))
            FP = np.sum((pred_img == 1) & (gt_img == 0))
            FN = np.sum((pred_img == 0) & (gt_img == 1))
            if (FN + FP + TP) <= 0:
                iou = 0
            else:
                iou_1 = TP / (FN + FP + TP)
                iou_0 = TN / (FN + FP + TN)
                iou = (iou_1 + iou_0)/2
            iou_list.append(iou)
        ave_iou = np.mean(np.array(iou_list))
        final_iou.append(ave_iou)
    mIoU = np.max(np.array(final_iou))
    return mIoU

def imread(path, load_size=0, load_mode=cv2.IMREAD_GRAYSCALE, convert_rgb=False, thresh=-1):
    im = cv2.imread(path, load_mode)
    if convert_rgb:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if load_size > 0:
        im = cv2.resize(im, (load_size, load_size), interpolation=cv2.INTER_CUBIC)
    if thresh > 0:
        _, im = cv2.threshold(im, thresh, 255, cv2.THRESH_BINARY)
    return im

def get_image_pairs(data_dir, suffix_gt='real_B', suffix_pred='fake_B'):
    gt_list = glob.glob(os.path.join(data_dir, '*{}.png'.format(suffix_gt)))
    pred_list = [ll.replace(suffix_gt, suffix_pred) for ll in gt_list]
    assert len(gt_list) == len(pred_list)
    pred_imgs, gt_imgs = [], []
    pred_imgs_names, gt_imgs_names = [], []
    for pred_path, gt_path in zip(pred_list, gt_list):
        pred_imgs.append(imread(pred_path))
        gt_imgs.append(imread(gt_path, thresh=127))
        pred_imgs_names.append(pred_path)
        gt_imgs_names.append(gt_path)
    return pred_imgs, gt_imgs, pred_imgs_names, gt_imgs_names

def eval(log_eval, results_dir, epoch):

    suffix_gt = "lab"
    suffix_pred = "pre"
    log_eval.info(results_dir)
    log_eval.info("checkpoints -> " + results_dir)
    src_img_list, tgt_img_list, pred_imgs_names, gt_imgs_names = get_image_pairs(results_dir, suffix_gt, suffix_pred)
    assert len(src_img_list) == len(tgt_img_list)
    final_accuracy_all = cal_prf_metrics(src_img_list, tgt_img_list)
    final_accuracy_all = np.array(final_accuracy_all)
    Precision_list, Recall_list, F_list = final_accuracy_all[:, 1], final_accuracy_all[:,2], final_accuracy_all[:, 3]
    mIoU = cal_mIoU_metrics(src_img_list, tgt_img_list, pred_imgs_names=pred_imgs_names, gt_imgs_names=gt_imgs_names)
    ODS = cal_ODS_metrics(src_img_list, tgt_img_list)
    OIS = cal_OIS_metrics(src_img_list, tgt_img_list)
    log_eval.info("mIouU -> " + str(mIoU))
    log_eval.info("ODS -> " + str(ODS))
    log_eval.info("OIS -> " + str(OIS))
    log_eval.info("F1 -> " + str(F_list[0]))
    log_eval.info("P -> " + str(Precision_list[0]))
    log_eval.info("R -> " + str(Recall_list[0]))
    log_eval.info("eval finish!")

    return {'epoch': epoch, 'mIoU': mIoU, 'ODS': ODS, 'OIS': OIS, 'F1': F_list[0], 'Precision': Precision_list[0], 'Recall': Recall_list[0]}

def cal_prf_metrics_fast(pred_list, gt_list, thresholds=[0.5]):
    """
    Fast PRF evaluator. Accepts list of thresholds (e.g., [0.5] or np.linspace(0,1,100)).
    Returns a list of [threshold, P, R, F1] rows.
    """
    thresholds = np.array(thresholds)
    results = []

    for thresh in thresholds:
        tp = fp = fn = 0
        for pred, gt in zip(pred_list, gt_list):
            pred_bin = ((pred / 255.0) > thresh).astype(np.uint8)
            gt_bin   = ((gt / 255.0) > 0.5).astype(np.uint8)

            tp += np.count_nonzero((pred_bin == 1) & (gt_bin == 1))
            fp += np.count_nonzero((pred_bin == 1) & (gt_bin == 0))
            fn += np.count_nonzero((pred_bin == 0) & (gt_bin == 1))

        prec = 1.0 if (tp + fp) == 0 else tp / (tp + fp)
        recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
        f1 = 0.0 if (prec + recall) == 0 else 2 * prec * recall / (prec + recall)
        results.append([thresh, prec, recall, f1])

    return np.array(results)

def cal_mIoU_metrics_fast(pred_list, gt_list, thresh_step=0.01):
    thresholds = np.arange(0.0, 1.0, thresh_step)
    preds = [p.astype(np.float32) / 255.0 for p in pred_list]
    gts   = [g.astype(np.uint8) // 255 for g in gt_list]  # assumes 0 or 255

    final_iou = []

    for thresh in thresholds:
        ious = []
        for p, g in zip(preds, gts):
            p_bin = (p > thresh).astype(np.uint8)

            tp = np.sum((p_bin == 1) & (g == 1))
            tn = np.sum((p_bin == 0) & (g == 0))
            fp = np.sum((p_bin == 1) & (g == 0))
            fn = np.sum((p_bin == 0) & (g == 1))

            iou_1 = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            iou_0 = tn / (tn + fp + fn) if (tn + fp + fn) > 0 else 0.0
            iou = (iou_1 + iou_0) / 2
            ious.append(iou)

        final_iou.append(np.mean(ious))

    return np.max(final_iou)


# 🧠 Utility: evaluate in-memory
def eval_from_memory(pred_list, gt_list):
    thresholds = [.5]
    final_accuracy_all = cal_prf_metrics_fast(pred_list, gt_list, thresholds)

    idx_05 = np.argmin(np.abs(final_accuracy_all[:, 0] - 0.5))
    precision = final_accuracy_all[idx_05, 1]
    recall    = final_accuracy_all[idx_05, 2]
    f1        = final_accuracy_all[idx_05, 3]

    #mIoU = cal_mIoU_metrics(pred_list, gt_list)
    mIoU = cal_mIoU_metrics_fast(pred_list, gt_list)
    #ODS = cal_ODS_metrics(pred_list, gt_list)
    #OIS = cal_OIS_metrics(pred_list, gt_list)

    return {
        "mIoU": mIoU,
        #"ODS": ODS,
        #"OIS": OIS,
        "F1": f1,
        "Precision": precision,
        "Recall": recall
    }

    
'''def eval_from_memory(pred_list, gt_list):
    assert len(pred_list) == len(gt_list), "Mismatched prediction and ground truth counts"

    # Precision, Recall, F1 at 0.5 threshold
    final_accuracy_all = cal_prf_metrics(pred_list, gt_list)
    final_accuracy_all = np.array(final_accuracy_all)
    
    thresh_col = final_accuracy_all[:, 0]
    idx_05 = np.argmin(np.abs(thresh_col - 0.5))

    precision = final_accuracy_all[idx_05, 1]
    recall    = final_accuracy_all[idx_05, 2]
    f1        = final_accuracy_all[idx_05, 3]
    Precision_list = final_accuracy_all[:, 1]
    Recall_list = final_accuracy_all[:, 2]
    F1_list = final_accuracy_all[:, 3]

    # mIoU, ODS, OIS
    mIoU = cal_mIoU_metrics(pred_list, gt_list)
    #ODS = cal_ODS_metrics(pred_list, gt_list)
    #OIS = cal_OIS_metrics(pred_list, gt_list)
    

    return {
        "mIoU": mIoU,
        #"ODS": ODS,
        #"OIS": OIS,
        "F1": F1_list[0],
        "Precision": Precision_list[0],
        "Recall": Recall_list[0]
    }'''
    
'''
def debug_compare_eval(pred_list, gt_list, max_samples=5):
    """
    Compare the original (slow but trusted) evaluation path with the new one.
    """
    from copy import deepcopy

    print("🧪 Debugging eval_from_memory vs cal_* functions...")
    
    # Clone data for both paths
    pred_list_a = deepcopy(pred_list)
    gt_list_a   = deepcopy(gt_list)

    pred_list_b = deepcopy(pred_list)
    gt_list_b   = deepcopy(gt_list)

    # Run old method (via cal_* manually)
    print("\n▶ Running original eval path...")
    accs = cal_prf_metrics(pred_list_a, gt_list_a)
    iou  = cal_mIoU_metrics(pred_list_a, gt_list_a)
    ods  = cal_ODS_metrics(pred_list_a, gt_list_a)
    ois  = cal_OIS_metrics(pred_list_a, gt_list_a)
    p, r, f = accs[0][1], accs[0][2], accs[0][3]
    print(p, r, f)
    accs = np.array(accs)
    idx_05 = np.argmin(np.abs(accs[:, 0] - 0.5))
    p, r, f = accs[idx_05][1], accs[idx_05][2], accs[idx_05][3]
    print(p, r, f)
    print(f"[OLD] P={p:.4f}, R={r:.4f}, F1={f:.4f}, mIoU={iou:.4f}, ODS={ods:.4f}, OIS={ois:.4f}")

    # Run new method
    print("\n▶ Running new eval_from_memory path...")
    print("▶ unique pred[0]:", np.unique(pred_list[0]))
    print("▶ unique gt[0]:", np.unique(gt_list[0]))
    new_metrics = eval_from_memory(pred_list_b, gt_list_b)
    for k, v in new_metrics.items():
        print(f"[NEW] {k}: {v:.4f}")

    # Check first few masks for exact per-image stats
    print("\n▶ Checking per-image TP/FP/FN for first few samples:")
    for i, (pa, ga) in enumerate(zip(pred_list, gt_list)):
        if i >= max_samples: break

        pa_bin = ((pa / 255.0) > 0.5).astype(np.uint8)
        ga_bin = ((ga / 255.0) > 0.5).astype(np.uint8)

        if pa_bin.shape != ga_bin.shape:
            ga_bin = cv2.resize(ga_bin, (pa_bin.shape[1], pa_bin.shape[0]), interpolation=cv2.INTER_NEAREST)

        tp, fp, fn = get_statistics(pa_bin, ga_bin)
        print(f"Image {i:02d} → TP: {tp}, FP: {fp}, FN: {fn}")

    print("\n✅ Debugging complete.")'''

# at the bottom of eval.py

# … remove old scale_mask & evaluate_multiscale …

def scale_mask(mask: np.ndarray, factor: float) -> np.ndarray:
    """
    Resize a mask (2D or logits) by `factor` using bicubic.
    Ensures output is uint8 H×W.
    """
    if mask is None:
        raise ValueError("scale_mask: input mask is None")

    # handle logits or extra channels
    if mask.ndim == 3:
        # (1,H,W)
        if mask.shape[0] == 1:
            mask = mask[0]
        # (H,W,1)
        elif mask.shape[-1] == 1:
            mask = mask[...,0]
        # (C,H,W) → take argmax
        else:
            mask = np.argmax(mask, axis=0).astype('uint8') * 255
    elif mask.ndim != 2:
        raise ValueError(f"scale_mask: expected 2D or 3D mask, got {mask.shape}")

    # ensure uint8 for OpenCV
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    h, w = mask.shape
    new_h, new_w = int(round(h*factor)), int(round(w*factor))
    if new_h < 1 or new_w < 1:
        raise ValueError(f"scale_mask: invalid new size {new_h}×{new_w}")

    return cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def evaluate_multiscale(preds_orig: list, preds_2x: list, preds_4x: list, gts: list) -> dict:
    """
    preds_orig @ original resolution
    preds_2x   @ 0.5× resolution  (must upsample ×2)
    preds_4x   @ 4× resolution    (must downsample ¼×)
    gts        @ original resolution

    Returns dict with mIoU/F1 for each plus ratio_2x & ratio_4x.
    """
    # 1×
    from time import time
    total_time = 0.0
    start_time = time()
    m0 = eval_from_memory(preds_orig, gts)
    total_time += time() - start_time
    

    # 0.5× → upsample by 2 back to original
    up2 = [scale_mask(p, .5) for p in preds_2x]
    start_time = time()
    m2 = eval_from_memory(up2, gts)
    total_time += time() - start_time

    # 4× → downsample by 0.25 back to original
    dn4 = [scale_mask(p, 0.25) for p in preds_4x]
    start_time = time()
    m4 = eval_from_memory(dn4, gts)
    total_time += time() - start_time
    print(total_time)

    '''return {
      'mIoU_orig': m0['mIoU'],
      'F1_orig':   m0['F1'],
      'mIoU_2x':   m2['mIoU'],
      'F1_2x':     m2['F1'],
      'mIoU_4x':   m4['mIoU'],
      'F1_4x':     m4['F1'],
      'ratio_2x':  (m2['mIoU']/m0['mIoU']) if m0['mIoU'] else float('nan'),
      'ratio_4x':  (m4['mIoU']/m0['mIoU']) if m0['mIoU'] else float('nan'),
    }'''
    return {
        'mIoU_orig': m0['mIoU'],     'F1_orig': m0['F1'],     'Precision_orig': m0['Precision'], 'Recall_orig': m0['Recall'],
        'mIoU_2x':   m2['mIoU'],     'F1_2x':   m2['F1'],     'Precision_2x':   m2['Precision'], 'Recall_2x':   m2['Recall'],
        'mIoU_4x':   m4['mIoU'],     'F1_4x':   m4['F1'],     'Precision_4x':   m4['Precision'], 'Recall_4x':   m4['Recall'],
        'ratio_2x':  (m2['mIoU'] / m0['mIoU']) if m0['mIoU'] else float('nan'),
        'ratio_4x':  (m4['mIoU'] / m0['mIoU']) if m0['mIoU'] else float('nan'),
    }

if __name__ == '__main__':
    suffix_gt = "lab"
    suffix_pred = "pre"
    results_dir = "../results/results_test/TUT_results"
    logging.info(results_dir)
    src_img_list, tgt_img_list, pred_imgs_names, gt_imgs_names = get_image_pairs(results_dir, suffix_gt, suffix_pred)
    assert len(src_img_list) == len(tgt_img_list)
    final_accuracy_all = cal_prf_metrics(src_img_list, tgt_img_list)
    final_accuracy_all = np.array(final_accuracy_all)
    Precision_list, Recall_list, F_list = final_accuracy_all[:,1], final_accuracy_all[:,2], final_accuracy_all[:,3]
    mIoU = cal_mIoU_metrics(src_img_list,tgt_img_list, pred_imgs_names=pred_imgs_names, gt_imgs_names=gt_imgs_names)
    ODS = cal_ODS_metrics(src_img_list, tgt_img_list)
    OIS = cal_OIS_metrics(src_img_list, tgt_img_list)
    print("mIouU -> " + str(mIoU))
    print("ODS -> " + str(ODS))
    print("OIS -> " + str(OIS))
    print("F1 -> " + str(F_list[0]))
    print("P -> " + str(Precision_list[0]))
    print("R -> " + str(Recall_list[0]))
    print("eval finish!")