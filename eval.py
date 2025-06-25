import numpy as np
import os
import logging
import glob
import cv2
import numpy as np

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
        "F1": f1,
        "Precision": precision,
        "Recall": recall
    }

# at the bottom of eval.py

# … remove old scale_mask & evaluate_multiscale …