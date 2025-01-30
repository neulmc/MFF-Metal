import numpy as np

# output accurate; miou score
def compute_accurate(pred, gt):
    iou_fenzi = []
    iou_fenmu = []
    for class_i in range(3):
        pred_tmp = np.zeros(pred.shape)
        gt_tmp = np.zeros(gt.shape)
        pred_tmp[pred == (class_i + 1)] = 1
        gt_tmp[gt == (class_i + 1)] = 1
        intersection = np.logical_and(pred_tmp, gt_tmp)
        totals = np.logical_or(pred_tmp, gt_tmp)
        iou_fenzi.append(intersection.sum())
        iou_fenmu.append(totals.sum())
    return iou_fenzi, iou_fenmu