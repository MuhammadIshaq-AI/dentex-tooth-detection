"""Box utilities and prediction-to-ground-truth matching (boxes are xyxy)."""
import numpy as np


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N, 4) and (M, 4) xyxy boxes -> (N, M)."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).clip(0).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).clip(0).prod(-1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match_detections(pred_boxes, pred_scores, pred_cls, gt_boxes, gt_cls, iou_thr=0.5):
    """Greedy, score-ordered, class-aware matching (COCO style).

    Returns:
        pred_to_gt: (N,) index of the matched GT box, or -1 for a false positive.
        gt_to_pred: (M,) index of the matched prediction, or -1 for a missed object.
    """
    pred_boxes = np.asarray(pred_boxes, dtype=float).reshape(-1, 4)
    gt_boxes = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
    pred_cls = np.asarray(pred_cls).reshape(-1)
    gt_cls = np.asarray(gt_cls).reshape(-1)
    pred_to_gt = np.full(len(pred_boxes), -1, dtype=int)
    gt_to_pred = np.full(len(gt_boxes), -1, dtype=int)
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return pred_to_gt, gt_to_pred

    iou = box_iou(pred_boxes, gt_boxes)
    iou[pred_cls[:, None] != gt_cls[None, :]] = 0.0
    for p in np.argsort(-np.asarray(pred_scores, dtype=float)):
        cand = np.where(gt_to_pred < 0, iou[p], -1.0)
        g = int(np.argmax(cand))
        if cand[g] >= iou_thr:
            pred_to_gt[p] = g
            gt_to_pred[g] = p
    return pred_to_gt, gt_to_pred


def average_precision(scores, is_tp, n_gt: int) -> float:
    """All-point interpolated AP (COCO/VOC2010+ style)."""
    if n_gt == 0:
        return float("nan")
    order = np.argsort(-np.asarray(scores, dtype=float))
    tp = np.asarray(is_tp, dtype=float)[order]
    if len(tp) == 0:
        return 0.0
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = np.concatenate([[0], ctp / n_gt, [1]])
    precision = np.concatenate([[1], ctp / np.maximum(ctp + cfp, 1e-9), [0]])
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    idx = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[idx + 1] - recall[idx]) * precision[idx + 1]))


def per_class_ap(preds, gts, num_classes: int, iou_thr=0.5) -> np.ndarray:
    """AP per class over a dataset. ``preds``/``gts`` are lists of objects with xyxy, cls (, conf)."""
    scores = [[] for _ in range(num_classes)]
    tps = [[] for _ in range(num_classes)]
    n_gt = np.zeros(num_classes, int)
    for p, g in zip(preds, gts):
        p2g, _ = match_detections(p.xyxy, p.conf, p.cls, g.xyxy, g.cls, iou_thr)
        for c in range(num_classes):
            m = p.cls == c
            scores[c].extend(p.conf[m])
            tps[c].extend(p2g[m] >= 0)
            n_gt[c] += int((g.cls == c).sum())
    return np.array([average_precision(scores[c], tps[c], n_gt[c]) for c in range(num_classes)])
