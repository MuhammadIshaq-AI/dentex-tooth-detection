"""Draw ground truth vs predictions (optionally with conformal box intervals and MC-dropout stats)."""
from pathlib import Path

import cv2
import numpy as np

from dentex.labels import Boxes
from dentex.uncertainty.conformal import box_intervals
from dentex.uncertainty.matching import match_detections

# BGR, colour-blind-friendly (Okabe-Ito)
CLASS_COLORS = [(0, 159, 230), (0, 114, 213), (167, 121, 204), (115, 158, 0), (66, 228, 240)]


def _label(img, text, x, y, color, scale, placed=None):
    th = max(1, int(scale * 2))
    (tw, tht), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    x, y = int(x), max(int(y), tht + base)
    if placed is not None:  # nudge down past labels already drawn so they never overlap
        while any(x < px2 and px1 < x + tw and y - tht - base < py2 and py1 < y for px1, py1, px2, py2 in placed):
            y += tht + base + 2
        placed.append((x, y - tht - base, x + tw, y))
    cv2.rectangle(img, (x, y - tht - base), (x + tw, y), color, -1)
    cv2.putText(img, text, (x, y - base), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), th, cv2.LINE_AA)


def draw(image, boxes: Boxes, names, title, margins=None, stats=None):
    img = image.copy()
    h, w = img.shape[:2]
    scale, lw = max(w / 2400, 0.6), max(2, w // 700)
    placed = []
    for i, (b, c) in enumerate(zip(boxes.xyxy, boxes.cls)):
        color = CLASS_COLORS[int(c) % len(CLASS_COLORS)]
        if margins is not None and np.isfinite(margins[i]):
            inner, outer = box_intervals(b[None], margins[i])
            cv2.rectangle(img, tuple(outer[0, :2].astype(int)), tuple(outer[0, 2:].astype(int)), color, 1, cv2.LINE_AA)
        cv2.rectangle(img, tuple(b[:2].astype(int)), tuple(b[2:].astype(int)), color, lw)
        text = names[int(c)]
        if boxes.conf is not None:
            text += f" {boxes.conf[i]:.2f}"
        if stats is not None:
            text += f" H={stats['entropy'][i]:.2f}"
        _label(img, text, b[0], b[1] - 4, color, scale, placed)
    _label(img, title, 10, 10 + 40 * scale, (40, 40, 40), scale * 1.4)
    return img


def side_by_side(image_path, gt: Boxes, pred: Boxes, names, out_path, conf_thr=0.25, margins_fn=None, stats=None):
    image = cv2.imread(str(image_path))
    keep = pred.conf >= conf_thr
    shown = Boxes(pred.xyxy[keep], pred.cls[keep], pred.conf[keep])
    st = {k: v[keep] for k, v in stats.items()} if stats is not None else None
    margins = np.array([margins_fn(c) for c in shown.cls]) if margins_fn else None
    top = draw(image, gt, names, "Ground truth")
    bottom = draw(image, shown, names, "Prediction" + (" + 90% conformal box" if margins_fn else ""), margins, st)
    canvas = np.vstack([top, np.full((12, image.shape[1], 3), 255, np.uint8), bottom])
    canvas = cv2.resize(canvas, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])


def image_f1(gt: Boxes, pred: Boxes, conf_thr=0.25):
    keep = pred.conf >= conf_thr
    p2g, _ = match_detections(pred.xyxy[keep], pred.conf[keep], pred.cls[keep], gt.xyxy, gt.cls)
    tp = int((p2g >= 0).sum())
    return 2 * tp / max(keep.sum() + len(gt.cls), 1)


def pick_examples(gts, preds, k_best=4, k_worst=2, conf_thr=0.25):
    """Indices of the best and worst images by per-image F1 (images with GT only)."""
    scored = [(image_f1(g, p, conf_thr), i) for i, (g, p) in enumerate(zip(gts, preds)) if len(g.cls)]
    scored.sort()
    worst = [i for _, i in scored[:k_worst]]
    best = [i for _, i in scored[::-1][:k_best]]
    return best + worst
