"""Torch-free YOLO26 inference with calibrated confidence and conformal box intervals (NumPy + ONNX Runtime).

Mirrors the Ultralytics pipeline used in training/evaluation: letterbox to a square input, raw one-to-many head
output (cx, cy, w, h, class scores), class-aware NMS, then the per-class Platt calibration and conformal margins
fitted on the held-out calibration split.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

DESCRIPTIONS = {
    "Impacted": "Tooth that has not erupted into its normal position (often a wisdom tooth).",
    "Caries": "Tooth decay limited to the outer layers of the tooth.",
    "Deep Caries": "Decay reaching close to the pulp; typically needs root-canal treatment or extraction.",
    "Periapical Lesion": "Radiolucent area at the root tip, usually a sign of infection or inflammation.",
}
EPS = 1e-6


def letterbox(image: Image.Image, size: int, stride: int = 32):
    """Match Ultralytics LetterBox(auto=True) used at predict time: resize the long side to ``size`` with
    OpenCV linear interpolation and pad only up to a multiple of ``stride`` (rectangular input), gray 114.
    Calibration was fitted on outputs produced this way, so the preprocessing must be identical."""
    rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (size - nw) % stride / 2, (size - nh) % stride / 2
    if (nw, nh) != (w, h):
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(rgb, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    tensor = np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    return tensor, r, left, top


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    order = scores.argsort()[::-1]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou < iou_thr]
    return np.array(keep, dtype=int)


def approx_region(box, width, height):
    """Approximate FDI quadrant from box position (image left = patient's right)."""
    cx, cy = (box[0] + box[2]) / 2 / width, (box[1] + box[3]) / 2 / height
    upper, patient_right = cy < 0.5, cx < 0.5
    quadrant = {(True, True): 1, (True, False): 2, (False, False): 3, (False, True): 4}[(upper, patient_right)]
    return f"Q{quadrant} ({'upper' if upper else 'lower'} {'right' if patient_right else 'left'})"


def status(conf):
    return "Confident" if conf >= 0.7 else "Probable" if conf >= 0.4 else "Uncertain - review"


class Detector:
    def __init__(self, model_dir: Path):
        model_dir = Path(model_dir)
        self.meta = json.loads((model_dir / "meta.json").read_text())
        self.names = {int(k): v for k, v in self.meta["names"].items()}
        self.imgsz = int(self.meta["imgsz"])
        self.session = ort.InferenceSession(str(model_dir / "dentex_yolo26s.onnx"), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def calibrate(self, scores, classes):
        a_g, b_g = self.meta["platt_global"]
        out = np.empty_like(scores)
        for i, (s, c) in enumerate(zip(scores, classes)):
            a, b = self.meta["platt"].get(str(int(c)), (a_g, b_g))
            p = min(max(float(s), EPS), 1 - EPS)
            out[i] = 1 / (1 + np.exp(-(a * np.log(p / (1 - p)) + b)))
        return out

    def detect(self, image: Image.Image, conf_floor=0.05, max_det=300):
        image = image.convert("RGB")
        W, H = image.size
        tensor, r, left, top = letterbox(image, self.imgsz)
        pred = self.session.run(None, {self.input_name: tensor})[0][0].T  # (N, 4 + nc)
        cls_scores = pred[:, 4:]
        classes = cls_scores.argmax(1)
        scores = cls_scores[np.arange(len(classes)), classes]
        m = scores >= conf_floor
        xywh, scores, classes = pred[m, :4], scores[m], classes[m]
        if not len(scores):
            return []
        xyxy = np.concatenate([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], axis=1)
        keep = nms(xyxy + classes[:, None] * 7680.0, scores, float(self.meta["iou_nms"]))[:max_det]  # class-aware
        xyxy, scores, classes = xyxy[keep], scores[keep], classes[keep]
        xyxy = (xyxy - np.array([left, top, left, top])) / r
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, W)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, H)
        calibrated = self.calibrate(scores, classes)

        findings = []
        for b, s, cal, c in sorted(zip(xyxy, scores, calibrated, classes), key=lambda t: -t[2]):
            name = self.names[int(c)]
            q = float(self.meta["margins"].get(str(int(c)), float("inf")))
            bw, bh = b[2] - b[0], b[3] - b[1]
            outer = [b[0] - q * bw, b[1] - q * bh, b[2] + q * bw, b[3] + q * bh] if np.isfinite(q) else None
            findings.append({
                "class_id": int(c), "name": name, "description": DESCRIPTIONS.get(name, ""),
                "confidence": round(float(cal), 4), "raw_confidence": round(float(s), 4),
                "status": status(float(cal)), "region": approx_region(b, W, H),
                "box": [round(float(v), 1) for v in b],
                "interval_outer": [round(float(v), 1) for v in outer] if outer else None,
                "interval_margin": round(q, 4) if np.isfinite(q) else None,
            })
        return findings
