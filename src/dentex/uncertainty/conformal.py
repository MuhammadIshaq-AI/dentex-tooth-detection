"""Split-conformal prediction and confidence calibration for object detection.

Three tools, all fitted on a held-out calibration split and evaluated on test:

1. Per-class Platt scaling of detection confidences (reported via ECE).
2. Conformal box margins: per-class margin ``q`` (in units of box width/height) so the
   ground-truth box lies within ``pred ± q·size`` for >= 1 - alpha of matched objects.
3. Recall-controlling score thresholds: per-class threshold so that >= 1 - alpha of
   ground-truth objects are detected (miss rate <= alpha).
"""
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


# --------------------------------------------------------------------------- quantiles
def conformal_quantile(scores, alpha: float) -> float:
    """Finite-sample-corrected (1 - alpha) upper quantile. Returns inf if n is too small."""
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float("inf") if n == 0 or k > n else float(s[k - 1])


def recall_threshold(gt_scores, alpha: float) -> float:
    """Largest threshold t with P(score_new >= t) >= 1 - alpha.

    ``gt_scores`` holds, for each calibration GT object, the confidence of the prediction
    matched to it (0 if the object was missed entirely). Keeping predictions with
    score >= t misses a new object with probability <= floor(alpha (n+1)) / (n+1) <= alpha.
    """
    s = np.sort(np.asarray(gt_scores, dtype=float))
    k = int(np.floor(alpha * (len(s) + 1)))
    return 0.0 if k < 1 else float(s[k - 1])


# --------------------------------------------------------------------------- calibration
def expected_calibration_error(conf, correct, n_bins: int = 15) -> float:
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if len(conf) == 0:
        return float("nan")
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, bins[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(ece)


def reliability_curve(conf, correct, n_bins: int = 10):
    """Return (bin_centers, accuracy, mean_confidence, counts)."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, bins[1:-1]), 0, n_bins - 1)
    acc, mconf, cnt = np.full(n_bins, np.nan), np.full(n_bins, np.nan), np.zeros(n_bins, int)
    for b in range(n_bins):
        m = idx == b
        cnt[b] = m.sum()
        if cnt[b]:
            acc[b], mconf[b] = correct[m].mean(), conf[m].mean()
    return (bins[:-1] + bins[1:]) / 2, acc, mconf, cnt


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


@dataclass
class PlattCalibrator:
    """Per-class Platt scaling ``sigmoid(a * logit(score) + b)``, global fallback for rare classes."""

    min_samples: int = 20
    params: dict = field(default_factory=dict)
    global_params: tuple = (1.0, 0.0)

    @staticmethod
    def _fit(scores, correct):
        y = np.asarray(correct, dtype=int)
        if len(np.unique(y)) < 2:
            return 1.0, 0.0
        lr = LogisticRegression(C=1e4, max_iter=1000).fit(_logit(scores).reshape(-1, 1), y)
        return float(lr.coef_[0, 0]), float(lr.intercept_[0])

    def fit(self, scores, correct, cls):
        scores, correct, cls = map(np.asarray, (scores, correct, cls))
        self.global_params = self._fit(scores, correct)
        for c in np.unique(cls):
            m = cls == c
            if m.sum() >= self.min_samples:
                self.params[int(c)] = self._fit(scores[m], correct[m])
        return self

    def transform(self, scores, cls):
        scores, cls = np.asarray(scores, dtype=float), np.asarray(cls)
        out = np.empty_like(scores)
        for c in np.unique(cls):
            m = cls == c
            a, b = self.params.get(int(c), self.global_params)
            out[m] = 1 / (1 + np.exp(-(a * _logit(scores[m]) + b)))
        return out


# --------------------------------------------------------------------------- box margins
def box_nonconformity(pred_boxes, gt_boxes) -> np.ndarray:
    """Max coordinate residual normalised by predicted box width (x) / height (y)."""
    p = np.asarray(pred_boxes, dtype=float).reshape(-1, 4)
    g = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
    w = np.maximum(p[:, 2] - p[:, 0], EPS)
    h = np.maximum(p[:, 3] - p[:, 1], EPS)
    scale = np.stack([w, h, w, h], axis=1)
    return (np.abs(p - g) / scale).max(axis=1)


def box_intervals(pred_boxes, margin) -> tuple[np.ndarray, np.ndarray]:
    """Inner and outer boxes for per-box margins (scalar or (N,)). Inner may be degenerate."""
    p = np.asarray(pred_boxes, dtype=float).reshape(-1, 4)
    m = np.broadcast_to(np.asarray(margin, dtype=float), (len(p),))
    w, h = p[:, 2] - p[:, 0], p[:, 3] - p[:, 1]
    d = np.stack([-m * w, -m * h, m * w, m * h], axis=1)
    return p - d, p + d  # inner (shrunk), outer (expanded)


@dataclass
class ConformalDetector:
    """Per-class conformal box margins and recall thresholds (class -1 = pooled fallback)."""

    alpha: float = 0.1
    min_samples: int = 10
    margins: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)

    def fit(self, matched_pred_boxes, matched_gt_boxes, matched_cls, gt_scores, gt_cls):
        """
        matched_*: prediction/GT pairs from calibration matching (for box margins).
        gt_scores / gt_cls: per calibration GT object, matched confidence (0 if missed) and class.
        """
        nc = box_nonconformity(matched_pred_boxes, matched_gt_boxes)
        matched_cls, gt_scores, gt_cls = map(np.asarray, (matched_cls, gt_scores, gt_cls))
        self.margins[-1] = conformal_quantile(nc, self.alpha)
        self.thresholds[-1] = recall_threshold(gt_scores, self.alpha)
        for c in np.unique(np.concatenate([matched_cls, gt_cls]).astype(int)):
            if (matched_cls == c).sum() >= self.min_samples:
                self.margins[int(c)] = conformal_quantile(nc[matched_cls == c], self.alpha)
            if (gt_cls == c).sum() >= self.min_samples:
                self.thresholds[int(c)] = recall_threshold(gt_scores[gt_cls == c], self.alpha)
        return self

    def margin(self, c: int) -> float:
        return self.margins.get(int(c), self.margins.get(-1, float("inf")))

    def threshold(self, c: int) -> float:
        return self.thresholds.get(int(c), self.thresholds.get(-1, 0.0))
