import numpy as np

from dentex.uncertainty.conformal import (
    ConformalDetector,
    PlattCalibrator,
    box_intervals,
    box_nonconformity,
    conformal_quantile,
    expected_calibration_error,
    recall_threshold,
)
from dentex.uncertainty.matching import box_iou, match_detections


def test_conformal_quantile_coverage():
    rng = np.random.default_rng(0)
    alpha, covs = 0.1, []
    for _ in range(200):
        s = rng.exponential(size=301)
        q = conformal_quantile(s[:300], alpha)
        covs.append(s[300] <= q)
    assert np.mean(covs) >= 1 - alpha - 0.05
    assert conformal_quantile([1.0, 2.0], 0.1) == float("inf")


def test_recall_threshold_controls_misses():
    rng = np.random.default_rng(1)
    alpha, misses = 0.1, []
    for _ in range(300):
        s = rng.beta(5, 2, size=201)
        t = recall_threshold(s[:200], alpha)
        misses.append(s[200] < t)
    assert np.mean(misses) <= alpha + 0.04
    assert recall_threshold([0.9, 0.8], 0.1) == 0.0


def test_box_margins_cover_gt():
    rng = np.random.default_rng(2)
    gt = rng.uniform(0, 500, size=(2000, 2))
    gt = np.hstack([gt, gt + rng.uniform(20, 80, size=(2000, 2))])
    pred = gt + rng.normal(0, 4, size=gt.shape)
    q = conformal_quantile(box_nonconformity(pred[:1000], gt[:1000]), 0.1)
    inner, outer = box_intervals(pred[1000:], q)
    g = gt[1000:]
    covered = (
        (outer[:, :2] <= g[:, :2]).all(1) & (g[:, :2] <= inner[:, :2]).all(1)
        & (inner[:, 2:] <= g[:, 2:]).all(1) & (g[:, 2:] <= outer[:, 2:]).all(1)
    )
    assert covered.mean() >= 0.87


def test_platt_reduces_ece():
    rng = np.random.default_rng(3)
    true_p = rng.uniform(0, 1, 4000)
    correct = rng.uniform(size=4000) < true_p
    overconf = np.clip(true_p ** 0.3, 0, 1)  # systematically overconfident scores
    cls = rng.integers(0, 2, 4000)
    cal = PlattCalibrator().fit(overconf[:2000], correct[:2000], cls[:2000])
    before = expected_calibration_error(overconf[2000:], correct[2000:])
    after = expected_calibration_error(cal.transform(overconf[2000:], cls[2000:]), correct[2000:])
    assert after < before * 0.5


def test_conformal_detector_fallback():
    det = ConformalDetector(alpha=0.1, min_samples=5).fit(
        np.array([[0, 0, 10, 10]] * 3), np.array([[1, 1, 11, 11]] * 3), [0, 0, 0],
        gt_scores=[0.5, 0.6, 0.7], gt_cls=[0, 0, 0],
    )
    assert 0 not in det.margins and det.margin(0) == det.margins[-1]


def test_matching_is_class_aware_and_greedy():
    gt = np.array([[0, 0, 10, 10], [20, 20, 30, 30]])
    pred = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30]])
    p2g, g2p = match_detections(pred, [0.5, 0.9, 0.8], [0, 0, 1], gt, [0, 0])
    assert p2g.tolist() == [-1, 0, -1] and g2p.tolist() == [1, -1]
    assert np.isclose(box_iou(gt, gt).diagonal(), 1).all()
