"""Uncertainty-aware evaluation: calibration, conformal prediction and MC-dropout.

    python scripts/run_uncertainty.py --weights runs/dentex/yolo26s_baseline/weights/best.pt \
        --mc-weights runs/dentex/yolo26s_mcdropout/weights/best.pt

Everything is fitted on the `calib` split and reported on `test`.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from dentex.labels import Boxes, load_data_yaml, load_gt, predict, split_images  # noqa: E402
from dentex.uncertainty.conformal import (  # noqa: E402
    ConformalDetector, PlattCalibrator, box_nonconformity, expected_calibration_error, reliability_curve,
)
from dentex.uncertainty.matching import match_detections, per_class_ap  # noqa: E402
from dentex.visualize import pick_examples, side_by_side  # noqa: E402

# reference categorical palette (light surface), fixed slot order
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelcolor=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def filt(b: Boxes, keep):
    return Boxes(b.xyxy[keep], b.cls[keep], b.conf[keep])


def collect(preds, gts, conf_floor):
    """Detection-level (score, correct, cls), matched box pairs, and per-GT matched scores."""
    det_s, det_ok, det_c, mp, mg, mc, gt_s, gt_c = [], [], [], [], [], [], [], []
    for p, g in zip(preds, gts):
        p = filt(p, p.conf >= conf_floor)
        p2g, g2p = match_detections(p.xyxy, p.conf, p.cls, g.xyxy, g.cls)
        det_s += list(p.conf); det_ok += list(p2g >= 0); det_c += list(p.cls)
        hit = p2g >= 0
        mp += list(p.xyxy[hit]); mg += list(g.xyxy[p2g[hit]]); mc += list(p.cls[hit])
        gt_s += [p.conf[j] if j >= 0 else 0.0 for j in g2p]; gt_c += list(g.cls)
    arr = lambda x, shape=None: np.array(x).reshape(shape) if shape else np.array(x)  # noqa: E731
    return dict(det_s=arr(det_s), det_ok=arr(det_ok), det_c=arr(det_c), mp=arr(mp, (-1, 4)), mg=arr(mg, (-1, 4)),
                mc=arr(mc), gt_s=arr(gt_s), gt_c=arr(gt_c))


def threshold_metrics(preds, gts, thr_fn, nc):
    """Per-class recall/precision when keeping predictions with conf >= thr_fn(cls)."""
    tp, fp, ngt = np.zeros(nc), np.zeros(nc), np.zeros(nc)
    for p, g in zip(preds, gts):
        keep = p.conf >= np.array([thr_fn(c) for c in p.cls]) if len(p.cls) else np.zeros(0, bool)
        q = filt(p, keep)
        p2g, _ = match_detections(q.xyxy, q.conf, q.cls, g.xyxy, g.cls)
        for c in range(nc):
            tp[c] += ((q.cls == c) & (p2g >= 0)).sum()
            fp[c] += ((q.cls == c) & (p2g < 0)).sum()
            ngt[c] += (g.cls == c).sum()
    return tp / np.maximum(ngt, 1), tp / np.maximum(tp + fp, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--mc-weights", default=None, help="dropout fine-tuned weights (enables MC-dropout)")
    ap.add_argument("--data", default="data/yolo/dentex.yaml")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--conf-floor", type=float, default=0.05, help="min confidence of reported detections")
    ap.add_argument("--deploy-conf", type=float, default=0.25, help="fixed-threshold baseline")
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--out", default="results")
    ap.add_argument("--examples", default="assets/examples")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    _, _, names = load_data_yaml(args.data)
    nc, cls_names = len(names), [names[i] for i in range(len(names))]
    calib_paths, test_paths = split_images(args.data, "calib"), split_images(args.data, "test")
    calib_gt, test_gt = [load_gt(p) for p in calib_paths], [load_gt(p) for p in test_paths]

    model = YOLO(args.weights)
    calib_pred = predict(model, calib_paths, imgsz=args.imgsz, conf=0.001)
    test_pred = predict(model, test_paths, imgsz=args.imgsz, conf=0.001)
    cal, tst = collect(calib_pred, calib_gt, args.conf_floor), collect(test_pred, test_gt, args.conf_floor)
    summary = {"alpha": args.alpha, "n_calib_images": len(calib_paths), "n_test_images": len(test_paths)}

    # ---------------------------------------------------------------- 1. calibration
    platt = PlattCalibrator().fit(cal["det_s"], cal["det_ok"], cal["det_c"])
    test_cal = platt.transform(tst["det_s"], tst["det_c"])
    rows = []
    for c in list(range(nc)) + [None]:
        m = np.ones(len(tst["det_c"]), bool) if c is None else tst["det_c"] == c
        rows.append({"class": "all" if c is None else cls_names[c], "detections": int(m.sum()),
                     "ECE_raw": expected_calibration_error(tst["det_s"][m], tst["det_ok"][m]),
                     "ECE_platt": expected_calibration_error(test_cal[m], tst["det_ok"][m])})
    calib_df = pd.DataFrame(rows)
    calib_df.to_csv(out / "uncertainty_calibration.csv", index=False)
    summary["ece_raw"], summary["ece_platt"] = rows[-1]["ECE_raw"], rows[-1]["ECE_platt"]

    fig, ax = plt.subplots(figsize=(5.2, 4.6), facecolor=SURFACE)
    style_axes(ax)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=(0, (4, 3)), label="Perfect calibration")
    for conf, label, color in [(tst["det_s"], f"Raw  (ECE {summary['ece_raw']:.3f})", SERIES[0]),
                               (test_cal, f"Platt (ECE {summary['ece_platt']:.3f})", SERIES[1])]:
        _, acc, mconf, cnt = reliability_curve(conf, tst["det_ok"], n_bins=10)
        ok = cnt > 0
        ax.plot(mconf[ok], acc[ok], color=color, linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted confidence", color=INK2); ax.set_ylabel("Observed precision (IoU ≥ 0.5)", color=INK2)
    ax.set_title("Confidence calibration on test", color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9, loc="upper left")
    fig.tight_layout(); fig.savefig(out / "reliability_diagram.png", dpi=160); plt.close(fig)

    # ---------------------------------------------------------------- 2. conformal
    cp = ConformalDetector(alpha=args.alpha).fit(cal["mp"], cal["mg"], cal["mc"], cal["gt_s"], cal["gt_c"])
    nonconf = box_nonconformity(tst["mp"], tst["mg"])
    rec_cp, prec_cp = threshold_metrics(test_pred, test_gt, cp.threshold, nc)
    rec_fx, prec_fx = threshold_metrics(test_pred, test_gt, lambda c: args.deploy_conf, nc)
    rows = []
    for c in range(nc):
        m = tst["mc"] == c
        rows.append({"class": cls_names[c], "box_margin_q": cp.margin(c),
                     "box_coverage": float((nonconf[m] <= cp.margin(c)).mean()) if m.any() else np.nan,
                     "matched_boxes": int(m.sum()), "conformal_threshold": cp.threshold(c),
                     "recall_conformal": rec_cp[c], "precision_conformal": prec_cp[c],
                     f"recall@{args.deploy_conf}": rec_fx[c], f"precision@{args.deploy_conf}": prec_fx[c]})
    conf_df = pd.DataFrame(rows)
    conf_df.to_csv(out / "uncertainty_conformal.csv", index=False)
    (out / "conformal_params.json").write_text(json.dumps(
        {"alpha": args.alpha, "margins": cp.margins, "thresholds": cp.thresholds,
         "platt": {str(k): v for k, v in platt.params.items()}, "platt_global": platt.global_params},
        indent=2, default=float))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), facecolor=SURFACE)
    x = np.arange(nc)
    for ax, col, title in [(axes[0], "box_coverage", "Box-interval coverage"),
                           (axes[1], "recall_conformal", "Recall at conformal threshold")]:
        style_axes(ax)
        vals = conf_df[col].to_numpy()
        ax.bar(x, vals, width=0.6, color=SERIES[:nc], edgecolor=SURFACE, linewidth=2)
        ax.axhline(1 - args.alpha, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(nc - 0.5, 1 - args.alpha + 0.015, f"target {1 - args.alpha:.0%}", color=INK2, fontsize=8, ha="right")
        for xi, v in zip(x, vals):
            if np.isfinite(v):
                ax.text(xi, v + 0.015, f"{v:.0%}", ha="center", color=INK, fontsize=9)
        ax.set_xticks(x, cls_names, fontsize=8.5); ax.set_ylim(0, 1.08)
        ax.set_title(title, color=INK, loc="left", fontsize=11)
    fig.tight_layout(); fig.savefig(out / "conformal_coverage.png", dpi=160); plt.close(fig)

    # ---------------------------------------------------------------- 3. MC-dropout
    mc_stats = None
    if args.mc_weights:
        from dentex.uncertainty.mc_dropout import mc_predict

        mc_model = YOLO(args.mc_weights)
        single = predict(mc_model, test_paths, imgsz=args.imgsz, conf=0.001)
        fused = mc_predict(mc_model, test_paths, T=args.T, imgsz=args.imgsz, conf=0.01, num_classes=nc)
        mc_pred, mc_stats = [f[0] for f in fused], [f[1] for f in fused]
        ap_rows = {"baseline": per_class_ap(test_pred, test_gt, nc),
                   "dropout_model_single_pass": per_class_ap(single, test_gt, nc),
                   f"mc_dropout_T{args.T}": per_class_ap(mc_pred, test_gt, nc)}
        ap_df = pd.DataFrame(ap_rows, index=cls_names).T
        ap_df["mean"] = ap_df.mean(axis=1)
        ap_df.to_csv(out / "uncertainty_mc_ap50.csv")

        # does uncertainty flag false positives? AUROC over detections with conf >= floor
        rows, fp_flags, feats = [], [], {k: [] for k in ("1 - confidence", "entropy", "1 - frequency", "box std")}
        for p, s, g in zip(mc_pred, mc_stats, test_gt):
            keep = p.conf >= args.conf_floor
            q = filt(p, keep)
            p2g, _ = match_detections(q.xyxy, q.conf, q.cls, g.xyxy, g.cls)
            fp_flags += list(p2g < 0)
            feats["1 - confidence"] += list(1 - q.conf)
            feats["entropy"] += list(s["entropy"][keep])
            feats["1 - frequency"] += list(1 - s["freq"][keep])
            feats["box std"] += list(s["box_std"][keep])
        base_fp, base_u = [], []
        for p, g in zip(test_pred, test_gt):
            q = filt(p, p.conf >= args.conf_floor)
            p2g, _ = match_detections(q.xyxy, q.conf, q.cls, g.xyxy, g.cls)
            base_fp += list(p2g < 0); base_u += list(1 - q.conf)
        rows.append({"model": "baseline", "signal": "1 - confidence", "AUROC_false_positive": roc_auc_score(base_fp, base_u)})
        for k, v in feats.items():
            rows.append({"model": f"mc_dropout_T{args.T}", "signal": k, "AUROC_false_positive": roc_auc_score(fp_flags, v)})
        pd.DataFrame(rows).to_csv(out / "uncertainty_mc_auroc.csv", index=False)
        summary["mc_ap50_mean"] = float(ap_df["mean"].iloc[-1])
        summary["baseline_ap50_mean"] = float(ap_df["mean"].iloc[0])

    # ---------------------------------------------------------------- examples
    show_pred, show_stats = (mc_pred, mc_stats) if mc_stats is not None else (test_pred, None)
    for rank, i in enumerate(pick_examples(test_gt, test_pred, conf_thr=args.deploy_conf)):
        side_by_side(test_paths[i], test_gt[i], show_pred[i], names,
                     Path(args.examples) / f"{rank:02d}_{test_paths[i].stem}.jpg",
                     conf_thr=args.deploy_conf, margins_fn=cp.margin,
                     stats=show_stats[i] if show_stats is not None else None)

    (out / "uncertainty_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(calib_df.to_string(index=False), "\n")
    print(conf_df.to_string(index=False), "\n")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
