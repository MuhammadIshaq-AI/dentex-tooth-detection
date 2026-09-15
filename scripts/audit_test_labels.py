"""Audit how the released test-label codes line up with DENTEX diagnosis classes.

The test set uses a LabelMe vocabulary ("<code>-<turkish name>-<FDI>") that differs from the
challenge's four classes. For every test shape (all codes, including the ones we drop), find
the best-overlapping prediction of a model trained on the official training labels and
tabulate code x predicted class. A code that genuinely corresponds to a DENTEX class should
be predicted as that class far more often than as anything else.

    python scripts/audit_test_labels.py --weights runs/dentex/yolo26s_baseline/weights/best.pt
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO

from dentex.convert import parse_labelme_label
from dentex.labels import load_data_yaml, predict
from dentex.uncertainty.matching import box_iou


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="data/yolo/dentex.yaml")
    ap.add_argument("--raw-test", default="data/raw/test_data/disease")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--out", default="results/test_label_audit.csv")
    args = ap.parse_args()

    _, _, names = load_data_yaml(args.data)
    raw = Path(args.raw_test)
    label_files = sorted((raw / "label").glob("*.json"))
    images = [raw / "input" / (f.stem + ".png") for f in label_files]
    preds = predict(YOLO(args.weights), images, imgsz=args.imgsz, conf=args.conf)

    table, code_names = defaultdict(Counter), {}
    for lf, p in zip(label_files, preds):
        for s in json.loads(lf.read_text(encoding="utf-8"))["shapes"]:
            code, _ = parse_labelme_label(s["label"])
            code_names[code] = s["label"].rsplit("-", 1)[0]
            pts = np.array(s["points"], dtype=float)
            box = np.concatenate([pts.min(0), pts.max(0)])[None]
            if len(p.cls):
                iou = box_iou(box, p.xyxy)[0]
                j = int(np.argmax(iou))
                table[code][names[int(p.cls[j])] if iou[j] >= args.iou else "no detection"] += 1
            else:
                table[code]["no detection"] += 1

    cols = [names[i] for i in sorted(names)] + ["no detection"]
    df = pd.DataFrame([{"code": code_names[c], "shapes": sum(table[c].values()),
                        **{k: table[c][k] for k in cols}} for c in sorted(table)])
    for k in cols:
        df[f"{k} %"] = (100 * df[k] / df["shapes"]).round(1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
