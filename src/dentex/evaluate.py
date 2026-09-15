"""Evaluate a trained model on a split and write per-class metrics.

    python -m dentex.evaluate --weights runs/dentex/yolo26s_baseline/weights/best.pt --split test
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="data/yolo/dentex.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    m = model.val(data=args.data, split=args.split, imgsz=args.imgsz, batch=args.batch, conf=0.001,
                  plots=True, project=str((out / "val_runs").resolve()), name=args.split, exist_ok=True)

    names = m.names
    rows = []
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        rows.append({"class": names[int(c)], "precision": p, "recall": r, "AP50": ap50, "AP50-95": ap})
    rows.append({"class": "all", "precision": m.box.mp, "recall": m.box.mr, "AP50": m.box.map50, "AP50-95": m.box.map})
    df = pd.DataFrame(rows)

    stem = f"metrics_{args.split}"
    df.to_csv(out / f"{stem}.csv", index=False)
    (out / f"{stem}.md").write_text(df.to_markdown(index=False, floatfmt=".3f") + "\n")
    (out / f"{stem}.json").write_text(json.dumps({"weights": args.weights, "split": args.split,
                                                  "rows": rows, "speed_ms": m.speed}, indent=2, default=float))
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
