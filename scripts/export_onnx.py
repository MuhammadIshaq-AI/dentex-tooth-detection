"""Export the trained detector to ONNX and package it for the Vercel deployment.

    python scripts/export_onnx.py

Writes deploy/vercel/model/dentex_yolo26s.onnx and deploy/vercel/model/meta.json (class names, input size,
per-class Platt calibration and conformal box margins from results/conformal_params.json).
"""
import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

from dentex.uncertainty.conformal import ConformalDetector


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="runs/dentex/yolo26s_baseline/weights/best.pt")
    ap.add_argument("--params", default="results/conformal_params.json")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--out", default="deploy/vercel/model")
    args = ap.parse_args()

    model = YOLO(args.weights)
    # dynamic H/W: the deployment feeds rectangular letterboxed inputs, exactly like Ultralytics predict
    onnx_path = Path(model.export(format="onnx", imgsz=args.imgsz, simplify=True, dynamic=True))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "dentex_yolo26s.onnx"
    shutil.copy2(onnx_path, target)

    params = json.loads(Path(args.params).read_text())
    cp = ConformalDetector(margins={int(k): v for k, v in params["margins"].items()})
    names = {int(k): v for k, v in model.names.items()}
    meta = {
        "names": {str(k): v for k, v in names.items()},
        "imgsz": args.imgsz,
        "iou_nms": 0.7,
        "alpha": params["alpha"],
        "platt": params["platt"],
        "platt_global": params["platt_global"],
        "margins": {str(c): cp.margin(c) for c in names},
        "source_weights": str(args.weights),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {target} ({target.stat().st_size / 1e6:.1f} MB) and {out / 'meta.json'}")


if __name__ == "__main__":
    main()
