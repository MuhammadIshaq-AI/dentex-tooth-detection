"""Local web demo: upload a panoramic dental X-ray and get diagnosis detections with uncertainty.

    python app/app.py                 # then open http://127.0.0.1:7860

Research prototype only - not a medical device and not for clinical decisions.
"""
import argparse
import json
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd
from ultralytics import YOLO

from dentex.labels import Boxes, predict
from dentex.uncertainty.conformal import ConformalDetector, PlattCalibrator
from dentex.visualize import draw

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "runs/dentex/yolo26s_baseline/weights/best.pt"
DEFAULT_MC_WEIGHTS = ROOT / "runs/dentex/yolo26s_mcdropout/weights/best.pt"
DEFAULT_PARAMS = ROOT / "results/conformal_params.json"

DESCRIPTIONS = {
    "Impacted": "Tooth that has not erupted into its normal position (often a wisdom tooth).",
    "Caries": "Tooth decay limited to the outer layers of the tooth.",
    "Deep Caries": "Decay reaching close to the pulp; typically needs root-canal treatment or extraction.",
    "Periapical Lesion": "Radiolucent area at the root tip, usually a sign of infection or inflammation.",
}
DISCLAIMER = ("**Research prototype - not a medical device.** Findings are model predictions on panoramic X-rays "
              "trained on the DENTEX dataset and must be reviewed by a qualified dentist.")


def approx_region(box, width, height):
    """Approximate FDI quadrant from box position (image left = patient's right)."""
    cx, cy = (box[0] + box[2]) / 2 / width, (box[1] + box[3]) / 2 / height
    upper, patient_right = cy < 0.5, cx < 0.5
    quadrant = {(True, True): 1, (True, False): 2, (False, False): 3, (False, True): 4}[(upper, patient_right)]
    return f"Q{quadrant} ({'upper' if upper else 'lower'} {'right' if patient_right else 'left'})"


def status_from_confidence(conf):
    if conf >= 0.7:
        return "Confident"
    if conf >= 0.4:
        return "Probable"
    return "Uncertain - review"


def status_from_mc(conf, freq, entropy, agree):
    """Combine calibrated confidence with MC-dropout stability; any weak signal flags the finding."""
    if not agree:
        return "Uncertain - models disagree"
    if conf < 0.4 or freq < 0.6 or entropy > 0.5:
        return "Uncertain - review"
    return "Confident" if conf >= 0.7 else "Probable"


class Engine:
    def __init__(self, weights, mc_weights, params_path, imgsz=1024):
        self.imgsz = imgsz
        self.model = YOLO(str(weights))
        self.names = {int(k): v for k, v in self.model.names.items()}
        self.mc_model = YOLO(str(mc_weights)) if mc_weights and Path(mc_weights).exists() else None
        self.platt, self.conformal = None, None
        if params_path and Path(params_path).exists():
            p = json.loads(Path(params_path).read_text())
            self.platt = PlattCalibrator(params={int(k): tuple(v) for k, v in p["platt"].items()},
                                         global_params=tuple(p["platt_global"]))
            self.conformal = ConformalDetector(alpha=p["alpha"], margins={int(k): v for k, v in p["margins"].items()},
                                               thresholds={int(k): v for k, v in p["thresholds"].items()})

    def analyze(self, image_path, conf_thr, calibrate, intervals, use_mc, mc_passes):
        if not image_path:
            raise gr.Error("Please upload a panoramic X-ray first.")
        image = cv2.imread(str(image_path))
        if image is None:
            raise gr.Error("Could not read the uploaded image.")
        h, w = image.shape[:2]

        boxes = predict(self.model, [image_path], imgsz=self.imgsz, conf=0.05)[0]
        use_cal = bool(calibrate and self.platt is not None and len(boxes.cls))
        shown_conf = self.platt.transform(boxes.conf, boxes.cls) if use_cal else boxes.conf
        mode = "Platt-calibrated confidence" if use_cal else "raw model confidence"

        # MC-dropout adds uncertainty to the baseline findings rather than replacing them: the dropout
        # fine-tuned model scores on a different scale, so its fused boxes are matched to baseline boxes
        stats = None
        if use_mc and self.mc_model is not None and len(boxes.cls):
            from dentex.uncertainty.matching import box_iou
            from dentex.uncertainty.mc_dropout import mc_predict

            fused, fused_stats = mc_predict(self.mc_model, [image_path], T=int(mc_passes), imgsz=self.imgsz,
                                            conf=0.01, num_classes=len(self.names))[0]
            n = len(boxes.cls)  # unmatched findings keep maximal uncertainty
            stats = {"freq": np.zeros(n), "entropy": np.ones(n), "agree": np.zeros(n, bool), "mc_cls": np.full(n, -1)}
            if len(fused.cls):
                iou = box_iou(boxes.xyxy, fused.xyxy)
                best = iou.argmax(1)
                ok = iou[np.arange(n), best] >= 0.5
                stats["freq"][ok] = fused_stats["freq"][best[ok]]
                stats["entropy"][ok] = fused_stats["class_entropy"][best[ok]]  # diagnosis disagreement
                stats["mc_cls"][ok] = fused.cls[best[ok]]
                stats["agree"] = ok & (stats["mc_cls"] == boxes.cls)
            mode += f"; MC-dropout uncertainty from {int(mc_passes)} stochastic passes"

        keep = shown_conf >= conf_thr
        order = np.argsort(-shown_conf[keep])
        shown = Boxes(boxes.xyxy[keep][order], boxes.cls[keep][order], shown_conf[keep][order])
        shown_stats = {k: v[keep][order] for k, v in stats.items()} if stats is not None else None

        margins = None
        if intervals and self.conformal is not None:
            margins = np.array([self.conformal.margin(c) for c in shown.cls])

        annotated = draw(image, shown, self.names, f"{len(shown.cls)} findings", margins, shown_stats)
        scale = min(1.0, 1800 / w)
        annotated = cv2.resize(annotated, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        rows = []
        for i, (b, c, s) in enumerate(zip(shown.xyxy, shown.cls, shown.conf)):
            name = self.names[int(c)]
            row = {"#": i + 1, "Finding": name, "Confidence": round(float(s), 3),
                   "Approx. region": approx_region(b, w, h)}
            if shown_stats is not None:
                mc_cls = int(shown_stats["mc_cls"][i])
                row["Seen in % of passes"] = round(100 * float(shown_stats["freq"][i]))
                row["Class entropy (0-1)"] = round(float(shown_stats["entropy"][i]), 3)
                row["MC class"] = self.names[mc_cls] if mc_cls >= 0 else "not detected"
                row["Status"] = status_from_mc(float(s), float(shown_stats["freq"][i]),
                                               float(shown_stats["entropy"][i]), bool(shown_stats["agree"][i]))
            else:
                row["Status"] = status_from_confidence(float(s))
            if margins is not None:
                m = margins[i]
                row["90% box interval (± px)"] = (f"±{m * (b[2] - b[0]):.0f} x, ±{m * (b[3] - b[1]):.0f} y"
                                                  if np.isfinite(m) else "n/a")
            row["Box (x1, y1, x2, y2)"] = ", ".join(f"{v:.0f}" for v in b)
            rows.append(row)
        table = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["#", "Finding", "Confidence", "Status"])

        counts = pd.Series([self.names[int(c)] for c in shown.cls], dtype=object).value_counts()
        lines = [f"### {len(shown.cls)} finding(s) at confidence ≥ {conf_thr:.2f}", f"_Mode: {mode}_", ""]
        for name, n in counts.items():
            review = sum(1 for r in rows if r["Finding"] == name and r["Status"].startswith("Uncertain"))
            lines.append(f"- **{name}** × {n}" + (f" ({review} flagged for review)" if review else "")
                         + f" - {DESCRIPTIONS.get(name, '')}")
        if not len(shown.cls):
            lines.append("No findings above the threshold. Try lowering the confidence threshold.")
        if margins is not None:
            lines += ["", "Thin outer boxes show the 90% conformal interval for where the lesion's true extent lies."]
        lines += ["", DISCLAIMER]
        return annotated, table, "\n".join(lines)


def build_ui(engine: Engine, examples):
    with gr.Blocks(title="DENTEX dental X-ray analysis") as demo:
        gr.Markdown("# Dental panoramic X-ray analysis\n"
                    "Upload a panoramic X-ray to detect **impacted teeth, caries, deep caries and periapical lesions** "
                    "with YOLO26, plus calibrated confidence and uncertainty estimates.\n\n" + DISCLAIMER)
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                image_in = gr.Image(type="filepath", label="Panoramic X-ray (PNG/JPG)", height=320)
                conf_thr = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Confidence threshold")
                calibrate = gr.Checkbox(value=True, label="Calibrated confidence (Platt scaling)")
                intervals = gr.Checkbox(value=True, label="Show 90% conformal box intervals")
                use_mc = gr.Checkbox(value=False, label="MC-dropout uncertainty (slower)",
                                     interactive=engine.mc_model is not None,
                                     info=None if engine.mc_model is not None else "MC-dropout weights not found")
                mc_passes = gr.Slider(5, 30, value=20, step=1, label="MC-dropout passes")
                run = gr.Button("Analyze", variant="primary")
                if examples:
                    gr.Examples(examples=[[str(e)] for e in examples], inputs=[image_in], label="Test-set examples")
            with gr.Column(scale=2, min_width=320):
                image_out = gr.Image(type="numpy", label="Detections", height=520)
                summary = gr.Markdown()
        table = gr.Dataframe(label="Findings", wrap=True)
        run.click(engine.analyze, [image_in, conf_thr, calibrate, intervals, use_mc, mc_passes],
                  [image_out, table, summary])
    return demo


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--mc-weights", default=str(DEFAULT_MC_WEIGHTS))
    ap.add_argument("--params", default=str(DEFAULT_PARAMS), help="calibration/conformal params from run_uncertainty.py")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    if not Path(args.weights).exists():
        raise SystemExit(f"Weights not found: {args.weights}. Train first (see README) or pass --weights.")
    engine = Engine(args.weights, args.mc_weights, args.params, args.imgsz)
    test_dir = ROOT / "data/yolo/images/test"
    examples = [p for p in (test_dir / n for n in ("test_110.png", "test_9.png", "test_42.png", "test_1.png")) if p.exists()]
    build_ui(engine, examples).launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
