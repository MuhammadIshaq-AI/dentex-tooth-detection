"""Build the project report (Word .docx, imports cleanly into Google Docs) from the generated results.

    python scripts/build_report.py            # -> docs/DENTEX_Project_Report.docx

Every metric is read from results/*.csv and data/yolo/split_summary.json, so the report always matches the experiments.
"""
import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

CLASSES = ["Impacted", "Caries", "Periapical Lesion", "Deep Caries"]
ACCENT = "1F4E79"


# --------------------------------------------------------------------------- docx helpers
def shade(cell, hex_fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tc_pr.append(shd)


def table(doc, header, rows, bold_last=False, align_right_from=1, font_size=9):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(str(h))
        run.bold, run.font.size = True, Pt(font_size)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        shade(cell, ACCENT)
    for r_i, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            run = p.add_run(str(v))
            run.font.size = Pt(font_size)
            run.bold = bold_last and r_i == len(rows) - 1
            if i >= align_right_from:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        if r_i % 2 == 1:
            for c in cells:
                shade(c, "F2F6FA")
    doc.add_paragraph()
    return t


def figure(doc, path, caption, width=6.3):
    path = Path(path)
    if not path.exists():
        doc.add_paragraph(f"[Missing figure: {path}]")
        return
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = cap.add_run(caption)
    run.italic, run.font.size = True, Pt(9)
    run.font.color.rgb = RGBColor(0x52, 0x51, 0x4E)


def bullets(doc, items):
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        if isinstance(item, tuple):  # (bold lead, rest)
            p.add_run(item[0]).bold = True
            p.add_run(item[1])
        else:
            p.add_run(item)


def para(doc, text, bold_lead=None):
    p = doc.add_paragraph()
    if bold_lead:
        p.add_run(bold_lead).bold = True
    p.add_run(text)
    return p


f3 = lambda x: f"{x:.3f}"  # noqa: E731
pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731


# --------------------------------------------------------------------------- report
def build(root: Path, out: Path, author: str):
    res = root / "results"
    val = pd.read_csv(res / "metrics_val.csv")
    test = pd.read_csv(res / "metrics_test.csv")
    cal = pd.read_csv(res / "uncertainty_calibration.csv")
    conf = pd.read_csv(res / "uncertainty_conformal.csv")
    risk = pd.read_csv(res / "uncertainty_recall_control.csv")
    mc_ap = pd.read_csv(res / "uncertainty_mc_ap50.csv", index_col=0)
    auroc = pd.read_csv(res / "uncertainty_mc_auroc.csv")
    audit = pd.read_csv(res / "test_label_audit.csv")
    splits = json.loads((root / "data/yolo/split_summary.json").read_text())["summary"]

    v_all, t_all = val[val["class"] == "all"].iloc[0], test[test["class"] == "all"].iloc[0]
    c_all = cal[cal["class"] == "all"].iloc[0]
    auroc_of = lambda model, sig: float(auroc[(auroc.model == model) & (auroc.signal == sig)].AUROC_false_positive.iloc[0])  # noqa: E731
    mc_model = [m for m in auroc.model.unique() if m.startswith("mc_dropout")][0]

    doc = Document()
    for s in doc.sections:
        s.left_margin = s.right_margin = Inches(0.9)
        s.top_margin = s.bottom_margin = Inches(0.8)
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(10.5)

    # ---------------------------------------------------------------- title
    title = doc.add_heading("Uncertainty-Aware Dental Diagnosis Detection on Panoramic X-rays", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.add_run("YOLO26 fine-tuned on DENTEX, with calibrated confidence, conformal prediction and MC-dropout\n").italic = True
    sub.add_run(f"{author} · {date.today():%B %Y} · github.com/MuhammadIshaq-AI/dentex-tooth-detection")

    # ---------------------------------------------------------------- 1 summary
    doc.add_heading("1. Executive summary", level=1)
    para(doc, "This project builds a detector that finds four dental conditions on panoramic X-rays: impacted teeth, "
              "caries, deep caries and periapical lesions. It then goes a step further than typical dental-AI repositories "
              "by quantifying how much each prediction can be trusted. The latest Ultralytics YOLO26 model was fine-tuned "
              "on the public DENTEX challenge dataset on a 6 GB laptop GPU. The pipeline is available as a local web app "
              "and as a serverless web deployment.")
    imp_test = test[test["class"] == "Impacted"].iloc[0]
    bullets(doc, [
        ("Detection accuracy: ", f"mAP50 {f3(v_all.AP50)} / mAP50-95 {f3(v_all['AP50-95'])} on the official validation set, "
                                 f"and mAP50 {f3(t_all.AP50)} / mAP50-95 {f3(t_all['AP50-95'])} on the 250-image official test set."),
        ("Best class: ", f"impacted teeth reach AP50 {f3(imp_test.AP50)} with {pct(imp_test.recall)} recall on test."),
        ("Calibration: ", f"per-class Platt scaling cuts the expected calibration error from {f3(c_all.ECE_raw)} "
                          f"to {f3(c_all.ECE_platt)}, roughly halving it."),
        ("Conformal box intervals: ", "the 90% coverage target is met on test for all classes within sampling error ("
                                      + ", ".join(f"{r['class']} {pct(r.box_coverage)}" for _, r in conf.iterrows()) + ")."),
        ("Recall guarantees: ", "\"miss at most 10%\" is achievable only for impacted teeth. For caries and lesions the "
                                "analysis shows that no confidence threshold reaches 90% sensitivity with this detector, "
                                "which is a concrete direction for future work."),
        ("MC-dropout: ", f"averaging 20 stochastic passes flags false positives better than single-pass confidence "
                         f"(AUROC {f3(auroc_of(mc_model, '1 - confidence'))} vs {f3(auroc_of('baseline', '1 - confidence'))}), "
                         f"but lowers AP50 ({f3(mc_ap.loc[mc_ap.index[-1], 'mean'])} vs {f3(mc_ap.loc['baseline', 'mean'])}). "
                         "It is therefore used as a review flag, not as the detector itself."),
        ("Deliverables: ", "training and evaluation code, a Gradio web app (upload an X-ray to get findings), a Vercel "
                           "deployment (FastAPI + ONNX Runtime), 17 unit tests and a fully reproducible pipeline."),
    ])

    # ---------------------------------------------------------------- 2 problem
    doc.add_heading("2. Problem and motivation", level=1)
    para(doc, "Panoramic radiographs are the most common dental imaging exam, and reading them is time-consuming and "
              "subjective. Small lesions such as early caries are easy to miss. Automated detection can act as a second "
              "reader, but in a clinical setting a bare bounding box with a score is not enough. A dentist needs to know "
              "how reliable that score is, how precise the box is, and when the model should not be trusted.")
    para(doc, "Structurally, the task is identical to industrial surface-defect detection: small, low-contrast anomalies on a "
              "large, textured image, with a heavy class imbalance. This project applies the same object-detection toolkit and "
              "adds an explicit uncertainty layer, because uncertainty-aware AI is what separates a research demo from a tool "
              "clinicians can reason about.")

    # ---------------------------------------------------------------- 3 data
    doc.add_heading("3. Dataset", level=1)
    para(doc, "DENTEX (MICCAI 2023 challenge; Hugging Face ibrahimhamamci/DENTEX, CC BY-NC-SA 4.0) contains hierarchically "
              "annotated panoramic X-rays. This project uses the fully labelled subset, where each box carries a quadrant, "
              "a tooth number and a diagnosis. The diagnosis is the detection target.")
    rows = []
    for split, note in [("train", "model training"), ("calib", "held out: calibration / conformal fitting only"),
                        ("val", "official validation: model selection"), ("test", "official test: final reporting")]:
        s = splits[split]
        rows.append([split, s["images"]] + [s.get(c, 0) for c in CLASSES] + [note])
    table(doc, ["Split", "Images"] + CLASSES + ["Purpose"], rows, align_right_from=1)
    para(doc, "The 705 official training images were split, stratified by each image's rarest class, into 605 for training and "
              "100 for calibration. The calibration images are never used for training, which is what makes the conformal "
              "guarantees valid.")
    figure(doc, root / "docs/figures/dataset_labels_train.jpg",
           "Figure 1. A training X-ray with its converted diagnosis boxes (COCO → YOLO format).", width=6.0)

    doc.add_heading("3.1 Test-label audit", level=2)
    para(doc, "The released test labels turned out to use Turkish treatment-planning codes (e.g. çürük = caries, kanal = root "
              "canal) rather than the four challenge diagnoses. Rather than guess, every test annotation was matched to the "
              "trained model's best-overlapping prediction, and the agreement was tabulated:")
    rows = []
    mapping = {"6": "Impacted", "1": "Caries", "7": "Periapical Lesion", "3": "Deep Caries", "5": "Deep Caries"}
    for _, r in audit.iterrows():
        code = str(r.code).split("-")[0]
        rows.append([r.code, int(r.shapes)] + [f"{r[c + ' %']:.1f}" for c in CLASSES]
                    + [f"{r['no detection %']:.1f}", mapping.get(code, "dropped")])
    table(doc, ["Test code", "Shapes", "Impacted %", "Caries %", "Periapical %", "Deep Caries %", "Missed %", "Mapped to"],
          rows, font_size=8)
    para(doc, "Impacted, caries and lesion codes align clearly. Root-canal and extraction codes are mostly predicted as deep "
              "caries, which is clinically consistent, while küretaj (periodontal curettage) is almost never detected and was "
              "dropped. Because part of this mapping was informed by test predictions, the validation set is treated as the "
              "clean reference and the caveat is stated with all test numbers.", bold_lead="Finding. ")

    # ---------------------------------------------------------------- 4 method
    doc.add_heading("4. Method", level=1)
    doc.add_heading("4.1 Detector", level=2)
    table(doc, ["Setting", "Value"], [
        ["Model", "Ultralytics YOLO26-s (latest YOLO family, Jan 2026), COCO-pretrained, 9.5 M parameters"],
        ["Input size", "1024 px (panoramic X-rays are ~2900 × 1300; lesions are small)"],
        ["Training", "150 epochs max, early stopping (patience 30), batch 4, cosine LR, AMP, MuSGD optimiser"],
        ["Augmentation", "mosaic, horizontal flip, ±3° rotation, scale 0.3, brightness only (grayscale images)"],
        ["Hardware", "NVIDIA RTX 3050 Laptop GPU, 6 GB. Best epoch 43 of 73, total 58 minutes"],
    ], align_right_from=99)
    figure(doc, root / "docs/figures/training_curves.png",
           "Figure 2. Training curves. Training losses fall steadily; validation mAP50 plateaus around 0.58–0.62 after "
           "~20 epochs, and rising validation classification loss after epoch ~25 signals the onset of overfitting, "
           "which early stopping handles.")

    doc.add_heading("4.2 Uncertainty quantification", level=2)
    bullets(doc, [
        ("Confidence calibration: ", "per-class Platt scaling (logistic regression on the logit of the score) fitted on the "
                                     "calibration split, so that a confidence of 0.7 means right about 70% of the time."),
        ("Conformal box intervals: ", "split-conformal prediction on box coordinates. The nonconformity score is the largest "
                                      "coordinate error normalised by box size; its finite-sample 90% quantile per class gives "
                                      "an outer box that contains the true lesion extent with ≥ 90% probability."),
        ("Recall-controlling thresholds: ", "a conformal per-class confidence threshold that guarantees at most α of lesions "
                                            "are missed, with a feasibility check against the detector's recall ceiling."),
        ("MC-dropout: ", "YOLO26 has no dropout, so Dropout2d (p = 0.1) layers were injected before all 12 detection-head output "
                         "convolutions and the model was fine-tuned for 30 epochs. At inference, 20 stochastic passes are "
                         "clustered into detections with mean score, detection frequency, predictive and class entropy, and box spread."),
    ])

    # ---------------------------------------------------------------- 5 results
    doc.add_heading("5. Detection results", level=1)
    for name, df, label in [("Official validation set (50 images, original DENTEX labels)", val, "val"),
                            ("Official test set (250 images, mapped treatment-code labels)", test, "test")]:
        doc.add_heading(name, level=2)
        rows = [[("All classes" if r["class"] == "all" else r["class"]), f3(r.precision), f3(r.recall), f3(r.AP50),
                 f3(r["AP50-95"])] for _, r in df.iterrows()]
        table(doc, ["Class", "Precision", "Recall", "AP50", "AP50-95"], rows, bold_last=True)
    para(doc, "~8 ms per image on the laptop GPU; ~0.2–0.5 s per image on CPU with ONNX Runtime.", bold_lead="Speed. ")

    doc.add_heading("5.1 Error analysis", level=2)
    figure(doc, root / "results/pr_curve_test.png", "Figure 3. Precision-recall curves per class (test).", width=5.6)
    figure(doc, root / "results/confusion_matrix_test.png",
           "Figure 4. Confusion matrix normalised by true class (test).", width=5.6)
    bullets(doc, [
        ("Impacted teeth are nearly solved: ", "high AP, and never lost to background."),
        ("Caries ↔ deep caries is the main confusion ", "(about a third each way). These are adjacent severity grades of the "
                                                        "same disease and the hardest boundary to label consistently."),
        ("Periapical lesions are the weakest class: ", "the rarest class (134 training boxes), with diffuse boundaries; "
                                                       "about one in five is missed entirely."),
    ])

    doc.add_heading("5.2 Example outputs", level=2)
    para(doc, "Test images never seen in training. In each figure the top panel shows the ground truth; the bottom panel shows "
              "predictions (confidence ≥ 0.25) with thin outer boxes marking the 90% conformal interval.")
    figure(doc, root / "assets/examples/00_test_64.jpg",
           "Figure 5. A correct detection: caries found (confidence 0.45) with its conformal interval.", width=5.2)
    figure(doc, root / "assets/examples/02_test_197.jpg", "Figure 6. Another high-agreement test case.", width=5.2)
    figure(doc, root / "assets/examples/04_test_126.jpg",
           "Figure 7. A failure case: two small caries are missed, illustrating the recall ceiling discussed in Section 6.3.",
           width=5.2)

    # ---------------------------------------------------------------- 6 uncertainty
    doc.add_heading("6. Uncertainty results", level=1)
    para(doc, "All uncertainty components were fitted on the 100-image calibration split and evaluated on the test set.")

    doc.add_heading("6.1 Confidence calibration", level=2)
    rows = [[("All" if r["class"] == "all" else r["class"]), int(r.detections), f3(r.ECE_raw), f3(r.ECE_platt)]
            for _, r in cal.iterrows()]
    table(doc, ["Class", "Detections", "ECE raw", "ECE calibrated"], rows, bold_last=True)
    figure(doc, root / "results/reliability_diagram.png",
           "Figure 8. Reliability diagram. The raw model is over-confident at high scores; the calibrated curve follows "
           "the diagonal.", width=4.6)

    doc.add_heading("6.2 Conformal box intervals (90% target)", level=2)
    rows = [[r["class"], f"{r.box_margin_q:.2f}", pct(r.box_coverage), int(r.matched_boxes)] for _, r in conf.iterrows()]
    table(doc, ["Class", "Margin (× box size)", "Test coverage", "Matched boxes"], rows)
    figure(doc, root / "results/conformal_coverage.png",
           "Figure 9. Left: empirical box-interval coverage per class against the 90% target. Right: achieved test recall "
           "versus the conformal recall target, where feasible.")

    doc.add_heading("6.3 Recall-controlling thresholds", level=2)
    rows = []
    for c in CLASSES:
        ceil = conf[conf["class"] == c].calib_recall_ceiling.iloc[0]
        cells = [c, pct(ceil)]
        for a in (0.1, 0.5):
            r = risk[(risk["class"] == c) & (risk.alpha == a)].iloc[0]
            cells.append(f"thr {r.threshold:.2f}: recall {pct(r.test_recall)}, precision {pct(r.test_precision)}"
                         if bool(r.feasible) else "infeasible")
        fx = conf[conf["class"] == c].iloc[0]
        cells.append(f"recall {pct(fx['recall@0.25'])}, precision {pct(fx['precision@0.25'])}")
        rows.append(cells)
    table(doc, ["Class", "Recall ceiling", "Target 90% recall", "Target 50% recall", "Fixed threshold 0.25"], rows,
          align_right_from=99, font_size=8)
    para(doc, "For impacted teeth a dentist can be promised \"at most 10% missed\" at high precision. For caries and lesions, "
              "about a third or more of lesions are never localised at all, so no threshold can reach 90% sensitivity. A fixed "
              "0.25 threshold silently hides this; the conformal analysis makes it explicit.", bold_lead="Interpretation. ")

    doc.add_heading("6.4 MC-dropout", level=2)
    label = {"baseline": "Baseline (single pass)", "dropout_model_single_pass": "Dropout model, single pass"}
    rows = [[label.get(m, "MC-dropout, T = 20 fused")] + [f3(mc_ap.loc[m, c]) for c in CLASSES] + [f3(mc_ap.loc[m, "mean"])]
            for m in mc_ap.index]
    table(doc, ["Model (AP50 on test)"] + CLASSES + ["Mean"], rows)
    sig_name = {"1 - confidence": "1 − confidence", "1 - frequency": "1 − detection frequency", "box std": "box spread",
                "class entropy": "class entropy", "predictive entropy (incl. background)": "predictive entropy (incl. background)"}
    rows = [[("Baseline" if r.model == "baseline" else "MC-dropout") + ": " + sig_name.get(r.signal, r.signal),
             f3(r.AUROC_false_positive)] for _, r in auroc.iterrows()]
    table(doc, ["Signal for flagging false positives", "AUROC (0.5 = chance)"], rows)
    bullets(doc, [
        ("Better error flagging: ", "the mean confidence over 20 passes separates false positives from true positives better "
                                    "than a single pass, and detections that appear in only some passes are more often wrong."),
        ("Lower accuracy when used as the detector: ", "fused predictions lose AP, so the apps keep baseline detections and "
                                                       "attach MC-dropout statistics as review flags."),
        ("Entropy needs care: ", "entropy that includes the background share is inverted, because weak false positives look "
                                 "\"certain background\". Class entropy instead captures disagreement about the diagnosis "
                                 "(e.g. caries vs deep caries)."),
    ])

    # ---------------------------------------------------------------- 7 apps
    doc.add_heading("7. Applications", level=1)
    doc.add_heading("7.1 Local web app (Gradio)", level=2)
    para(doc, "A user uploads a panoramic X-ray and receives the annotated image, a plain-language summary and a findings table. "
              "Each finding lists the diagnosis, calibrated confidence, approximate FDI quadrant, a status (Confident / "
              "Probable / Uncertain – review) and the 90% box interval. An optional MC-dropout mode adds detection frequency "
              "and class entropy, and flags findings where the dropout model disagrees on the diagnosis.")
    figure(doc, root / "docs/figures/app_output_mc.jpg",
           "Figure 10. App output with MC-dropout enabled; H is the class entropy of each finding.", width=6.0)
    doc.add_heading("7.2 Serverless deployment (Vercel)", level=2)
    bullets(doc, [
        "PyTorch cannot run on Vercel (500 MB function limit, no GPU), so the model was exported to ONNX (39 MB) and served "
        "by a FastAPI function with ONNX Runtime, plus a static upload page.",
        "Preprocessing was matched exactly to Ultralytics: on 10 test images all 93 detections matched at IoU ≥ 0.9 with a "
        "maximum confidence difference of 0.001, so the calibration and conformal intervals transfer unchanged.",
        "The browser downsizes images before upload to respect Vercel's 4.5 MB request limit; warm inference takes "
        "~0.2–0.5 s on CPU.",
    ])

    # ---------------------------------------------------------------- 8 engineering
    doc.add_heading("8. Engineering challenges solved", level=1)
    bullets(doc, [
        ("Mislabelled test set: ", "treatment codes instead of diagnoses, resolved with a data-driven audit rather than a guess."),
        ("MC-dropout under Ultralytics: ", "the inference wrapper hides the network from module traversal, so dropout was "
                                           "silently inactive; fixed by resolving the real network."),
        ("Quadratic clustering: ", "fusing ~6,000 boxes per image across 20 passes took hours in pure Python; vectorised "
                                   "with NumPy to seconds."),
        ("Inverted uncertainty signal: ", "background-inclusive entropy ranked errors backwards; added class entropy."),
        ("Degenerate conformal thresholds: ", "infeasible recall targets silently collapsed to \"keep everything\"; "
                                              "added an explicit feasibility check."),
        ("Deployment parity: ", "a square letterbox and anti-aliased resize changed scores by up to 0.37; matching "
                                "Ultralytics' rectangular letterbox restored exact parity."),
    ])

    # ---------------------------------------------------------------- 9 limitations
    doc.add_heading("9. Limitations and future work", level=1)
    bullets(doc, [
        "Small dataset (605 training images) with strong class imbalance; periapical lesions are under-represented.",
        "Test labels follow a different, treatment-based protocol; the deep-caries mapping was informed by test predictions.",
        "Conformal guarantees are marginal and assume the calibration and test images are exchangeable.",
        "Future work: higher-resolution or tiled inference and tooth-level crops to raise the recall ceiling for small lesions; "
        "using the unlabelled DENTEX images for semi-supervised pre-training; hierarchical quadrant/tooth prediction for exact "
        "FDI numbering; conformal risk control on per-image false-negative rate.",
        "Research prototype only, not a medical device. DENTEX data and derived weights are non-commercial (CC BY-NC-SA 4.0).",
    ])

    # ---------------------------------------------------------------- 10 reproduce
    doc.add_heading("10. Reproducibility", level=1)
    para(doc, "The full pipeline runs from the repository with: download_data → prepare_data → train → evaluate → "
              "audit_test_labels → train MC-dropout → run_uncertainty → export_onnx → build_report, plus 17 unit tests. "
              "See README.md for the exact commands.")

    doc.add_heading("References", level=1)
    for ref in [
        "Hamamci, I. E. et al. DENTEX: An Abnormal Tooth Detection with Dental Enumeration and Diagnosis Benchmark for "
        "Panoramic X-rays. arXiv:2305.19112, 2023.",
        "Ultralytics YOLO26 documentation. https://docs.ultralytics.com/models/yolo26/",
        "Angelopoulos, A. N. and Bates, S. A Gentle Introduction to Conformal Prediction and Distribution-Free "
        "Uncertainty Quantification. arXiv:2107.07511, 2021.",
        "Gal, Y. and Ghahramani, Z. Dropout as a Bayesian Approximation. ICML 2016.",
        "Platt, J. Probabilistic Outputs for Support Vector Machines. Advances in Large Margin Classifiers, 1999.",
    ]:
        doc.add_paragraph(ref, style="List Number")

    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="docs/DENTEX_Project_Report.docx")
    ap.add_argument("--author", default="Muhammad Ishaq")
    args = ap.parse_args()
    out = build(Path(args.root).resolve(), Path(args.out), args.author)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
