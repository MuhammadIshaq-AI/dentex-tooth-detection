"""Build a short, plain-language project overview (Word .docx, Google Docs compatible).

    python scripts/build_overview.py          # -> docs/DENTEX_Project_Overview.docx

Covers what the project is about, the DENTEX dataset, the key results and the findings. Metrics are read from
results/*.csv. For the full technical write-up see scripts/build_report.py.
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_report import CLASSES, bullets, f3, figure, para, pct, table  # noqa: E402


def build(root: Path, out: Path, author: str):
    res = root / "results"
    val = pd.read_csv(res / "metrics_val.csv")
    test = pd.read_csv(res / "metrics_test.csv")
    cal = pd.read_csv(res / "uncertainty_calibration.csv")
    conf = pd.read_csv(res / "uncertainty_conformal.csv")
    mc_ap = pd.read_csv(res / "uncertainty_mc_ap50.csv", index_col=0)
    auroc = pd.read_csv(res / "uncertainty_mc_auroc.csv")
    splits = json.loads((root / "data/yolo/split_summary.json").read_text())["summary"]

    v_all, t_all = val[val["class"] == "all"].iloc[0], test[test["class"] == "all"].iloc[0]
    c_all = cal[cal["class"] == "all"].iloc[0]
    get_auroc = lambda m, s: float(auroc[(auroc.model == m) & (auroc.signal == s)].AUROC_false_positive.iloc[0])  # noqa: E731
    mc_model = [m for m in auroc.model.unique() if m.startswith("mc_dropout")][0]
    cls_row = lambda df, c: df[df["class"] == c].iloc[0]  # noqa: E731

    doc = Document()
    for s in doc.sections:
        s.left_margin = s.right_margin = Inches(0.9)
        s.top_margin = s.bottom_margin = Inches(0.8)
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)

    t = doc.add_heading("DENTEX Tooth Detection: Project Overview", level=0)
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.add_run("AI that finds dental problems on X-rays and says how sure it is\n").italic = True
    sub.add_run(f"{author} · {date.today():%B %Y}")

    # ---------------------------------------------------------------- about
    doc.add_heading("1. What the project is about", level=1)
    para(doc, "Dentists use panoramic X-rays, a single wide image of the whole mouth, to spot problems such as tooth "
              "decay, infections at the root tip and wisdom teeth that are stuck under the gum. Reading these images takes "
              "time, and small problems are easy to miss.")
    para(doc, "This project trains an AI model that looks at a panoramic X-ray and draws a box around each problem it "
              "finds, labelled with the type of problem. Most dental AI projects stop there. This one also measures "
              "how much each prediction can be trusted, so that a dentist knows which findings are solid and which "
              "need a second look.")
    bullets(doc, [
        ("Model: ", "YOLO26, the newest version of the widely used YOLO object-detection family (Ultralytics, 2026), "
                    "fine-tuned on a laptop GPU."),
        ("Trust layer: ", "calibrated confidence scores, conformal prediction (statistically guaranteed error ranges) and "
                          "MC-dropout (asking the model the same question 20 times and checking whether it agrees with itself)."),
        ("Usable output: ", "a web app where you upload an X-ray and get the findings, plus an online version deployable on Vercel."),
    ])

    # ---------------------------------------------------------------- dataset
    doc.add_heading("2. The dataset: DENTEX", level=1)
    para(doc, "DENTEX (Dental Enumeration and Diagnosis on Panoramic X-rays) is a public benchmark released for the "
              "DENTEX challenge at MICCAI 2023, a leading medical-imaging conference. It is published on Hugging Face "
              "(ibrahimhamamci/DENTEX) under the CC BY-NC-SA 4.0 licence, which allows non-commercial research use.")
    doc.add_heading("What is in it", level=2)
    para(doc, "The X-rays are annotated hierarchically, following the FDI numbering system dentists use. Each box can "
              "carry up to three labels:")
    bullets(doc, [
        ("Quadrant: ", "which quarter of the mouth (upper right, upper left, lower left, lower right)."),
        ("Tooth number: ", "which tooth, 1 to 8, within that quadrant."),
        ("Diagnosis: ", "what is wrong with the tooth."),
    ])
    table(doc, ["DENTEX subset", "X-rays", "Labels"], [
        ["Quadrant only", "693", "quadrant"],
        ["Quadrant + tooth", "634", "quadrant, tooth number"],
        ["Fully labelled (used in this project)", "1,005", "quadrant, tooth number, diagnosis"],
        ["Unlabelled", "1,571", "none (for pre-training)"],
    ], align_right_from=1)
    doc.add_heading("The four diagnoses", level=2)
    table(doc, ["Diagnosis", "What it means"], [
        ["Impacted tooth", "A tooth that has not grown into its normal position, most often a wisdom tooth."],
        ["Caries", "Tooth decay in the outer layers of the tooth."],
        ["Deep caries", "Decay that reaches close to the nerve; usually needs a root canal or extraction."],
        ["Periapical lesion", "A dark area at the tip of the root, typically a sign of infection."],
    ], align_right_from=99)
    doc.add_heading("How the data was used", level=2)
    rows = [[name, splits[k]["images"]] + [splits[k].get(c, 0) for c in CLASSES]
            for k, name in [("train", "Training"), ("calib", "Calibration (held out)"), ("val", "Validation"),
                            ("test", "Test")]]
    table(doc, ["Split", "X-rays"] + CLASSES, rows)
    bullets(doc, [
        "The 705 official training X-rays were split into 605 for training and 100 kept aside to tune the trust layer, "
        "so it is never tuned on images the model learned from.",
        "The official 50 validation and 250 test X-rays were used for model selection and final results.",
        "Caries is by far the most common finding and periapical lesions the rarest, which makes lesions the hardest to learn.",
        "Images are large (about 2,900 × 1,300 pixels) while many problems cover only a small part of the image.",
    ])
    para(doc, "The released test labels use Turkish treatment codes (for example \"root canal\" or \"extraction\") rather "
              "than the four diagnoses. They were translated into diagnoses using a data-driven check, and the "
              "validation set, which uses the original labels, is treated as the clean benchmark.", bold_lead="Note on the test set. ")

    # ---------------------------------------------------------------- results
    doc.add_heading("3. Results", level=1)
    doc.add_heading("How accurate is the model?", level=2)
    para(doc, "AP50 is the standard detection score (0 to 1, higher is better). It rewards finding each problem with a box "
              "that overlaps the true one by at least half. Precision is the share of predicted findings that are correct; "
              "recall is the share of real problems that were found.")
    rows = []
    for c in CLASSES + ["all"]:
        v, te = cls_row(val, c), cls_row(test, c)
        rows.append(["All four" if c == "all" else c, f3(v.AP50), f3(te.AP50), pct(te.precision), pct(te.recall)])
    table(doc, ["Diagnosis", "AP50 validation", "AP50 test", "Precision (test)", "Recall (test)"], rows, bold_last=True)
    figure(doc, root / "assets/examples/00_test_64.jpg",
           "Example on an unseen X-ray. Top: dentist's annotation. Bottom: the model's finding (caries, 0.45) with "
           "its 90% error range shown as a thin outer box.", width=5.0)

    doc.add_heading("How trustworthy are its confidence scores?", level=2)
    rows = [["Confidence scores match reality (calibration error, lower is better)",
             f"{f3(c_all.ECE_raw)} before → {f3(c_all.ECE_platt)} after calibration"]]
    rows += [[f"90% error range contains the true box: {c}", pct(cls_row(conf, c).box_coverage)] for c in CLASSES]
    rows += [["Spotting wrong findings: single prediction (AUROC)", f3(get_auroc("baseline", "1 - confidence"))],
             ["Spotting wrong findings: MC-dropout, 20 passes (AUROC)", f3(get_auroc(mc_model, "1 - confidence"))]]
    table(doc, ["Trust measure", "Result"], rows, align_right_from=1)
    figure(doc, root / "results/reliability_diagram.png",
           "Before calibration the model is over-confident at high scores; after calibration its confidence matches "
           "how often it is actually right.", width=4.2)

    # ---------------------------------------------------------------- findings
    doc.add_heading("4. Key findings", level=1)
    imp = cls_row(test, "Impacted")
    findings = [
        ("Impacted teeth are reliably detected. ", f"AP50 {f3(imp.AP50)} with {pct(imp.recall)} of impacted teeth found on "
                                                   "the test set. This is close to a solved problem."),
        ("Caries and deep caries are hard to tell apart. ", "About a third of cases are confused in each direction, because "
                                                            "they are two severity levels of the same disease."),
        ("Periapical lesions are the weakest class. ", "They are rare in the data and have fuzzy edges; about one in five is "
                                                       "missed entirely."),
        ("Calibration halves the confidence error. ", f"Calibration error drops from {f3(c_all.ECE_raw)} to "
                                                      f"{f3(c_all.ECE_platt)}, so a score of 0.7 really means roughly 70%."),
        ("The error ranges keep their promise. ", "The 90% conformal boxes contained the true lesion 88–100% of the time "
                                                  "on the test set, as guaranteed."),
        ("Honest limits on sensitivity. ", "The conformal analysis shows the model can promise \"at most 10% missed\" only "
                                           "for impacted teeth. For caries and lesions no threshold reaches 90% sensitivity, "
                                           "which a simple fixed cut-off would have hidden."),
        ("MC-dropout helps flag mistakes but costs accuracy. ", f"Asking the model 20 times spots wrong findings better "
                                                                f"(AUROC {f3(get_auroc(mc_model, '1 - confidence'))} vs "
                                                                f"{f3(get_auroc('baseline', '1 - confidence'))}), but lowers "
                                                                f"AP50 ({f3(mc_ap.loc[mc_ap.index[-1], 'mean'])} vs "
                                                                f"{f3(mc_ap.loc['baseline', 'mean'])}). It works best as a "
                                                                "\"please review\" flag."),
        ("The dataset needed checking. ", "The official test labels did not match the stated diagnoses, and auditing them "
                                          "was essential for honest results."),
    ]
    for lead, text in findings:
        p = doc.add_paragraph(style="List Number")
        p.add_run(lead).bold = True
        p.add_run(text)

    # ---------------------------------------------------------------- next
    doc.add_heading("5. Limitations and next steps", level=1)
    bullets(doc, [
        "Only 605 training X-rays; more data, especially lesions, would help most.",
        "Higher-resolution or tooth-by-tooth analysis should raise sensitivity for small caries and lesions.",
        "Predicting exact tooth numbers (not just the quadrant) would make findings easier to act on.",
        "This is a research prototype, not a medical device, and must not be used for clinical decisions.",
    ])

    doc.add_heading("6. Try it", level=1)
    bullets(doc, [
        ("Local web app: ", "conda activate dentex, then python app/app.py, and open http://127.0.0.1:7860. Upload an "
                            "X-ray (examples in data/yolo/images/test/)."),
        ("Online: ", "deploy the deploy/vercel folder to Vercel (set Root Directory to deploy/vercel)."),
        ("Code and full technical report: ", "github.com/MuhammadIshaq-AI/dentex-tooth-detection, "
                                             "docs/DENTEX_Project_Report.docx."),
    ])
    para(doc, "Hamamci, I. E. et al., DENTEX: An Abnormal Tooth Detection with Dental Enumeration and Diagnosis Benchmark "
              "for Panoramic X-rays, arXiv:2305.19112 (2023). Ultralytics YOLO26, docs.ultralytics.com.",
         bold_lead="Dataset and model credits: ")

    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="docs/DENTEX_Project_Overview.docx")
    ap.add_argument("--author", default="Muhammad Ishaq")
    args = ap.parse_args()
    out = build(Path(args.root).resolve(), Path(args.out), args.author)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
