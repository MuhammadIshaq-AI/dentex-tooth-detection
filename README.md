# dentex-tooth-detection

**Uncertainty-aware dental diagnosis detection on panoramic X-rays.**
Fine-tunes Ultralytics **YOLO26** (the latest YOLO release) on the
[DENTEX](https://huggingface.co/datasets/ibrahimhamamci/DENTEX) diagnosis set. It then adds
what most public dental-detection repos don't have: **calibrated confidence, conformal
guarantees, and MC-dropout uncertainty**.

| Diagnosis classes | Impacted · Caries · Periapical Lesion · Deep Caries |
|---|---|
| Detector | YOLO26-s, 1024 px, trained locally on an RTX 3050 Laptop (6 GB) |
| Uncertainty | Platt calibration · split-conformal box intervals · recall-controlling conformal thresholds · MC-dropout |

> Results tables and example images below are filled in by the pipeline (`results/`, `assets/examples/`).

---

## Results

### Per-class detection

YOLO26-s, 1024 px, best epoch 43 of 73 (early stopping, patience 30). Trained in 58 min on an RTX 3050 Laptop (6 GB).

**Official validation set** (50 images, original DENTEX diagnosis labels):

| Class | Precision | Recall | AP50 | AP50-95 |
|---|---:|---:|---:|---:|
| Impacted | 0.742 | 0.865 | 0.897 | 0.593 |
| Caries | 0.455 | 0.465 | 0.449 | 0.308 |
| Periapical Lesion | 0.665 | 0.442 | 0.491 | 0.294 |
| Deep Caries | 0.508 | 0.656 | 0.640 | 0.462 |
| **All** | **0.593** | **0.607** | **0.619** | **0.415** |

**Official test set** (250 images, treatment-code labels mapped to diagnoses; see [label audit](#test-label-audit)):

| Class | Precision | Recall | AP50 | AP50-95 |
|---|---:|---:|---:|---:|
| Impacted | 0.853 | 0.928 | 0.932 | 0.586 |
| Caries | 0.497 | 0.427 | 0.482 | 0.335 |
| Periapical Lesion | 0.567 | 0.373 | 0.431 | 0.264 |
| Deep Caries | 0.491 | 0.532 | 0.434 | 0.271 |
| **All** | **0.602** | **0.565** | **0.569** | **0.364** |

Inference: ~8 ms per image at 1024 px on the laptop GPU.

#### Where the errors are

| PR curve (test) | Confusion matrix (test, normalised by true class) |
|---|---|
| ![pr](results/pr_curve_test.png) | ![cm](results/confusion_matrix_test.png) |

- **Impacted** teeth are nearly solved (AP50 0.93) and are never lost to background.
- **Caries ↔ Deep Caries** is the dominant confusion (~34% each way). These are adjacent severity grades of the same
  disease, and the boundary is also where the test set's treatment-code labels differ most from DENTEX's.
- **Periapical lesions** are the rarest class (134 training boxes). 19% are missed entirely, which matches their low recall
  ceiling in the conformal analysis below.

Validation-set versions: [PR curve](results/pr_curve_val.png) · [confusion matrix](results/confusion_matrix_val.png).

### Example outputs

Top: ground truth. Bottom: prediction, with the thin outer box showing the 90% conformal box interval.

<!-- assets/examples/*.jpg -->

### Uncertainty

| | |
|---|---|
| ![reliability](results/reliability_diagram.png) | ![coverage](results/conformal_coverage.png) |

All uncertainty components are **fitted on the 100-image `calib` split** (never trained on) and **evaluated on test**.

#### 1. Confidence calibration: per-class Platt scaling

Expected calibration error of detections (conf ≥ 0.05; "correct" = matched at IoU ≥ 0.5):

| Class | Detections | ECE raw | ECE calibrated |
|---|---:|---:|---:|
| Impacted | 272 | 0.164 | 0.103 |
| Caries | 1448 | 0.081 | 0.032 |
| Periapical Lesion | 112 | 0.117 | 0.099 |
| Deep Caries | 318 | 0.108 | 0.093 |
| **All** | **2150** | **0.070** | **0.037** |

Calibration **halves the ECE**. The raw model is over-confident at the top end: detections scored ~0.9 are right only ~67% of the time, which Platt scaling corrects.

#### 2. Conformal box intervals (α = 0.1)

Each detection gets an inner/outer box expanded by `q × box size`. Split-conformal theory says the true box
lies between them for ≥ 90% of detected lesions.

| Class | Margin q | Test coverage | Matched boxes |
|---|---:|---:|---:|
| Impacted | 0.16 | 88.2% | 212 |
| Caries | 0.14 | 92.9% | 504 |
| Periapical Lesion | 0.63 | 100% | 36 |
| Deep Caries | 0.24 | 96.5% | 115 |

Coverage holds within sampling error even though test labels come from a different annotation protocol. Box
uncertainty is largest for periapical lesions, which have diffuse boundaries.

#### 3. Recall-controlling thresholds: "miss at most α of lesions"

A per-class confidence threshold fitted so that ≥ 1 − α of calib lesions are kept. This is only **feasible** when the
detector finds at least that many lesions at all (its recall ceiling, measured on calib at conf ≥ 0.05, IoU ≥ 0.5):

| Class | Calib recall ceiling | α = 0.1 (target 90%) | α = 0.5 (target 50%) | Fixed conf 0.25 |
|---|---:|---|---|---|
| Impacted | 95.4% | thr 0.40 → **recall 88.7%, precision 87.1%** | thr 0.75 → recall 48.0%, precision 93.8% | recall 91.4%, precision 85.6% |
| Caries | 66.3% | infeasible | thr 0.17 → recall 50.2%, precision 47.1% | recall 43.0%, precision 50.3% |
| Periapical Lesion | 54.2% | infeasible | thr 0.07 → recall 42.7%, precision 35.6% | recall 37.3%, precision 54.9% |
| Deep Caries | 55.2% | infeasible | thr 0.26 → recall 54.2%, precision 51.5% | recall 55.3%, precision 51.0% |

**Takeaway.** The conformal analysis makes explicit what a fixed 0.25 threshold hides. For impacted teeth you can
promise "≤ 10% missed" at 87% precision. For caries and lesions, **no threshold can deliver 90% sensitivity with this
detector**, because about a third or more of those lesions are never localised at IoU ≥ 0.5. That points to the next
improvements: higher resolution or tiling for small lesions, and tooth-level crops. Periapical lesions fall short even at α = 0.5
(42.7% vs 50%). With only 24 calib instances the finite-sample guarantee is loose, and the test protocol shift matters more.

#### 4. MC-dropout

_Pending._

---

## Method

### Data
- **train / calib**: the 705 fully-annotated DENTEX training images, split (stratified by rarest class) into
  ~605 for training and **100 held out for calibration**. Calibration images are never trained on.
- **val**: official 50-image validation set (model selection).
- **test**: official 250-image test set. Its labels are released as LabelMe polygons with Turkish
  **treatment-planning codes**, not the four challenge diagnoses. They are converted to boxes and mapped using the audit below.

#### Test-label audit

`scripts/audit_test_labels.py` matches every test polygon (all codes) to the best-overlapping baseline prediction
(conf ≥ 0.25, IoU ≥ 0.3) and tabulates which class the model predicts (% of shapes):

| Test code | Shapes | Impacted | Caries | Periapical | Deep Caries | No detection | Mapped to |
|---|---:|---:|---:|---:|---:|---:|---|
| 6 gömülü (impacted) | 221 | **91.9** | 1.4 | 0.0 | 1.4 | 5.4 | Impacted |
| 1 çürük (caries) | 747 | 0.9 | **43.4** | 0.3 | 5.6 | 49.8 | Caries |
| 7 lezyon (lesion) | 75 | 0.0 | 10.7 | **36.0** | 17.3 | 36.0 | Periapical Lesion |
| 3 kanal (root canal) | 161 | 1.9 | 19.3 | 6.8 | **49.1** | 23.0 | Deep Caries |
| 5 çekim (extraction) | 29 | 0.0 | 0.0 | 10.3 | **65.5** | 24.1 | Deep Caries |
| 2 küretaj (curettage) | 265 | 0.0 | 5.3 | 0.4 | 2.3 | **92.1** | dropped |
| 0 sağlam (healthy) | 91 | 2.2 | 5.5 | 0.0 | 0.0 | **92.3** | dropped |
| 8 kırık (fracture) | 11 | 0.0 | 0.0 | 0.0 | 9.1 | **90.9** | dropped |

Clinically this is consistent: deep caries is what gets root-canal treatment or extraction, while küretaj is
periodontal curettage. **Caveat:** the Deep Caries mapping was chosen after looking at test-set predictions, so treat
test numbers for that class as optimistic. The validation set, which uses the original labels, is the clean reference.

### Uncertainty quantification (`src/dentex/uncertainty/`)
1. **Confidence calibration.** Per-class Platt scaling fitted on `calib`; ECE and reliability diagram on `test`.
2. **Conformal box intervals.** Nonconformity = max |pred − GT| coordinate residual normalised by box size.
   The finite-sample (1−α) quantile per class gives inner/outer boxes that contain the true box for ≥ 90% of
   detected lesions (exchangeability assumption).
3. **Recall-controlling thresholds.** Per class, the confidence threshold that misses ≤ α of lesions on `calib`
   (with finite-sample correction), e.g. *"at most 10% of caries missed"*. This replaces an arbitrary 0.25 cut-off.
4. **MC-dropout.** YOLO26 has no dropout, so `Dropout2d(p=0.1)` is inserted before every head output conv and the model is
   briefly fine-tuned with it. At inference, T=20 stochastic passes are clustered into detections with mean score,
   detection frequency, predictive entropy and box std. Reported as AUROC for flagging false positives.

---

## Reproduce

```bash
conda create -n dentex python=3.12 -y && conda activate dentex
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt && pip install -e .

python scripts/download_data.py                 # ~12 GB from Hugging Face
python scripts/prepare_data.py                  # -> data/yolo (train/calib/val/test)
python -m dentex.train --config configs/train.yaml
python -m dentex.evaluate --weights runs/dentex/yolo26s_baseline/weights/best.pt --split test
python -m dentex.train --config configs/train_mcdropout.yaml
python scripts/run_uncertainty.py \
    --weights runs/dentex/yolo26s_baseline/weights/best.pt \
    --mc-weights runs/dentex/yolo26s_mcdropout/weights/best.pt
pytest
```

## Try it

### Where the data is

After `download_data.py` and `prepare_data.py`:

| Path | Contents |
|---|---|
| `data/raw/` | original DENTEX zips and extracted folders |
| `data/yolo/images/{train,calib,val,test}/` | converted images per split (labels in `data/yolo/labels/`) |
| `data/yolo/dentex.yaml` | Ultralytics dataset config |
| `data/yolo/images/test/*.png` | 250 unseen panoramic X-rays, good for trying the app |

### Local web app (Gradio, uses your GPU)

```bash
python app/app.py        # open http://127.0.0.1:7860
```

Upload a panoramic X-ray, or pick a test-set example. The app returns:
- the annotated image
- a findings table: diagnosis, calibrated confidence, approximate FDI quadrant, status, 90% conformal box interval
- a plain-language summary

Turn on **MC-dropout** to add detection frequency and diagnosis entropy across stochastic passes. It also flags
findings where the dropout model disagrees on the diagnosis.

### Deploy to Vercel

Vercel can't host PyTorch (the Python function bundle limit is 500 MB and there's no GPU), so `deploy/vercel/` contains a torch-free
build: FastAPI + ONNX Runtime with the same preprocessing, calibration and conformal intervals, plus a static upload page.

```bash
python scripts/export_onnx.py      # writes deploy/vercel/model/{dentex_yolo26s.onnx, meta.json}
cd deploy/vercel && vercel --prod  # or import the repo in Vercel with Root Directory = deploy/vercel
```

See [`deploy/vercel/README.md`](deploy/vercel/README.md) for details.

## Repository layout
```
configs/            training configs (baseline, MC-dropout fine-tune)
scripts/            download, prepare, uncertainty evaluation
src/dentex/         conversion, splits, training, evaluation, visualisation
src/dentex/uncertainty/  matching, conformal prediction + calibration, MC-dropout
tests/              unit tests (conversion, conformal guarantees, matching)
results/  assets/   generated metrics, plots and example predictions
```

## Limitations
- Small dataset (~600 training images) and strong class imbalance (few periapical lesions).
- Conformal guarantees are marginal (per class, on average over images) and assume calibration and test images are
  exchangeable. The test set comes from the same challenge but was annotated with a different label format.
- Research code. **Not a medical device** and not for clinical use.

## Citation
```bibtex
@article{hamamci2023dentex,
  title   = {DENTEX: An Abnormal Tooth Detection with Dental Enumeration and Diagnosis Benchmark for Panoramic X-rays},
  author  = {Hamamci, Ibrahim Ethem and Er, Sezgin and Simsar, Enis and others},
  journal = {arXiv preprint arXiv:2305.19112},
  year    = {2023}
}
```
Detector: [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/). Code: MIT. Data: CC BY-NC-SA 4.0.
