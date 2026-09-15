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

### Per-class detection (official DENTEX test set)

<!-- results/metrics_test.md -->
_Pending training run._

### Example outputs

Top: ground truth. Bottom: prediction, with the thin outer box showing the 90% conformal box interval.

<!-- assets/examples/*.jpg -->

### Uncertainty

| | |
|---|---|
| ![reliability](results/reliability_diagram.png) | ![coverage](results/conformal_coverage.png) |

_Pending._

---

## Method

### Data
- **train / calib**: the 705 fully-annotated DENTEX training images, split (stratified by rarest class) into
  ~605 for training and **100 held out for calibration**. Calibration images are never trained on.
- **val**: official 50-image validation set (model selection).
- **test**: official 250-image test set. Its labels are released as LabelMe polygons with a wider clinical
  vocabulary. They are converted to boxes and mapped `çürük → Caries`, `küretaj → Deep Caries`,
  `gömülü → Impacted` and `lezyon → Periapical Lesion`; non-diagnosis codes (healthy, root canal,
  extraction, fracture) are dropped. See `src/dentex/convert.py`.

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
