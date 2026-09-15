"""Build the YOLO dataset (train / calib / val / test) from the raw DENTEX download.

Splits:
  train  - official diagnosis training set minus the calibration holdout
  calib  - held out from training; used only to fit calibration / conformal prediction
  val    - official validation set (model selection)
  test   - official test set if its labels are available, otherwise a train holdout
"""
import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path

import yaml

from dentex.convert import (
    LABELME_CODE_TO_DIAGNOSIS, convert_labelme_split, convert_split, load_dentex, write_sidecar,
)
from dentex.splits import stratified_holdout


def extract_if_needed(raw: Path, name: str) -> Path:
    dest = raw / name
    z = raw / f"{name}.zip"
    if not dest.exists() and z.exists():
        print(f"[extract] {z}")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest)
    return dest


def find_disease_set(root: Path, json_hint: Path | None = None):
    """Locate the diagnosis-level annotation JSON and its xrays directory under ``root``."""
    jsons = [json_hint] if json_hint and json_hint.exists() else [
        p for p in root.rglob("*.json") if "disease" in p.name.lower() or "triple" in p.name.lower()
    ]
    img_dirs = [
        p for p in root.rglob("*") if p.is_dir() and "disease" in str(p).lower()
        and ".ipynb_checkpoints" not in str(p) and any(p.glob("*.png"))
    ]
    if not jsons or not img_dirs:
        return None, None
    # prefer an image dir containing the files referenced by the json
    images, *_ = load_dentex(jsons[0])
    first = next(iter(images.values()))["file_name"]
    img_dir = next((d for d in img_dirs if (d / first).exists()), img_dirs[0])
    return jsons[0], img_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/yolo")
    ap.add_argument("--calib", type=int, default=100, help="images held out for calibration")
    ap.add_argument("--test-frac", type=float, default=0.15, help="fallback test holdout fraction")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    train_root = extract_if_needed(raw, "training_data")
    val_root = extract_if_needed(raw, "validation_data")
    test_root = extract_if_needed(raw, "test_data")

    train_json, train_imgs = find_disease_set(train_root)
    val_json, val_imgs = find_disease_set(val_root, raw / "validation_triple.json")
    test_labels = next((p for p in test_root.rglob("label") if p.is_dir()), None) if test_root.exists() else None
    test_imgs = next((p for p in test_root.rglob("input") if p.is_dir()), None) if test_root.exists() else None
    if train_json is None or val_json is None:
        raise SystemExit("Could not locate diagnosis annotations - check data/raw contents.")
    print(f"train: {train_json} | {train_imgs}\nval:   {val_json} | {val_imgs}\ntest:  {test_labels} | {test_imgs}")

    images, anns, names, *_ = load_dentex(train_json)
    ids = list(images)
    notes = {}
    use_official_test = test_labels is not None and test_imgs is not None and any(test_labels.glob("*.json"))
    if use_official_test:
        notes["test"] = ("official DENTEX test set (LabelMe polygons -> boxes); codes mapped "
                         + ", ".join(f"{k}->{v}" for k, v in LABELME_CODE_TO_DIAGNOSIS.items())
                         + "; non-diagnosis codes dropped")
    else:
        ids, test_ids = stratified_holdout(ids, anns, args.test_frac, seed=args.seed)
        notes["test"] = f"official test labels unavailable; {len(test_ids)} images held out from training set"
    ids, calib_ids = stratified_holdout(ids, anns, args.calib, seed=args.seed)

    rows = []
    convert_split(train_json, train_imgs, out, "train", ids, rows)
    convert_split(train_json, train_imgs, out, "calib", calib_ids, rows)
    convert_split(val_json, val_imgs, out, "val", None, rows)
    if use_official_test:
        convert_labelme_split(test_labels, test_imgs, out, "test", names, rows)
    else:
        convert_split(train_json, train_imgs, out, "test", test_ids, rows)
    write_sidecar(rows, out / "annotations_fdi.csv")

    data_yaml = {
        "path": str(out.resolve()),
        "train": "images/train", "val": "images/val", "test": "images/test", "calib": "images/calib",
        "names": {int(k): v for k, v in sorted(names.items())},
    }
    (out / "dentex.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))

    summary = {}
    for split in ["train", "calib", "val", "test"]:
        n_img = len(list((out / "images" / split).glob("*.png")))
        c = Counter(r["class_name"] for r in rows if r["split"] == split)
        summary[split] = {"images": n_img, **dict(sorted(c.items()))}
        print(f"{split:6s} images={n_img:4d} " + " ".join(f"{k}={v}" for k, v in sorted(c.items())))
    (out / "split_summary.json").write_text(json.dumps({"summary": summary, "notes": notes}, indent=2))
    if notes:
        print("NOTE:", notes)


if __name__ == "__main__":
    main()
