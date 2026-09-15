"""Convert DENTEX hierarchical COCO annotations (diagnosis level) to YOLO format."""
import csv
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path


def load_dentex(json_path):
    """Return (images_by_id, annotations_by_image_id, diagnosis_names, quadrant_names, tooth_names)."""
    d = json.loads(Path(json_path).read_text())
    images = {im["id"]: im for im in d["images"]}
    anns = defaultdict(list)
    for a in d["annotations"]:
        anns[a["image_id"]].append(a)
    names = lambda key: {c["id"]: c["name"] for c in d.get(key, [])}  # noqa: E731
    return images, anns, names("categories_3"), names("categories_1"), names("categories_2")


def coco_bbox_to_yolo(bbox, img_w, img_h):
    """COCO [x, y, w, h] (pixels) -> YOLO (cx, cy, w, h) normalised, clipped to the image."""
    x, y, w, h = bbox
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(img_w), x + w), min(float(img_h), y + h)
    return ((x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h, (x2 - x1) / img_w, (y2 - y1) / img_h)


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    return ((cx - w / 2) * img_w, (cy - h / 2) * img_h, (cx + w / 2) * img_w, (cy + h / 2) * img_h)


def fdi_number(ann, quadrant_names, tooth_names):
    """FDI two-digit tooth number, e.g. quadrant '3' + tooth '6' -> '36' (empty if unavailable)."""
    q, t = quadrant_names.get(ann.get("category_id_1")), tooth_names.get(ann.get("category_id_2"))
    return f"{q}{t}" if q is not None and t is not None else ""


def _place_image(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)  # hardlink: no extra disk usage on the same volume
    except OSError:
        shutil.copy2(src, dst)


def convert_split(json_path, img_dir, out_root, split, image_ids=None, sidecar_rows=None):
    """Write images/<split> and labels/<split> for the given image ids (default: all).

    Returns the diagnosis class-name mapping {id: name}.
    """
    images, anns, diag, quads, teeth = load_dentex(json_path)
    out_root, img_dir = Path(out_root), Path(img_dir)
    ids = images.keys() if image_ids is None else image_ids
    for iid in ids:
        im = images[iid]
        src = img_dir / im["file_name"]
        if not src.exists():
            raise FileNotFoundError(src)
        _place_image(src, out_root / "images" / split / im["file_name"])
        lines = []
        for a in anns.get(iid, []):
            cls = a["category_id_3"]
            cx, cy, w, h = coco_bbox_to_yolo(a["bbox"], im["width"], im["height"])
            if w <= 0 or h <= 0:
                continue
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            if sidecar_rows is not None:
                x, y, bw, bh = a["bbox"]
                sidecar_rows.append({
                    "split": split, "image": im["file_name"], "class_id": cls, "class_name": diag[cls],
                    "fdi": fdi_number(a, quads, teeth), "x1": x, "y1": y, "x2": x + bw, "y2": y + bh,
                })
        label = out_root / "labels" / split / (Path(im["file_name"]).stem + ".txt")
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("\n".join(lines) + ("\n" if lines else ""))
    return diag


# The released DENTEX test labels are per-image LabelMe polygons labelled "<code>-<turkish name>-<FDI>"
# using a wider clinical vocabulary. Only the four challenge diagnoses are kept; the rest
# (0 saglam/healthy, 3 kanal/root canal, 5 cekim/extraction, 8 kirik/fracture) are dropped.
# "2-kuretaj" (curettage) is the treatment for deep caries and matches its class frequency.
LABELME_CODE_TO_DIAGNOSIS = {1: "Caries", 2: "Deep Caries", 6: "Impacted", 7: "Periapical Lesion"}


def parse_labelme_label(label: str):
    """'1-çürük-15' -> (1, '15')."""
    parts = label.split("-")
    return int(parts[0]), parts[-1]


def convert_labelme_split(label_dir, img_dir, out_root, split, diag_names, sidecar_rows=None,
                          code_map=LABELME_CODE_TO_DIAGNOSIS):
    """Convert per-image LabelMe polygon JSONs to YOLO boxes (bbox = polygon extent)."""
    name_to_id = {v: k for k, v in diag_names.items()}
    code_to_id = {code: name_to_id[name] for code, name in code_map.items()}
    out_root, img_dir = Path(out_root), Path(img_dir)
    for jp in sorted(Path(label_dir).glob("*.json")):
        d = json.loads(jp.read_text(encoding="utf-8"))
        fname = Path(d.get("imagePath") or jp.with_suffix(".png").name).name
        src = img_dir / fname
        if not src.exists():
            raise FileNotFoundError(src)
        W, H = d["imageWidth"], d["imageHeight"]
        _place_image(src, out_root / "images" / split / fname)
        lines = []
        for s in d["shapes"]:
            code, fdi = parse_labelme_label(s["label"])
            if code not in code_to_id:
                continue
            pts = [(x, y) for x, y in s["points"]]
            x1, y1 = min(p[0] for p in pts), min(p[1] for p in pts)
            x2, y2 = max(p[0] for p in pts), max(p[1] for p in pts)
            cls = code_to_id[code]
            cx, cy, w, h = coco_bbox_to_yolo([x1, y1, x2 - x1, y2 - y1], W, H)
            if w <= 0 or h <= 0:
                continue
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            if sidecar_rows is not None:
                sidecar_rows.append({"split": split, "image": fname, "class_id": cls, "class_name": diag_names[cls],
                                     "fdi": fdi, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        label = out_root / "labels" / split / (Path(fname).stem + ".txt")
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("\n".join(lines) + ("\n" if lines else ""))


def write_sidecar(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "image", "class_id", "class_name", "fdi", "x1", "y1", "x2", "y2"])
        w.writeheader()
        w.writerows(rows)
