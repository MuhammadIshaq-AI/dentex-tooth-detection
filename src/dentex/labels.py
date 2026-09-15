"""Load YOLO ground truth for a split and collect model predictions as numpy arrays."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from dentex.convert import yolo_to_xyxy


@dataclass
class Boxes:
    xyxy: np.ndarray            # (N, 4) pixels
    cls: np.ndarray             # (N,)
    conf: np.ndarray | None = None

    @classmethod
    def empty(cls, with_conf=False):
        return cls(np.zeros((0, 4)), np.zeros(0, int), np.zeros(0) if with_conf else None)


def load_data_yaml(path):
    d = yaml.safe_load(Path(path).read_text())
    root = Path(d["path"])
    names = {int(k): v for k, v in d["names"].items()}
    return root, d, names


def split_images(data_yaml, split):
    root, d, _ = load_data_yaml(data_yaml)
    return sorted((root / d[split]).glob("*.png"))


def load_gt(image_path) -> Boxes:
    image_path = Path(image_path)
    label = image_path.parent.parent.parent / "labels" / image_path.parent.name / (image_path.stem + ".txt")
    if not label.exists() or not label.read_text().strip():
        return Boxes.empty()
    w, h = Image.open(image_path).size
    rows = np.loadtxt(label, ndmin=2)
    xyxy = np.array([yolo_to_xyxy(*r[1:], w, h) for r in rows])
    return Boxes(xyxy, rows[:, 0].astype(int))


def predict(model, image_paths, imgsz=1024, conf=0.001, batch=4, **kw) -> list[Boxes]:
    """Run an Ultralytics model over images; returns one Boxes per image (in input order)."""
    out = []
    paths = [str(p) for p in image_paths]
    for i in range(0, len(paths), batch):
        for r in model.predict(paths[i:i + batch], imgsz=imgsz, conf=conf, verbose=False, **kw):
            b = r.boxes
            out.append(Boxes(b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int), b.conf.cpu().numpy()))
    return out
