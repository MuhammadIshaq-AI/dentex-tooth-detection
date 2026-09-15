import json

import numpy as np

from dentex.convert import coco_bbox_to_yolo, convert_split, fdi_number, yolo_to_xyxy
from dentex.splits import stratified_holdout


def test_bbox_roundtrip_within_one_pixel():
    rng = np.random.default_rng(0)
    for _ in range(100):
        W, H = 2900, 1300
        x, y = rng.uniform(0, W - 300), rng.uniform(0, H - 300)
        w, h = rng.uniform(10, 300, size=2)
        x1, y1, x2, y2 = yolo_to_xyxy(*coco_bbox_to_yolo([x, y, w, h], W, H), W, H)
        assert np.allclose([x1, y1, x2, y2], [x, y, x + w, y + h], atol=1.0)


def test_bbox_clipped_to_image():
    cx, cy, w, h = coco_bbox_to_yolo([-10, -10, 60, 60], 100, 100)
    assert np.allclose([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], [0, 0, 0.5, 0.5])


def test_fdi_number():
    quads, teeth = {2: "3"}, {5: "6"}
    assert fdi_number({"category_id_1": 2, "category_id_2": 5}, quads, teeth) == "36"


def _fake_dataset(tmp_path, n=20):
    from PIL import Image

    img_dir = tmp_path / "xrays"
    img_dir.mkdir()
    images, anns = [], []
    for i in range(n):
        Image.new("L", (200, 100)).save(img_dir / f"im_{i}.png")
        images.append({"id": i, "file_name": f"im_{i}.png", "width": 200, "height": 100})
        anns.append({"id": i, "image_id": i, "bbox": [10, 10, 50, 40],
                     "category_id_1": 0, "category_id_2": 1, "category_id_3": i % 4})
    cats = lambda names: [{"id": k, "name": v} for k, v in enumerate(names)]  # noqa: E731
    d = {"images": images, "annotations": anns, "categories_1": cats("1234"),
         "categories_2": cats("12345678"), "categories_3": cats(["Impacted", "Caries", "Periapical Lesion", "Deep Caries"])}
    jp = tmp_path / "train.json"
    jp.write_text(json.dumps(d))
    return jp, img_dir, {a["image_id"]: [a] for a in anns}


def test_labelme_conversion_maps_and_drops_codes(tmp_path):
    from PIL import Image

    from dentex.convert import convert_labelme_split

    (tmp_path / "input").mkdir(); (tmp_path / "label").mkdir()
    Image.new("L", (200, 100)).save(tmp_path / "input" / "t_1.png")
    shapes = [
        {"label": "1-çürük-15", "points": [[10, 10], [60, 10], [60, 50], [10, 50]], "shape_type": "polygon"},
        {"label": "3-kanal-36", "points": [[100, 20], [140, 20], [120, 90]], "shape_type": "polygon"},
        {"label": "2-küretaj-11", "points": [[0, 0], [5, 0], [5, 5]], "shape_type": "polygon"},
    ]
    (tmp_path / "label" / "t_1.json").write_text(json.dumps(
        {"shapes": shapes, "imagePath": "t_1.png", "imageWidth": 200, "imageHeight": 100}), encoding="utf-8")
    names = {0: "Impacted", 1: "Caries", 2: "Periapical Lesion", 3: "Deep Caries"}
    rows = []
    convert_labelme_split(tmp_path / "label", tmp_path / "input", tmp_path / "yolo", "test", names, rows)
    lines = (tmp_path / "yolo" / "labels" / "test" / "t_1.txt").read_text().split("\n")
    assert [l.split()[0] for l in lines if l] == ["1", "3"]
    assert [r["fdi"] for r in rows] == ["15", "36"]


def test_convert_split_and_holdout(tmp_path):
    jp, img_dir, anns = _fake_dataset(tmp_path)
    keep, held = stratified_holdout(list(range(20)), anns, 8, seed=0)
    assert len(held) == 8 and not set(keep) & set(held)
    rows = []
    names = convert_split(jp, img_dir, tmp_path / "yolo", "calib", held, rows)
    assert names[1] == "Caries"
    labels = sorted((tmp_path / "yolo" / "labels" / "calib").glob("*.txt"))
    assert len(labels) == 8 and len(rows) == 8
    cls, cx, cy, w, h = labels[0].read_text().split()
    assert np.allclose([float(cx), float(cy), float(w), float(h)], [35 / 200, 30 / 100, 50 / 200, 40 / 100])
    assert rows[0]["fdi"] == "12"
