"""Monte-Carlo dropout for Ultralytics YOLO detection heads.

YOLO26 has no dropout layers, so ``inject_dropout`` wraps the final 1x1 conv of every box
and class branch in the Detect head as ``Sequential(MCDropout2d, conv)``. Existing
parameters are reused, so weights, optimizer state and checkpoints stay valid.
"""
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from dentex.labels import Boxes
from dentex.uncertainty.matching import box_iou


class MCDropout2d(nn.Dropout2d):
    """Dropout2d that can be forced on at inference via the ``mc`` flag."""

    mc = False

    def forward(self, x):
        return F.dropout2d(x, self.p, self.training or self.mc, self.inplace)


def _detect_head(model: nn.Module):
    head = model
    while hasattr(head, "model") and isinstance(getattr(head, "model"), nn.Module):
        head = head.model  # YOLO -> DetectionModel -> Sequential
    return head[-1] if isinstance(head, nn.Sequential) else head


def inject_dropout(model: nn.Module, p: float = 0.1) -> int:
    """Insert MCDropout2d before the output conv of each head branch. Returns #insertions."""
    head, n = _detect_head(model), 0
    for name, branches in head.named_children():
        if not isinstance(branches, nn.ModuleList):
            continue
        for seq in branches:
            if isinstance(seq, nn.Sequential) and isinstance(seq[-1], nn.Conv2d):
                seq[-1] = nn.Sequential(MCDropout2d(p), seq[-1])
                n += 1
    if n == 0:
        raise RuntimeError(f"no head branches found in {type(head).__name__}")
    return n


def set_mc(model: nn.Module, enabled: bool):
    for m in model.modules():
        if isinstance(m, MCDropout2d):
            m.mc = enabled


def has_dropout(model: nn.Module) -> bool:
    return any(isinstance(m, MCDropout2d) for m in model.modules())


def dropout_training_callback(p: float):
    """Ultralytics ``on_pretrain_routine_end`` callback: inject into both model and EMA."""

    def cb(trainer):
        if has_dropout(trainer.model):  # resumed from a checkpoint that already carries dropout
            return
        n = inject_dropout(trainer.model, p)
        if getattr(trainer, "ema", None) is not None:
            inject_dropout(trainer.ema.ema, p)
        print(f"[mc-dropout] inserted {n} MCDropout2d(p={p}) layers into the detection head")

    return cb


# --------------------------------------------------------------------------- sampling
def mc_predict(yolo, image_paths, T=20, imgsz=1024, conf=0.01, batch=4, iou_thr=0.5, num_classes=4):
    """Run T stochastic passes per image and fuse detections. Returns list of (Boxes, stats dict)."""
    from dentex.labels import predict

    predict(yolo, image_paths[:1], imgsz=imgsz, conf=conf, batch=1)  # build predictor
    net = yolo.predictor.model
    if not has_dropout(net):
        raise RuntimeError("model has no MCDropout2d layers; train with configs/train_mcdropout.yaml")
    set_mc(net, True)
    try:
        passes = [predict(yolo, image_paths, imgsz=imgsz, conf=conf, batch=batch) for _ in range(T)]
    finally:
        set_mc(net, False)
    return [fuse_passes([p[i] for p in passes], iou_thr, num_classes) for i in range(len(image_paths))]


def fuse_passes(dets: list[Boxes], iou_thr=0.5, num_classes=4):
    """Class-agnostic clustering of detections across T passes (at most one per pass per cluster).

    Per cluster:
      conf     = mean over T passes of the class score (0 where the pass missed it)
      freq     = fraction of passes that produced the detection
      entropy  = entropy of [per-class mean score..., background], normalised to [0, 1]
      box_std  = mean coordinate std normalised by box size
    """
    T = len(dets)
    boxes = np.concatenate([d.xyxy for d in dets]) if T else np.zeros((0, 4))
    if len(boxes) == 0:
        return Boxes.empty(with_conf=True), {k: np.zeros(0) for k in ("freq", "entropy", "box_std")}
    scores = np.concatenate([d.conf for d in dets])
    classes = np.concatenate([d.cls for d in dets])
    pass_id = np.concatenate([np.full(len(d.xyxy), t) for t, d in enumerate(dets)])

    order = np.argsort(-scores)
    iou = box_iou(boxes, boxes)
    assigned = np.full(len(boxes), -1)
    clusters = []
    for i in order:
        if assigned[i] >= 0:
            continue
        members, used = [i], {pass_id[i]}
        assigned[i] = len(clusters)
        for j in order:
            if assigned[j] < 0 and pass_id[j] not in used and iou[i, j] >= iou_thr:
                members.append(j)
                used.add(pass_id[j])
                assigned[j] = len(clusters)
        clusters.append(np.array(members))

    xyxy, cls, conf, freq, ent, bstd = [], [], [], [], [], []
    for m in clusters:
        w = scores[m] / scores[m].sum()
        mean_box = (boxes[m] * w[:, None]).sum(0)
        class_mass = np.bincount(classes[m], weights=scores[m], minlength=num_classes) / T
        dist = np.append(class_mass, max(0.0, 1 - class_mass.sum()))
        dist = dist / dist.sum()
        nz = dist[dist > 0]
        size = np.maximum(mean_box[2:] - mean_box[:2], 1e-6)
        xyxy.append(mean_box)
        cls.append(int(np.argmax(class_mass)))
        conf.append(float(class_mass.max()))
        freq.append(len(m) / T)
        ent.append(float(-(nz * np.log(nz)).sum() / np.log(len(dist))))
        bstd.append(float((boxes[m].std(0) / np.tile(size, 2)).mean()) if len(m) > 1 else 1.0)
    fused = Boxes(np.array(xyxy), np.array(cls), np.array(conf))
    return fused, {"freq": np.array(freq), "entropy": np.array(ent), "box_std": np.array(bstd)}
