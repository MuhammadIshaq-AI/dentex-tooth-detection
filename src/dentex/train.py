"""Fine-tune YOLO26 on DENTEX diagnoses.

    python -m dentex.train --config configs/train.yaml
    python -m dentex.train --config configs/train.yaml --set epochs=1 fraction=0.1 name=smoke
    python -m dentex.train --config configs/train_mcdropout.yaml   # dropout fine-tune for MC-dropout
"""
import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def parse_overrides(pairs):
    return {k: yaml.safe_load(v) for k, v in (p.split("=", 1) for p in pairs)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text()) | parse_overrides(args.set)
    dropout_p = cfg.pop("mc_dropout", None)
    model = YOLO(cfg.pop("model"))

    if dropout_p:
        from dentex.uncertainty.mc_dropout import dropout_training_callback

        model.add_callback("on_pretrain_routine_end", dropout_training_callback(float(dropout_p)))

    model.train(**cfg)


if __name__ == "__main__":
    main()
