"""Download and extract the DENTEX dataset from Hugging Face.

Dataset: https://huggingface.co/datasets/ibrahimhamamci/DENTEX (CC-BY-NC-SA 4.0).
If the dataset is gated for your account, run `huggingface-cli login` first.
"""
import argparse
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "ibrahimhamamci/DENTEX"
FILES = ["validation_triple.json", "validation_data.zip", "test_data.zip", "training_data.zip"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/raw", help="download directory")
    ap.add_argument("--skip-extract", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = out / name
        if target.exists():
            print(f"[skip] {target} already exists")
        else:
            print(f"[download] {name}")
            path = hf_hub_download(REPO_ID, f"DENTEX/{name}", repo_type="dataset", local_dir=out)
            Path(path).replace(target)

        if name.endswith(".zip") and not args.skip_extract:
            dest = out / name.removesuffix(".zip")
            if dest.exists():
                print(f"[skip] {dest} already extracted")
                continue
            print(f"[extract] {name} -> {dest}")
            with zipfile.ZipFile(target) as zf:
                zf.extractall(dest)


if __name__ == "__main__":
    main()
