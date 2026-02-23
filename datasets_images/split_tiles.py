#!/usr/bin/env python3
"""Split downloaded tiles into train/val/test folders.

Usage: python3 datasets_images/split_tiles.py
"""
import random
from pathlib import Path
import shutil


def main(seed=42, train_ratio=0.8, val_ratio=0.1):
    src = Path("datasets_images/real/delhi/z15")
    if not src.exists():
        print(f"Source directory not found: {src}")
        return

    files = sorted([p for p in src.iterdir() if p.suffix.lower() in (".jpg", ".png", ".jpeg")])
    n = len(files)
    if n == 0:
        print(f"No image files found in {src}")
        return

    random.seed(seed)
    random.shuffle(files)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    base = src.parent  # datasets_images/real/delhi
    dst_train = base / "train"
    dst_val = base / "val"
    dst_test = base / "test"
    for d in (dst_train, dst_val, dst_test):
        d.mkdir(parents=True, exist_ok=True)

    for i, p in enumerate(files):
        if i < n_train:
            dst = dst_train
        elif i < n_train + n_val:
            dst = dst_val
        else:
            dst = dst_test
        shutil.copy2(p, dst / p.name)

    print(f"Total files: {n}")
    print(f"Train: {n_train} -> {dst_train}")
    print(f"Val:   {n_val} -> {dst_val}")
    print(f"Test:  {n_test} -> {dst_test}")


if __name__ == "__main__":
    main()
