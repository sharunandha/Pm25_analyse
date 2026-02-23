#!/usr/bin/env python3
"""Create `clean/` and `noisy/` copies for train/val/test.

For each image in `datasets_images/real/delhi/{train,val,test}`, this script
copies the original into `clean/` and writes a noisy version into `noisy/`.
"""
from pathlib import Path
import shutil
import numpy as np
from PIL import Image, ImageFilter


def make_noisy(img: Image.Image, sigma=25, blur_radius=0.8):
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    arr += noise
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    noisy = Image.fromarray(arr)
    if blur_radius > 0:
        noisy = noisy.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return noisy


def process_split(src_dir: Path):
    images = sorted([p for p in src_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    if not images:
        print(f"No images in {src_dir}")
        return 0, 0

    dst_clean = src_dir.parent / 'train' if False else None  # placeholder
    # Actual destination structure: same parent (delhi) with train/val/test subfolders
    split_name = src_dir.name
    base = src_dir.parent
    dst_clean = base / split_name / 'clean'
    dst_noisy = base / split_name / 'noisy'
    dst_clean.mkdir(parents=True, exist_ok=True)
    dst_noisy.mkdir(parents=True, exist_ok=True)

    for p in images:
        shutil.copy2(p, dst_clean / p.name)
        try:
            img = Image.open(p).convert('RGB')
            noisy = make_noisy(img)
            noisy.save(dst_noisy / p.name)
        except Exception:
            # if processing fails, copy original as fallback
            shutil.copy2(p, dst_noisy / p.name)

    return len(images), len(images)


def main():
    root = Path('datasets_images/real/delhi')
    if not root.exists():
        print('Expected folder datasets_images/real/delhi not found')
        return

    splits = ['train', 'val', 'test']
    total = 0
    for s in splits:
        src = root / s
        if not src.exists():
            print(f"Skipping missing split: {src}")
            continue
        n_clean, n_noisy = process_split(src)
        print(f"Processed {s}: clean={n_clean}, noisy={n_noisy}")
        total += n_clean

    print(f"Done. Total images processed: {total}")


if __name__ == '__main__':
    main()
