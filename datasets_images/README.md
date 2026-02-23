Place dataset images here for experiments.

This folder is intentionally separate from the application data directories so it won't interfere with the project.

Guidance:
- Add subfolders like `train/`, `val/`, `test/` as needed.
- This folder is listed in `.gitignore` to avoid accidentally committing large images.

Real tiles for Delhi were downloaded into `datasets_images/real/delhi/z15/` and split into
`train/`, `val/`, and `test/` under `datasets_images/real/delhi/`.

To create paired `clean/` and `noisy/` datasets (one-to-one), run:

```
python3 datasets_images/create_clean_noisy.py
```

This script creates `clean/` and `noisy/` subfolders inside each split and places the originals
in `clean/` and generated noisy variants in `noisy/`.

Other helper scripts:
- `datasets_images/download_tiles_delhi.py` — download Esri World Imagery tiles for a Delhi bbox
- `datasets_images/split_tiles.py` — split downloaded tiles into train/val/test
- `datasets_images/generate_sample_satellite.py` — create a synthetic sample image

