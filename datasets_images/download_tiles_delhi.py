#!/usr/bin/env python3
"""Download Esri World Imagery tiles for a Delhi bbox.

Saves tiles to `datasets_images/real/delhi/z{z}/` as `{z}_{x}_{y}.jpg`.
"""
import math
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def deg2num(lat, lon, zoom):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def download_tile(z, x, y, outdir, tries=3, timeout=20):
    url = f"https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    dest = outdir / f"{z}_{x}_{y}.jpg"
    if dest.exists():
        return True
    for attempt in range(tries):
        try:
            urllib.request.urlretrieve(url, dest)
            return True
        except Exception:
            pass
    return False


def main():
    # Delhi bbox (approx) chosen to yield ~700 tiles at zoom 15
    lat_min, lat_max = 28.55, 28.85
    lon_min, lon_max = 76.95, 77.25
    z = 15

    x0, y0 = deg2num(lat_min, lon_min, z)
    x1, y1 = deg2num(lat_max, lon_max, z)
    x_min, x_max = min(x0, x1), max(x0, x1)
    y_min, y_max = min(y0, y1), max(y0, y1)

    outdir = Path("datasets_images/real/delhi") / f"z{z}"
    outdir.mkdir(parents=True, exist_ok=True)

    tiles = [(z, x, y) for x in range(x_min, x_max + 1) for y in range(y_min, y_max + 1)]
    total = len(tiles)
    print(f"Downloading {total} tiles to {outdir}")

    success = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(download_tile, z, x, y, outdir): (x, y) for (z, x, y) in tiles}
        for fut in as_completed(futures):
            ok = fut.result()
            if ok:
                success += 1

    print(f"Downloaded {success}/{total} tiles to {outdir}")


if __name__ == "__main__":
    main()
