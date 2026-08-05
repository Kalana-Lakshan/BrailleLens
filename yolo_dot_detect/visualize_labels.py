"""Preview a few converted YOLO labels overlaid on page images.

Usage:
    py -3.11 -m yolo_dot_detect.visualize_labels --n 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _draw_yolo_labels(img_path: Path, lbl_path: Path):
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    if not lbl_path.exists():
        return bgr
    for line in lbl_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, xc, yc, bw, bh = map(float, parts[:5])
        x0 = int((xc - bw / 2) * w)
        y0 = int((yc - bh / 2) * h)
        x1 = int((xc + bw / 2) * w)
        y1 = int((yc + bh / 2) * h)
        cv2.rectangle(bgr, (x0, y0), (x1, y1), (0, 220, 80), 1)
        cv2.circle(bgr, (int(xc * w), int(yc * h)), 2, (0, 80, 255), -1)
    return bgr


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=here / "datasets" / "braille_dots",
    )
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument(
        "--out",
        type=Path,
        default=here / "datasets" / "label_previews",
    )
    args = parser.parse_args()

    img_dir = args.dataset / "images" / args.split
    lbl_dir = args.dataset / "labels" / args.split
    if not img_dir.exists():
        raise SystemExit(f"Missing {img_dir} - run prepare_dataset first")

    args.out.mkdir(parents=True, exist_ok=True)
    images = sorted(img_dir.glob("*.jpg"))[: args.n]
    for img_path in images:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        vis = _draw_yolo_labels(img_path, lbl_path)
        if vis is None:
            continue
        # downscale for quick viewing
        scale = min(1.0, 1280 / vis.shape[1])
        if scale < 1.0:
            vis = cv2.resize(vis, None, fx=scale, fy=scale)
        out = args.out / f"{img_path.stem}_labels.png"
        cv2.imwrite(str(out), vis)
        n_boxes = len(lbl_path.read_text().splitlines()) if lbl_path.exists() else 0
        print(f"  {out.name}: {n_boxes} boxes")
    print(f"Wrote {len(images)} previews -> {args.out}")


if __name__ == "__main__":
    main()
