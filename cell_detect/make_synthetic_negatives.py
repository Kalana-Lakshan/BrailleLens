"""Generate synthetic hard-negative images for the cell detector.

The detector has never seen a non-braille image during training (every
training image is a tightly-cropped braille page), so it fires confidently
on structured look-alikes it happens to resemble -- most concretely, a
real Sri Lankan ID card's barcode/QR grid (dense small dark marks on white,
tiled in rows) and a VS Code editor's text (also dense marks arranged in
rows, same reason the geometric "drop isolated detections" filter in
cell_detector_service.dart can't catch it: these aren't isolated, they
cluster in plausible rows just like real cells do).

This script produces two categories entirely programmatically, since
they're structured/synthetic by nature and don't need a real camera:

    grid   Random black/white module blocks, mimicking a barcode/QR code
           (the ID-card false-positive case).
    text   Rows of random monospace-ish characters on a light background,
           mimicking a code editor or printed document (the VS Code
           false-positive case).

Other false-positive categories seen in testing (keyboards, faces, screens
showing non-text UI, generic room clutter) are NOT covered here -- those
need real photos of real objects/people, which this script deliberately
does not fabricate. Add those to the same --out-dir manually (any
.jpeg/.jpg/.png file, any filename) before passing it to
gold_finetune_2.py's --negative-dir.

    py -3.11 -m cell_detect.make_synthetic_negatives
"""

from __future__ import annotations

import argparse
import random
import string
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = ROOT / "cell_detect" / "datasets" / "hard_negatives"

IMG_W, IMG_H = 1280, 960


def make_grid_pattern(rng: random.Random, w: int = IMG_W, h: int = IMG_H) -> np.ndarray:
    """One or more blocks of random black/white modules on a light
    background -- mimics a barcode/QR code grid (see module docstring)."""
    bg = rng.randint(200, 245)
    img = np.full((h, w), bg, dtype=np.uint8)

    n_blocks = rng.randint(1, 6)
    block_h = rng.randint(int(h * 0.25), int(h * 0.55))
    block_w_total = int(w * rng.uniform(0.5, 0.85))
    gap = rng.randint(4, 20)
    block_w = max(10, (block_w_total - gap * (n_blocks - 1)) // max(n_blocks, 1))
    y0 = rng.randint(0, h - block_h)
    x_cursor = rng.randint(0, max(1, w - (block_w * n_blocks + gap * (n_blocks - 1))))

    module_size = rng.randint(4, 10)
    for _ in range(n_blocks):
        nx = max(1, block_w // module_size)
        ny = max(1, block_h // module_size)
        block = (np.random.rand(ny, nx) > 0.5).astype(np.uint8) * 255
        block_img = cv2.resize(block, (block_w, block_h), interpolation=cv2.INTER_NEAREST)
        img[y0:y0 + block_h, x_cursor:x_cursor + block_w] = block_img
        x_cursor += block_w + gap

    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    noise = np.random.normal(0, rng.uniform(2, 8), bgr.shape).astype(np.int16)
    bgr = np.clip(bgr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return bgr


def make_text_pattern(rng: random.Random, w: int = IMG_W, h: int = IMG_H) -> np.ndarray:
    """Rows of random monospace-ish characters -- mimics a code editor or
    printed document (dense marks arranged in rows, same as real cells)."""
    bg = rng.randint(225, 250)
    img = np.full((h, w, 3), bg, dtype=np.uint8)

    font = rng.choice([cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_PLAIN])
    font_scale = rng.uniform(0.4, 0.7)
    thickness = 1
    line_height = int(30 * font_scale * 1.8)
    x_margin = rng.randint(10, 60)
    y = rng.randint(20, 40)
    charset = string.ascii_letters + string.digits + "   (){}[]<>=+-_.,;:\"'"

    while y < h - 10:
        indent = rng.choice([0, 0, 0, 20, 40, 60])
        length = rng.randint(10, 70)
        text = "".join(rng.choice(charset) for _ in range(length))
        gray = rng.randint(20, 90)
        cv2.putText(
            img, text, (x_margin + indent, y), font, font_scale,
            (gray, gray, gray), thickness, cv2.LINE_AA,
        )
        y += line_height

    return img


def generate(out_dir: Path, n_grid: int, n_text: int, seed: int = 0) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    np.random.seed(seed)
    count = 0
    for i in range(n_grid):
        img = make_grid_pattern(rng)
        cv2.imwrite(str(out_dir / f"synthetic-grid-{i:03d}.jpeg"), img)
        count += 1
    for i in range(n_text):
        img = make_text_pattern(rng)
        cv2.imwrite(str(out_dir / f"synthetic-text-{i:03d}.jpeg"), img)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic hard-negative images")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-grid", type=int, default=15, help="Barcode/QR-like grid images")
    parser.add_argument("--n-text", type=int, default=15, help="Code/text-like images")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    count = generate(args.out_dir, args.n_grid, args.n_text, args.seed)
    print(f"Wrote {count} synthetic negative images to {args.out_dir}")
    print("Add real photos (keyboards, screens, faces, generic clutter) to the same "
          "directory before passing it to gold_finetune_2.py's --negative-dir -- "
          "these two categories alone don't cover every false-positive case seen.")


if __name__ == "__main__":
    main()
