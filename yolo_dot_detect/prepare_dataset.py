"""Step 1 — Convert DSBI page annotations into YOLO detection labels.

DSBI stores Braille geometry as vertical/horizontal grid line positions plus
per-cell 6-bit raised-dot patterns. This script turns every *raised* dot into
a YOLO bounding box:

    class_id  x_center  y_center  width  height   (all normalized to [0, 1])

Single class: 0 = braille_dot

Usage (from repo root):
    py -3.11 -m yolo_dot_detect.prepare_dataset \\
        --dbsi-root "data DBSI/data" \\
        --out yolo_dot_detect/datasets/braille_dots
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml
from PIL import Image


# Braille cell layout (same convention as braille_cnn):
#   d1 d4
#   d2 d5
#   d3 d6
DOT_OFFSETS = [
    (0, 0),  # d1
    (0, 1),  # d2
    (0, 2),  # d3
    (1, 0),  # d4
    (1, 1),  # d5
    (1, 2),  # d6
]


def _parse_split_file(split_path: Path):
    entries = []
    with open(split_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            book, filename = line.replace("/", "\\").split("\\")
            base = filename[:-4] if filename.lower().endswith(".jpg") else filename
            entries.append((book, base))
    return entries


def _parse_annotation(txt_path: Path):
    with open(txt_path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    if len(lines) <= 3:
        return None
    verticals = [int(x) for x in lines[1].split()]
    horizontals = [int(x) for x in lines[2].split()]
    cells = []
    for line in lines[3:]:
        if not line.strip():
            continue
        parts = line.split()
        row, col = int(parts[0]), int(parts[1])
        dots = tuple(int(x) for x in parts[2:8])
        cells.append((row, col, dots))
    return verticals, horizontals, cells


def _estimate_box_half(verticals, horizontals) -> float:
    """Half-width of a YOLO box ≈ 35% of the median within-cell dot pitch."""
    dx = [verticals[i + 1] - verticals[i] for i in range(0, len(verticals) - 1, 2)]
    dy = []
    for i in range(0, len(horizontals) - 2, 3):
        dy.append(horizontals[i + 1] - horizontals[i])
        dy.append(horizontals[i + 2] - horizontals[i + 1])
    pitches = [p for p in dx + dy if p > 0]
    if not pitches:
        return 8.0
    median = sorted(pitches)[len(pitches) // 2]
    return max(median * 0.35, 4.0)


def _raised_dot_centers(verticals, horizontals, cells):
    """Yield (x, y) pixel centers for every raised embossed dot."""
    for row, col, dots in cells:
        vx0 = verticals[(col - 1) * 2]
        vx1 = verticals[(col - 1) * 2 + 1]
        hy = [
            horizontals[(row - 1) * 3],
            horizontals[(row - 1) * 3 + 1],
            horizontals[(row - 1) * 3 + 2],
        ]
        xs = [vx0, vx1]
        for bit, (ox, oy) in zip(dots, DOT_OFFSETS):
            if bit == 1:
                yield float(xs[ox]), float(hy[oy])


def _yolo_lines(centers, img_w, img_h, half: float):
    lines = []
    for x, y in centers:
        # clamp box inside image
        x0 = max(x - half, 0.0)
        y0 = max(y - half, 0.0)
        x1 = min(x + half, img_w - 1.0)
        y1 = min(y + half, img_h - 1.0)
        bw = x1 - x0
        bh = y1 - y0
        if bw < 1 or bh < 1:
            continue
        xc = (x0 + x1) / 2.0 / img_w
        yc = (y0 + y1) / 2.0 / img_h
        nw = bw / img_w
        nh = bh / img_h
        lines.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
    return lines


def convert_split(root: Path, split: str, out_root: Path, sides, copy_images: bool):
    entries = _parse_split_file(root / f"{split}.txt")
    img_dir = out_root / "images" / split
    lbl_dir = out_root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    n_pages, n_dots = 0, 0
    for book, base in entries:
        book_dir = root / book
        for side in sides:
            txt_path = book_dir / f"{base}+{side}.txt"
            img_path = book_dir / f"{base}+{side}.jpg"
            if not txt_path.exists() or not img_path.exists():
                continue
            parsed = _parse_annotation(txt_path)
            if parsed is None:
                continue
            verticals, horizontals, cells = parsed
            with Image.open(img_path) as im:
                img_w, img_h = im.size

            centers = list(_raised_dot_centers(verticals, horizontals, cells))
            half = _estimate_box_half(verticals, horizontals)
            yolo = _yolo_lines(centers, img_w, img_h, half)
            if not yolo:
                continue

            stem = f"{book}__{base}+{side}".replace(" ", "_")
            out_img = img_dir / f"{stem}.jpg"
            out_lbl = lbl_dir / f"{stem}.txt"

            if copy_images:
                shutil.copy2(img_path, out_img)
            else:
                # write a small sidecar pointing at the original (symlink fallback: copy)
                try:
                    if out_img.exists() or out_img.is_symlink():
                        out_img.unlink()
                    out_img.symlink_to(img_path.resolve())
                except OSError:
                    shutil.copy2(img_path, out_img)

            out_lbl.write_text("\n".join(yolo) + "\n", encoding="utf-8")
            n_pages += 1
            n_dots += len(yolo)

    return n_pages, n_dots


def write_data_yaml(out_root: Path):
    data = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/test",  # DSBI official test split used as val
        "names": {0: "braille_dot"},
        "nc": 1,
    }
    yaml_path = out_root / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="DSBI -> YOLO Braille-dot dataset")
    parser.add_argument(
        "--dbsi-root",
        type=Path,
        default=Path("data DBSI/data"),
        help="Folder containing book dirs + train.txt/test.txt",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("yolo_dot_detect/datasets/braille_dots"),
        help="Output YOLO dataset root",
    )
    parser.add_argument(
        "--sides",
        nargs="+",
        default=["recto", "verso"],
        choices=["recto", "verso"],
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy JPEGs into the dataset folder (default: try symlink, else copy)",
    )
    args = parser.parse_args()

    if not args.dbsi_root.exists():
        raise SystemExit(f"DSBI root not found: {args.dbsi_root}")

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Converting DSBI from {args.dbsi_root} -> {args.out}")

    for split in ("train", "test"):
        pages, dots = convert_split(
            args.dbsi_root, split, args.out, args.sides, args.copy_images
        )
        print(f"  {split:5s}: {pages:4d} pages, {dots:6d} raised-dot boxes")

    yaml_path = write_data_yaml(args.out)
    print(f"Wrote {yaml_path}")
    print("Done. Next: py -3.11 -m yolo_dot_detect.train")


if __name__ == "__main__":
    main()
