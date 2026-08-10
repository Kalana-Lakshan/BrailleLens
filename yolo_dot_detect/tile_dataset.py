"""Slice full Braille pages into overlapping tiles for YOLO training.

A DSBI page holds ~1000-5000 raised dots. YOLO's label assigner allocates
memory proportional to (boxes x anchors), so a full page at imgsz=1280 blows
up GPU memory ("CUDA OutOfMemoryError in TaskAlignedAssigner"). Tiling cuts
each page into ~640px patches holding ~50-150 dots, which also keeps dots at
native pixel size instead of shrinking them during letterbox resize.

Boxes are assigned to a tile when their *center* lands inside it, then clipped
to the tile bounds.

Usage (from repo root):
    py -3.11 -m yolo_dot_detect.tile_dataset
    py -3.11 -m yolo_dot_detect.tile_dataset --tile 640 --overlap 96
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def _read_labels(lbl_path: Path):
    """Returns list of (cls, xc, yc, w, h) in normalized coords."""
    if not lbl_path.exists():
        return []
    rows = []
    for line in lbl_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, w, h = (float(v) for v in parts[1:5])
        rows.append((cls, xc, yc, w, h))
    return rows


def _tile_origins(total: int, tile: int, stride: int):
    """Tile start offsets covering [0, total), last tile flush to the edge."""
    if total <= tile:
        return [0]
    origins = list(range(0, total - tile + 1, stride))
    if origins[-1] != total - tile:
        origins.append(total - tile)
    return origins


def tile_split(
    src_root: Path,
    dst_root: Path,
    split: str,
    tile: int,
    overlap: int,
    min_boxes: int,
    max_pages: int | None,
    quality: int,
):
    src_img_dir = src_root / "images" / split
    src_lbl_dir = src_root / "labels" / split
    dst_img_dir = dst_root / "images" / split
    dst_lbl_dir = dst_root / "labels" / split
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    pages = sorted(src_img_dir.glob("*.jpg"))
    if max_pages is not None:
        pages = pages[:max_pages]

    stride = max(tile - overlap, 1)
    n_tiles, n_boxes = 0, 0

    for page_path in pages:
        rows = _read_labels(src_lbl_dir / (page_path.stem + ".txt"))
        if not rows:
            continue

        with Image.open(page_path) as im:
            im = im.convert("RGB")
            pw, ph = im.size

            # denormalize once per page
            abs_boxes = []
            for cls, xc, yc, w, h in rows:
                acx, acy = xc * pw, yc * ph
                aw, ah = w * pw, h * ph
                abs_boxes.append((cls, acx, acy, aw, ah))

            for oy in _tile_origins(ph, tile, stride):
                for ox in _tile_origins(pw, tile, stride):
                    tw = min(tile, pw)
                    th = min(tile, ph)
                    keep = []
                    for cls, acx, acy, aw, ah in abs_boxes:
                        if not (ox <= acx < ox + tw and oy <= acy < oy + th):
                            continue
                        # shift into tile space, clip to tile
                        x0 = max(acx - aw / 2.0 - ox, 0.0)
                        y0 = max(acy - ah / 2.0 - oy, 0.0)
                        x1 = min(acx + aw / 2.0 - ox, float(tw))
                        y1 = min(acy + ah / 2.0 - oy, float(th))
                        bw, bh = x1 - x0, y1 - y0
                        if bw < 1.0 or bh < 1.0:
                            continue
                        keep.append(
                            f"{cls} "
                            f"{((x0 + x1) / 2.0 / tw):.6f} "
                            f"{((y0 + y1) / 2.0 / th):.6f} "
                            f"{(bw / tw):.6f} "
                            f"{(bh / th):.6f}"
                        )

                    if len(keep) < min_boxes:
                        continue

                    stem = f"{page_path.stem}__x{ox}_y{oy}"
                    crop = im.crop((ox, oy, ox + tw, oy + th))
                    crop.save(dst_img_dir / f"{stem}.jpg", quality=quality)
                    (dst_lbl_dir / f"{stem}.txt").write_text(
                        "\n".join(keep) + "\n", encoding="utf-8"
                    )
                    n_tiles += 1
                    n_boxes += len(keep)

    return len(pages), n_tiles, n_boxes


def write_data_yaml(dst_root: Path):
    data = {
        "path": str(dst_root.resolve()),
        "train": "images/train",
        "val": "images/test",
        "names": {0: "braille_dot"},
        "nc": 1,
    }
    out = dst_root / "data.yaml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return out


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Tile Braille pages for YOLO")
    parser.add_argument(
        "--src",
        type=Path,
        default=here / "datasets" / "braille_dots",
        help="Full-page YOLO dataset (from prepare_dataset.py)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=here / "datasets" / "braille_dots_tiled",
    )
    parser.add_argument("--tile", type=int, default=640)
    parser.add_argument("--overlap", type=int, default=96)
    parser.add_argument(
        "--min-boxes",
        type=int,
        default=5,
        help="Discard tiles holding fewer dots than this",
    )
    parser.add_argument(
        "--train-pages", type=int, default=None, help="Limit train pages (debug)"
    )
    parser.add_argument(
        "--val-pages",
        type=int,
        default=40,
        help="Limit val pages to keep per-epoch validation fast (None = all)",
    )
    parser.add_argument("--quality", type=int, default=92)
    args = parser.parse_args()

    if not (args.src / "images" / "train").exists():
        raise SystemExit(
            f"Source dataset not found: {args.src}\n"
            "Run first: py -3.11 -m yolo_dot_detect.prepare_dataset "
            '--dbsi-root "data DBSI/data" --copy-images'
        )

    print(f"Tiling {args.src} -> {args.out}")
    print(f"  tile={args.tile} overlap={args.overlap} min_boxes={args.min_boxes}")

    for split, limit in (("train", args.train_pages), ("test", args.val_pages)):
        pages, tiles, boxes = tile_split(
            args.src,
            args.out,
            split,
            args.tile,
            args.overlap,
            args.min_boxes,
            limit,
            args.quality,
        )
        per_tile = (boxes / tiles) if tiles else 0
        print(
            f"  {split:5s}: {pages:4d} pages -> {tiles:5d} tiles, "
            f"{boxes:7d} boxes ({per_tile:.0f} per tile)"
        )

    print(f"Wrote {write_data_yaml(args.out)}")
    print("Next: train with imgsz=640 on this tiled dataset")


if __name__ == "__main__":
    main()
