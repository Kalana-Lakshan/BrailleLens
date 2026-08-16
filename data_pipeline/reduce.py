"""Stage 2c - Reduction: 432 full pages -> one compact crop array per split.

This is where the real data reduction happens. A DSBI page is about 1700x2340
colour pixels; the model needs a 64x64 grayscale cell. That is roughly a
2900-fold reduction per cell, and it is what makes training on a laptop-built
dataset feasible at all.

    manifest_clean.csv  ->  crops_train.npz / crops_val.npz / crops_test.npz

Each page image is opened exactly once and all of its cells are cropped in one
pass. Doing it lazily per sample instead would re-decode a multi-megapixel JPEG
on almost every training sample once the loader shuffles.

Crops are stored as uint8, not float32. Same pixels, a quarter of the memory,
and the float conversion is cheap enough to do per batch.

Usage (from repo root):
    py -3.11 -m data_pipeline.reduce
    py -3.11 -m data_pipeline.reduce --splits train val --img-size 64
    py -3.11 -m data_pipeline.reduce --cap-per-class 4000
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .contracts import SPLITS, read_manifest, repo_root
from .transform import IMG_SIZE_DEFAULT, extract_crop, margin_for

ROOT = repo_root()
DEFAULT_IN = ROOT / "data_pipeline" / "manifests" / "manifest_clean.csv"
DEFAULT_OUT_DIR = ROOT / "data_pipeline" / "crops"


def _cap_per_class(frame, cap: int, seed: int = 42):
    """Randomly subsample over-represented codes.

    Real Braille text is heavily skewed - a few frequent letters dominate and
    rare codes are almost absent. Capping keeps the frequent classes from
    swamping the loss without touching the rare ones. Sampling is seeded so the
    dataset is reproducible.
    """
    parts = []
    for code, group in frame.groupby("code"):
        if len(group) > cap:
            group = group.sample(n=cap, random_state=seed)
        parts.append(group)
    import pandas as pd

    return pd.concat(parts).sort_index()


def build_split(frame, img_size: int, margin_override: float | None):
    """Extract every cell crop for one split. Returns arrays plus a stats dict."""
    crops: list[np.ndarray] = []
    codes: list[int] = []
    sources: list[str] = []
    groups: list[str] = []

    stats = {"images_read": 0, "images_missing": 0, "cells_ok": 0, "cells_skipped": 0}

    # sort by path so each page is decoded once, in one contiguous run
    for image_path, page_rows in frame.groupby("image_path", sort=True):
        full_path = ROOT / str(image_path)
        image = cv2.imread(str(full_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            stats["images_missing"] += 1
            stats["cells_skipped"] += len(page_rows)
            continue
        stats["images_read"] += 1

        for row in page_rows.itertuples(index=False):
            margin = margin_override if margin_override is not None else margin_for(row.source)
            crop = extract_crop(
                image, (row.x0, row.y0, row.x1, row.y1), margin=margin, img_size=img_size
            )
            if crop is None:
                stats["cells_skipped"] += 1
                continue
            crops.append(crop)
            codes.append(int(row.code))
            sources.append(str(row.source))
            groups.append(str(row.page_group))
            stats["cells_ok"] += 1

    if not crops:
        return None, stats

    payload = {
        "crops": np.stack(crops).astype(np.uint8),
        "codes": np.asarray(codes, dtype=np.int64),
        "sources": np.asarray(sources),
        "page_groups": np.asarray(groups),
    }
    return payload, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2c - extract cell crops")
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    parser.add_argument("--sources", nargs="+", default=None,
                        help="Restrict to these sources (default: all in the manifest)")
    parser.add_argument("--img-size", type=int, default=IMG_SIZE_DEFAULT)
    parser.add_argument("--margin", type=float, default=None,
                        help="Override the per-source margin from transform.SOURCE_MARGINS")
    parser.add_argument("--cap-per-class", type=int, default=None,
                        help="Max crops per code in the train split (class balancing)")
    parser.add_argument("--no-compress", action="store_true",
                        help="Write uncompressed .npz (much faster, roughly 3x larger)")
    args = parser.parse_args()

    frame = read_manifest(args.in_path)
    if args.sources:
        frame = frame[frame["source"].isin(args.sources)]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest_meta = {
        "img_size": args.img_size,
        "margin_override": args.margin,
        "cap_per_class": args.cap_per_class,
        "source_manifest": str(args.in_path.relative_to(ROOT) if args.in_path.is_absolute() else args.in_path),
        "splits": {},
    }

    for split in args.splits:
        split_rows = frame[frame["split"] == split]
        if split_rows.empty:
            print(f"{split:5s}: no rows, skipped")
            continue

        if args.cap_per_class and split == "train":
            before = len(split_rows)
            split_rows = _cap_per_class(split_rows, args.cap_per_class)
            print(f"{split:5s}: class cap {args.cap_per_class} -> "
                  f"{before:,} to {len(split_rows):,} cells")

        print(f"{split:5s}: extracting {len(split_rows):,} cells from "
              f"{split_rows['image_path'].nunique():,} images ...")
        payload, stats = build_split(split_rows, args.img_size, args.margin)
        if payload is None:
            print(f"{split:5s}: produced no crops, skipped")
            continue

        out_path = args.out_dir / f"crops_{split}.npz"
        saver = np.savez if args.no_compress else np.savez_compressed
        saver(out_path, **payload)

        code_counts = Counter(payload["codes"].tolist())
        source_counts = Counter(payload["sources"].tolist())
        size_mb = out_path.stat().st_size / 1e6
        print(
            f"{split:5s}: {stats['cells_ok']:,} crops  "
            f"{len(code_counts)}/64 classes present  "
            f"{dict(source_counts)}  "
            f"{size_mb:.1f} MB -> {out_path.name}"
        )
        if stats["cells_skipped"]:
            print(f"       {stats['cells_skipped']:,} cells skipped "
                  f"({stats['images_missing']} images unreadable)")

        manifest_meta["splits"][split] = {
            "crops": int(stats["cells_ok"]),
            "skipped": int(stats["cells_skipped"]),
            "classes_present": len(code_counts),
            "by_source": {k: int(v) for k, v in source_counts.items()},
            "file": out_path.name,
            "size_mb": round(size_mb, 1),
        }

    meta_path = args.out_dir / "crops_meta.json"
    meta_path.write_text(json.dumps(manifest_meta, indent=2), encoding="utf-8")
    print(f"\nWrote {meta_path}")
    print("Next: py -3.11 -m braille_cnn.train_classifier --help")


if __name__ == "__main__":
    main()
