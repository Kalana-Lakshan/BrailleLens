"""Stage 4a step 1 - cell manifest -> single-class YOLO detection dataset.

Reads the Stage 2b manifest, so it works for every source at once (DSBI,
Angelina, and Gold once labelled). This is deliberately a new script rather
than an edit of yolo_dot_detect/prepare_dataset.py, which is DSBI-only, hard-codes
the "{base}+{side}.txt" layout, and emits *dot* boxes rather than cell boxes.

    manifest_clean.csv  ->  datasets/braille_cells/
                              images/{train,val,test}/*.jpg
                              labels/{train,val,test}/*.txt
                              data.yaml

Label format, one line per cell:

    0  x_center  y_center  width  height      (all normalized to [0, 1])

Single class: 0 = braille_cell. The dot pattern is deliberately *not* encoded
in the class - that is the CNN's job. A 64-class detector would have to solve
detection and classification at once from far less data per class.

Usage (from repo root):
    py -3.11 -m cell_detect.prepare_cell_dataset
    py -3.11 -m cell_detect.prepare_cell_dataset --sources dbsi angelina gold
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

import yaml

from data_pipeline.contracts import SPLITS, read_manifest, repo_root

ROOT = repo_root()
DEFAULT_IN = ROOT / "data_pipeline" / "manifests" / "manifest_clean.csv"
DEFAULT_OUT = ROOT / "cell_detect" / "datasets" / "braille_cells"

CLASS_ID = 0
CLASS_NAME = "braille_cell"


def _flat_stem(image_path: str) -> str:
    """Unique flat filename for a page that may sit in any nested folder.

    "data DBSI/data/Massage/M+1+recto.jpg" -> "data_DBSI_data_Massage_M+1+recto"
    Ultralytics keeps images and labels in sibling folders matched by stem, so
    the stem has to be unique across every source.
    """
    path = Path(image_path)
    flat = "_".join(path.parts[:-1] + (path.stem,))
    for bad in " ()[],":
        flat = flat.replace(bad, "_")
    return flat


def _yolo_line(row, img_w: int, img_h: int) -> str | None:
    x0 = max(float(row.x0), 0.0)
    y0 = max(float(row.y0), 0.0)
    x1 = min(float(row.x1), float(img_w))
    y1 = min(float(row.y1), float(img_h))
    if x1 - x0 < 1.0 or y1 - y0 < 1.0:
        return None
    xc = (x0 + x1) / 2.0 / img_w
    yc = (y0 + y1) / 2.0 / img_h
    nw = (x1 - x0) / img_w
    nh = (y1 - y0) / img_h
    if not (0.0 < nw <= 1.0 and 0.0 < nh <= 1.0):
        return None
    return f"{CLASS_ID} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}"


def _link_or_copy(src: Path, dst: Path, force_copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if force_copy:
        shutil.copy2(src, dst)
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        # Windows needs developer mode or admin for symlinks; copying is fine,
        # just uses more disk.
        shutil.copy2(src, dst)


def convert_split(frame, split: str, out_root: Path, copy_images: bool) -> dict:
    img_dir = out_root / "images" / split
    lbl_dir = out_root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    for image_path, page_rows in frame.groupby("image_path", sort=True):
        src = ROOT / str(image_path)
        if not src.exists():
            stats["pages_missing"] += 1
            continue

        img_w = int(page_rows["img_w"].iloc[0])
        img_h = int(page_rows["img_h"].iloc[0])
        lines = []
        for row in page_rows.itertuples(index=False):
            line = _yolo_line(row, img_w, img_h)
            if line is None:
                stats["cells_skipped"] += 1
                continue
            lines.append(line)

        if not lines:
            stats["pages_empty"] += 1
            continue

        stem = _flat_stem(str(image_path))
        _link_or_copy(src, img_dir / f"{stem}{src.suffix}", copy_images)
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        stats["pages"] += 1
        stats["cells"] += len(lines)

    return dict(stats)


def write_data_yaml(out_root: Path, has_test: bool) -> Path:
    data = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {CLASS_ID: CLASS_NAME},
        "nc": 1,
    }
    if has_test:
        data["test"] = "images/test"
    path = out_root / "data.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 4a step 1 - build a single-class YOLO cell dataset"
    )
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sources", nargs="+", default=None,
                        help="Restrict to these sources (default: everything in the manifest)")
    parser.add_argument("--copy-images", action="store_true",
                        help="Copy page images instead of symlinking (uses more disk)")
    args = parser.parse_args()

    frame = read_manifest(args.in_path)
    if args.sources:
        frame = frame[frame["source"].isin(args.sources)]
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Building YOLO cell dataset from {len(frame):,} cells -> {args.out}")
    totals = {}
    for split in SPLITS:
        split_rows = frame[frame["split"] == split]
        if split_rows.empty:
            print(f"  {split:5s}: no rows")
            totals[split] = {}
            continue
        stats = convert_split(split_rows, split, args.out, args.copy_images)
        totals[split] = stats
        print(f"  {split:5s}: {stats.get('pages', 0):4,d} pages  "
              f"{stats.get('cells', 0):7,d} cell boxes"
              + (f"  ({stats['cells_skipped']:,} cells skipped)" if stats.get("cells_skipped") else "")
              + (f"  ({stats['pages_missing']} pages missing)" if stats.get("pages_missing") else ""))

    yaml_path = write_data_yaml(args.out, bool(totals.get("test", {}).get("pages")))
    print(f"Wrote {yaml_path}")

    if not totals.get("val", {}).get("pages"):
        print(
            "\nWARNING: the val split is empty, so training has nothing to early-stop on.\n"
            "  The official dataset splits do not line up across sources - DSBI ships only\n"
            "  train/test and Angelina only train/val. Rebuild with a proper three-way split:\n"
            "    py -3.11 -m data_pipeline.integrate --split-mode rebalance\n"
            "    py -3.11 -m data_pipeline.clean\n"
            "    py -3.11 -m cell_detect.prepare_cell_dataset"
        )
    else:
        print("\nNext: upload to Colab and run BrailleLens_CellDetector_Colab.ipynb")
        print("  (local training is CPU-only here; see cell_detect/README.md)")


if __name__ == "__main__":
    main()
