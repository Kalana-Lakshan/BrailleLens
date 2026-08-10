"""Prepare a single-class YOLO fingertip dataset from Roboflow + TI1K.

Roboflow export (4 classes: index/little/middle/ring) → remap to class 0 ``fingertip``.
TI1K tip points → small YOLO boxes around the **index** tip.

Output (Ultralytics layout)::

    finger_cell_track/datasets/fingertip_yolo26/
      images/{train,val,test}/
      labels/{train,val,test}/
      data.yaml

Run from BrailleLens repo root::

    finger_cell_track\\.venv\\Scripts\\python.exe finger_cell_track/prepare_tip_yolo_dataset.py
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from PIL import Image

_HERE = Path(__file__).resolve().parent
_DATA = _HERE / "datasets"

_ROBOFLOW_DEFAULT = _DATA / "Finger Tip Detection.v1i.yolo26"
_TI1K_DEFAULT = _DATA / "TI1K-Dataset-master" / "TI1K-Dataset-master"
_OUT_DEFAULT = _DATA / "fingertip_yolo26"


def _resolve_ti1k(root: Path) -> Path:
    if (root / "annotation" / "label.txt").exists():
        return root
    nested = root / "TI1K-Dataset-master"
    if (nested / "annotation" / "label.txt").exists():
        return nested
    raise FileNotFoundError(f"TI1K label.txt not found under {root}")


def _tip_to_yolo_line(
    xi: float,
    yi: float,
    box_frac: float = 0.08,
) -> str:
    """Normalized tip (xi,yi) → YOLO line class 0 with square box."""
    half = box_frac / 2.0
    xc = min(max(xi, half), 1.0 - half)
    yc = min(max(yi, half), 1.0 - half)
    return f"0 {xc:.6f} {yc:.6f} {box_frac:.6f} {box_frac:.6f}"


def convert_ti1k(
    ti1k_root: Path,
    out_root: Path,
    *,
    box_frac: float,
    val_ratio: float,
    seed: int,
) -> tuple[int, int]:
    ti1k = _resolve_ti1k(ti1k_root)
    label_path = ti1k / "annotation" / "label.txt"
    lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    rows: list[tuple[Path, float, float]] = []
    for ln in lines:
        parts = ln.split()
        if len(parts) < 9:
            continue
        name = parts[0]
        # format: name xtl ytl xbr ybr xt yt xi yi  (normalized 0-1)
        xi, yi = float(parts[7]), float(parts[8])
        img = None
        for split in ("train", "test"):
            cand = ti1k / split / name
            if cand.exists():
                img = cand
                break
        if img is None:
            continue
        rows.append((img, xi, yi))

    rng = random.Random(seed)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * val_ratio))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    n_train = n_val_out = 0
    for split, subset in (("train", train_rows), ("val", val_rows)):
        img_dir = out_root / "images" / split
        lbl_dir = out_root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img, xi, yi in subset:
            dst_name = f"ti1k_{img.stem}{img.suffix}"
            shutil.copy2(img, img_dir / dst_name)
            (lbl_dir / f"ti1k_{img.stem}.txt").write_text(
                _tip_to_yolo_line(xi, yi, box_frac) + "\n",
                encoding="utf-8",
            )
            if split == "train":
                n_train += 1
            else:
                n_val_out += 1
    return n_train, n_val_out


def _remap_label_file(src: Path, dst: Path) -> None:
    """Map any class id → 0 (single fingertip class). Skip empty."""
    out_lines = []
    for ln in src.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) < 5:
            continue
        parts[0] = "0"
        out_lines.append(" ".join(parts))
    dst.write_text(("\n".join(out_lines) + ("\n" if out_lines else "")), encoding="utf-8")


def copy_roboflow(
    rf_root: Path,
    out_root: Path,
    *,
    include_test: bool,
) -> dict[str, int]:
    mapping = {"train": "train", "valid": "val", "test": "test"}
    counts = {"train": 0, "val": 0, "test": 0}
    for rf_split, out_split in mapping.items():
        if out_split == "test" and not include_test:
            continue
        img_src = rf_root / rf_split / "images"
        lbl_src = rf_root / rf_split / "labels"
        if not img_src.is_dir():
            continue
        img_dst = out_root / "images" / out_split
        lbl_dst = out_root / "labels" / out_split
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)
        for img in img_src.glob("*.*"):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                continue
            stem = img.stem
            lbl = lbl_src / f"{stem}.txt"
            dst_img = img_dst / f"rf_{img.name}"
            shutil.copy2(img, dst_img)
            if lbl.exists():
                _remap_label_file(lbl, lbl_dst / f"rf_{stem}.txt")
            else:
                (lbl_dst / f"rf_{stem}.txt").write_text("", encoding="utf-8")
            counts[out_split] += 1
    return counts


def write_data_yaml(out_root: Path) -> Path:
    yaml_path = out_root / "data.yaml"
    # Paths relative to this yaml file (Ultralytics convention)
    text = f"""# BrailleLens fingertip detector (single class)
path: {out_root.resolve().as_posix()}
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: fingertip
"""
    yaml_path.write_text(text, encoding="utf-8")
    return yaml_path


def main() -> None:
    p = argparse.ArgumentParser(description="Build fingertip_yolo26 dataset")
    p.add_argument("--roboflow", type=Path, default=_ROBOFLOW_DEFAULT)
    p.add_argument("--ti1k", type=Path, default=_TI1K_DEFAULT)
    p.add_argument("--out", type=Path, default=_OUT_DEFAULT)
    p.add_argument("--box-frac", type=float, default=0.08, help="TI1K tip box size (normalized)")
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-roboflow", action="store_true")
    p.add_argument("--skip-ti1k", action="store_true")
    p.add_argument("--no-test", action="store_true")
    p.add_argument("--clean", action="store_true", help="Delete output dir first")
    args = p.parse_args()

    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Output → {args.out}")

    if not args.skip_roboflow:
        if not args.roboflow.is_dir():
            raise SystemExit(f"Roboflow folder not found: {args.roboflow}")
        counts = copy_roboflow(args.roboflow, args.out, include_test=not args.no_test)
        print(f"Roboflow copied: {counts}")

    if not args.skip_ti1k:
        if not args.ti1k.exists() and not (_DATA / "TI1K-Dataset-master").exists():
            print("TI1K not found — skipping")
        else:
            ti_root = args.ti1k if args.ti1k.exists() else _DATA / "TI1K-Dataset-master"
            n_tr, n_va = convert_ti1k(
                ti_root,
                args.out,
                box_frac=args.box_frac,
                val_ratio=args.val_ratio,
                seed=args.seed,
            )
            print(f"TI1K added: train+={n_tr} val+={n_va}")

    yaml_path = write_data_yaml(args.out)
    # quick sanity: ensure at least one train image opens
    sample = next((args.out / "images" / "train").glob("*.*"), None)
    if sample:
        with Image.open(sample) as im:
            print(f"Sample OK: {sample.name} size={im.size}")
    print(f"Wrote {yaml_path}")
    print("Next: zip fingertip_yolo26 for Kaggle, or train locally if you have a CUDA GPU.")


if __name__ == "__main__":
    main()
