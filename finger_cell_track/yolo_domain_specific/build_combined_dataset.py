"""Merge Braille_fingertip into fingertip_yolo26 (TI1K + Roboflow) and zip for Colab.

Sources (all must exist locally):
  - finger_cell_track/datasets/fingertip_yolo26/     (TI1K + Roboflow via prepare_tip_yolo_dataset.py)
  - yolo_domain_specific/datasets/braille_fingertip_yolo/  (via build_dataset.py)

Adds 60 Braille images with ``braille_`` prefix into train/val/test, then optionally zips.

Usage (from repo root)::

    finger_cell_track\\.venv\\Scripts\\python.exe finger_cell_track/yolo_domain_specific/build_dataset.py --clean
    finger_cell_track\\.venv\\Scripts\\python.exe finger_cell_track/yolo_domain_specific/build_combined_dataset.py --zip
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_FCT = _HERE.parent
_BASE_DEFAULT = _FCT / "datasets" / "fingertip_yolo26"
_BRAILLE_DEFAULT = _HERE / "datasets" / "braille_fingertip_yolo"
_ZIP_DEFAULT = _HERE / "colab_upload" / "fingertip_combined_yolo26.zip"
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _count_split(base: Path, split: str) -> int:
    d = base / "images" / split
    if not d.is_dir():
        return 0
    return sum(1 for f in d.iterdir() if f.suffix.lower() in _IMG_EXTS)


def _ensure_braille_dataset(braille_root: Path) -> None:
    if (braille_root / "data.yaml").exists() and _count_split(braille_root, "train") > 0:
        return
    script = _HERE / "build_dataset.py"
    print("Braille YOLO set missing — running build_dataset.py --clean ...")
    subprocess.run(
        [sys.executable, str(script), "--clean"],
        check=True,
        cwd=str(_FCT.parent),
    )


def _ensure_base_dataset(base: Path) -> None:
    if (base / "data.yaml").exists() and _count_split(base, "train") > 0:
        return
    script = _FCT / "prepare_tip_yolo_dataset.py"
    print("fingertip_yolo26 missing — running prepare_tip_yolo_dataset.py --clean ...")
    subprocess.run(
        [sys.executable, str(script), "--clean"],
        check=True,
        cwd=str(_FCT.parent),
    )


def merge_braille(base: Path, braille: Path, *, prefix: str = "braille_") -> dict[str, int]:
    added = {"train": 0, "val": 0, "test": 0}
    for split in ("train", "val", "test"):
        img_src = braille / "images" / split
        lbl_src = braille / "labels" / split
        img_dst = base / "images" / split
        lbl_dst = base / "labels" / split
        if not img_src.is_dir():
            continue
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)
        for img in sorted(img_src.iterdir()):
            if img.suffix.lower() not in _IMG_EXTS:
                continue
            stem = img.stem
            out_name = f"{prefix}{img.name}"
            shutil.copy2(img, img_dst / out_name)
            lbl = lbl_src / f"{stem}.txt"
            out_lbl = lbl_dst / f"{prefix}{stem}.txt"
            if lbl.exists():
                shutil.copy2(lbl, out_lbl)
            else:
                out_lbl.write_text("", encoding="utf-8")
            added[split] += 1
    return added


def update_data_yaml(base: Path) -> None:
    yaml_path = base / "data.yaml"
    text = f"""# BrailleLens fingertip detector — TI1K + Roboflow + Braille_fingertip (single class)
path: {base.resolve().as_posix()}
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: fingertip
"""
    yaml_path.write_text(text, encoding="utf-8")


def pack_zip(base: Path, out_zip: Path) -> None:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    if out_zip.exists():
        out_zip.unlink()
    folder_name = "fingertip_combined_yolo26"
    n = 0
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.lower() not in {".cache"}:
                arc = Path(folder_name) / path.relative_to(base)
                zf.write(path, arcname=str(arc).replace("\\", "/"))
                n += 1
    mb = out_zip.stat().st_size / (1024 * 1024)
    print(f"Packed {n} files -> {out_zip} ({mb:.1f} MB)")


def main() -> None:
    p = argparse.ArgumentParser(description="Merge Braille into fingertip_yolo26 + optional zip")
    p.add_argument("--base", type=Path, default=_BASE_DEFAULT, help="TI1K+Roboflow YOLO folder")
    p.add_argument("--braille", type=Path, default=_BRAILLE_DEFAULT)
    p.add_argument("--zip", action="store_true", help="Create colab_upload zip after merge")
    p.add_argument("--zip-out", type=Path, default=_ZIP_DEFAULT)
    p.add_argument("--skip-build", action="store_true", help="Do not auto-run missing dataset scripts")
    args = p.parse_args()

    if not args.skip_build:
        _ensure_base_dataset(args.base)
        _ensure_braille_dataset(args.braille)

    if not args.base.is_dir():
        raise SystemExit(f"Base dataset not found: {args.base}")
    if not args.braille.is_dir():
        raise SystemExit(f"Braille dataset not found: {args.braille}")

    print(f"Base (TI1K+Roboflow): {args.base}")
    before = {s: _count_split(args.base, s) for s in ("train", "val", "test")}
    print(f"  before merge — train:{before['train']} val:{before['val']} test:{before['test']}")

    added = merge_braille(args.base, args.braille)
    update_data_yaml(args.base)

    after = {s: _count_split(args.base, s) for s in ("train", "val", "test")}
    print(f"Braille added — train:+{added['train']} val:+{added['val']} test:+{added['test']}")
    print(f"  after merge  — train:{after['train']} val:{after['val']} test:{after['test']}")
    print(f"  data.yaml: {args.base / 'data.yaml'}")

    if args.zip:
        print("Zipping (may take several minutes)...")
        pack_zip(args.base, args.zip_out)
        print("Upload to Drive: MyDrive/BrailleLens_Fingertip_Domain/fingertip_combined_yolo26.zip")


if __name__ == "__main__":
    main()
