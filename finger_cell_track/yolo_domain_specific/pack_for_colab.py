"""Zip braille_fingertip_yolo for Google Drive / Colab upload.

Creates:
  finger_cell_track/yolo_domain_specific/colab_upload/braille_fingertip_yolo.zip

Upload to: MyDrive/BrailleLens_Fingertip_Domain/braille_fingertip_yolo.zip

Usage (from repo root)::

    finger_cell_track\\.venv\\Scripts\\python.exe finger_cell_track/yolo_domain_specific/pack_for_colab.py
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DEFAULT_DATASET = _HERE / "datasets" / "braille_fingertip_yolo"
_DEFAULT_OUT = _HERE / "colab_upload" / "braille_fingertip_yolo.zip"


def main() -> None:
    p = argparse.ArgumentParser(description="Pack Braille fingertip YOLO dataset for Colab")
    p.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET)
    p.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    args = p.parse_args()

    if not args.dataset.exists():
        raise SystemExit(
            f"Dataset not found: {args.dataset}\n"
            "Run first: finger_cell_track/yolo_domain_specific/build_dataset.py"
        )
    if not (args.dataset / "data.yaml").exists():
        raise SystemExit(f"Missing data.yaml in {args.dataset}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()

    n = 0
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in args.dataset.rglob("*"):
            if path.is_file() and path.suffix.lower() not in {".cache"}:
                arc = Path("braille_fingertip_yolo") / path.relative_to(args.dataset)
                zf.write(path, arcname=str(arc).replace("\\", "/"))
                n += 1

    mb = args.out.stat().st_size / (1024 * 1024)
    print(f"Packed {n} files -> {args.out} ({mb:.1f} MB)")
    print("Upload to Google Drive:")
    print("  MyDrive/BrailleLens_Fingertip_Domain/braille_fingertip_yolo.zip")


if __name__ == "__main__":
    main()
