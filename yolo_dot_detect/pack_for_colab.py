"""Zip the prepared YOLO Braille-dot dataset for Google Drive / Colab upload.

Creates:
  yolo_dot_detect/colab_upload/braille_dots.zip

Upload that zip to Drive as described in COLAB_SETUP.md.

Usage (from repo root):
    py -3.11 -m yolo_dot_detect.pack_for_colab
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=here / "datasets" / "braille_dots",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=here / "colab_upload" / "braille_dots.zip",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        raise SystemExit(
            f"Dataset not found: {args.dataset}\n"
            "Run first: py -3.11 -m yolo_dot_detect.prepare_dataset --dbsi-root \"data DBSI/data\" --copy-images"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()

    n = 0
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in args.dataset.rglob("*"):
            if path.is_file() and path.suffix.lower() not in {".cache"}:
                arc = Path("braille_dots") / path.relative_to(args.dataset)
                zf.write(path, arcname=str(arc).replace("\\", "/"))
                n += 1

    mb = args.out.stat().st_size / (1024 * 1024)
    print(f"Packed {n} files -> {args.out} ({mb:.1f} MB)")
    print("Upload this zip to Google Drive:")
    print("  MyDrive/BrailleLens_YOLO26/braille_dots.zip")


if __name__ == "__main__":
    main()
