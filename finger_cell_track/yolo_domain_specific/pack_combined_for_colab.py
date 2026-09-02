"""Zip fingertip_combined_yolo for Google Drive upload.

    finger_cell_track\\.venv\\Scripts\\python.exe finger_cell_track/yolo_domain_specific/pack_combined_for_colab.py

Upload to: MyDrive/BrailleLens_Fingertip_Combined/fingertip_combined_yolo.zip
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DEFAULT = _HERE / "datasets" / "fingertip_combined_yolo"
_OUT = _HERE / "colab_upload" / "fingertip_combined_yolo.zip"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=_DEFAULT)
    p.add_argument("--out", type=Path, default=_OUT)
    args = p.parse_args()

    if not (args.dataset / "data.yaml").exists():
        raise SystemExit(
            f"Missing {args.dataset}/data.yaml\n"
            "Run: finger_cell_track/yolo_domain_specific/build_combined_dataset.py --clean"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()

    n = 0
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in args.dataset.rglob("*"):
            if path.is_file() and path.suffix.lower() != ".cache":
                arc = Path("fingertip_combined_yolo") / path.relative_to(args.dataset)
                zf.write(path, arcname=str(arc).replace("\\", "/"))
                n += 1

    mb = args.out.stat().st_size / (1024 * 1024)
    print(f"Packed {n} files -> {args.out} ({mb:.1f} MB)")
    print("Upload to Drive: MyDrive/BrailleLens_Fingertip_Combined/fingertip_combined_yolo.zip")


if __name__ == "__main__":
    main()
