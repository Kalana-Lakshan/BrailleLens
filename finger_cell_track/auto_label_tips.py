"""Auto-label fingertip boxes with trained YOLO (review/correct in Roboflow).

Writes YOLO-format labels next to images (or under --out-labels).

    python finger_cell_track/auto_label_tips.py --images path/to/photos --out-dir finger_cell_track/datasets/braille_tip_auto
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from tip_yolo import TipYOLO  # noqa: E402

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    p = argparse.ArgumentParser(description="Auto-label fingertips with YOLO")
    p.add_argument("--images", type=Path, required=True, help="Folder of images")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_HERE / "datasets" / "braille_tip_auto",
        help="Output YOLO dataset root (images/ + labels/)",
    )
    p.add_argument("--tip-weights", type=Path, default=None)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--skip-empty",
        action="store_true",
        help="Do not copy images with zero detections",
    )
    args = p.parse_args()

    src = args.images
    if not src.is_dir():
        raise SystemExit(f"Not a folder: {src}")

    img_out = args.out_dir / "images"
    lab_out = args.out_dir / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lab_out.mkdir(parents=True, exist_ok=True)

    tipper = TipYOLO(
        weights=args.tip_weights,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
    )
    print(f"Using {tipper.weights}", flush=True)

    paths = sorted(
        f for f in src.rglob("*") if f.suffix.lower() in _IMG_EXTS and f.is_file()
    )
    n_ok, n_empty = 0, 0
    for i, path in enumerate(paths, 1):
        bgr = cv2.imread(str(path))
        if bgr is None:
            print(f"skip unreadable: {path}", flush=True)
            continue
        h, w = bgr.shape[:2]
        r = tipper.model.predict(
            source=bgr,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]
        lines: list[str] = []
        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                bw = max(x2 - x1, 1.0)
                bh = max(y2 - y1, 1.0)
                cx = ((x1 + x2) / 2.0) / w
                cy = ((y1 + y2) / 2.0) / h
                nw = bw / w
                nh = bh / h
                # class 0 = fingertip
                lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        if not lines:
            n_empty += 1
            if args.skip_empty:
                continue

        stem = f"{path.parent.name}__{path.stem}" if path.parent != src else path.stem
        # avoid collisions
        out_name = f"{stem}{path.suffix.lower()}"
        dst_img = img_out / out_name
        dst_lab = lab_out / f"{Path(out_name).stem}.txt"
        shutil.copy2(path, dst_img)
        dst_lab.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        n_ok += 1
        if i % 20 == 0 or i == len(paths):
            print(f"[{i}/{len(paths)}] labeled={n_ok} empty={n_empty}", flush=True)

    yaml_path = args.out_dir / "data.yaml"
    yaml_path.write_text(
        f"path: {args.out_dir.resolve().as_posix()}\n"
        f"train: images\n"
        f"val: images\n"
        f"names:\n  0: fingertip\n",
        encoding="utf-8",
    )
    print(
        f"Done. {n_ok} images → {args.out_dir}\n"
        f"Empty detections: {n_empty}\n"
        f"Review/correct labels in Roboflow, then fine-tune from best.pt.",
        flush=True,
    )


if __name__ == "__main__":
    main()
