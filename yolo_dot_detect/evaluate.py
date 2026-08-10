"""Evaluate a trained YOLOv8 dot detector on the YOLO val/test split.

Runs Ultralytics validation (precision / recall / mAP50 / mAP50-95) and
optionally reports per-image detection counts vs ground-truth box counts.

Usage:
    py -3.11 -m yolo_dot_detect.evaluate
    py -3.11 -m yolo_dot_detect.evaluate --weights path/to/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _default_weights() -> Path:
    here = Path(__file__).resolve().parent / "runs" / "detect"
    for name in ("braille_dot_yolo26", "braille_dot_yolov8"):
        cand = here / name / "weights" / "best.pt"
        if cand.exists():
            return cand
    return here / "braille_dot_yolo26" / "weights" / "best.pt"


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 Braille-dot model")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument(
        "--data",
        type=Path,
        default=here / "datasets" / "braille_dots" / "data.yaml",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    weights = args.weights or _default_weights()
    if not weights.exists():
        raise SystemExit(f"Weights not found: {weights}")
    if not args.data.exists():
        raise SystemExit(f"data.yaml not found: {args.data}")

    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(args.data),
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        max_det=3000,
        split="val",
        plots=True,
    )

    box = metrics.box
    print("\n=== Braille-dot detection metrics ===")
    print(f"  Precision : {box.mp:.4f}")
    print(f"  Recall    : {box.mr:.4f}")
    print(f"  mAP50     : {box.map50:.4f}")
    print(f"  mAP50-95  : {box.map:.4f}")


if __name__ == "__main__":
    main()
