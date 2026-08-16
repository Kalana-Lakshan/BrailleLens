"""Stage 4a / 6 — cell-detector mAP on the prepared YOLO val split.

    py -3.11 -m cell_detect.evaluate_detector
    py -3.11 -m cell_detect.evaluate_detector --weights cell_detect/weights/braille_cell_best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "datasets" / "braille_cells" / "data.yaml"
DEFAULT_WEIGHTS = HERE / "weights" / "braille_cell_best.pt"


def _resolve_weights(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    if DEFAULT_WEIGHTS.exists():
        return DEFAULT_WEIGHTS
    for name in ("braille_cell_yolo26", "smoke_test"):
        cand = HERE / "runs" / "detect" / name / "weights" / "best.pt"
        if cand.exists():
            return cand
    return DEFAULT_WEIGHTS


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Braille cell detector")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    weights = _resolve_weights(args.weights)
    if not weights.exists():
        raise SystemExit(f"Weights not found: {weights}\nSee cell_detect/COLAB_SETUP.md")
    if not args.data.exists():
        raise SystemExit(
            f"data.yaml not found: {args.data}\n"
            "Build it first: py -3.11 -m cell_detect.prepare_cell_dataset"
        )

    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(args.data),
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        plots=True,
    )
    box = metrics.box
    print("=== Braille-cell detection metrics ===")
    print(f"  Precision : {box.mp:.4f}")
    print(f"  Recall    : {box.mr:.4f}")
    print(f"  mAP50     : {box.map50:.4f}")
    print(f"  mAP50-95  : {box.map:.4f}")


if __name__ == "__main__":
    main()
