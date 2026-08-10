"""Mark Braille dots on every image in a folder.

Drop photos into:
    yolo_dot_detect/test_images/input/

Run:
    py -3.11 -m yolo_dot_detect.mark_folder

Marked overlays are written to:
    yolo_dot_detect/test_images/output/<name>_dots.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _default_weights() -> Path:
    here = Path(__file__).resolve().parent / "runs" / "detect"
    for name in (
        "braille_dot_yolo26_tiled",
        "braille_dot_yolo26",
        "braille_dot_yolov8",
    ):
        cand = here / name / "weights" / "best.pt"
        if cand.exists():
            return cand
    return here / "braille_dot_yolo26_tiled" / "weights" / "best.pt"


def _draw(bgr, detections, clusters=None):
    out = bgr.copy()
    for d in detections:
        x0, y0, x1, y1 = (int(round(v)) for v in d["xyxy"])
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 220, 80), 1)
        cx, cy = (int(round(v)) for v in d["center"])
        cv2.circle(out, (cx, cy), 3, (0, 80, 255), -1)

    if clusters:
        for cl in clusters:
            x0, y0, x1, y1 = (int(round(v)) for v in cl["bbox"])
            color = (0, 0, 255) if cl["merged"] else (255, 140, 0)
            cv2.rectangle(out, (x0 - 4, y0 - 4), (x1 + 4, y1 + 4), color, 2)

    label = f"dots: {len(detections)}"
    if clusters is not None:
        n_ok = sum(1 for c in clusters if not c["merged"])
        label += f"  cells: {n_ok}"
    cv2.putText(
        out,
        label,
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 220, 80),
        2,
        cv2.LINE_AA,
    )
    return out


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Mark Braille dots on a folder of images")
    parser.add_argument(
        "--input",
        type=Path,
        default=here / "test_images" / "input",
        help="Folder with photos to process",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "test_images" / "output",
        help="Folder for marked overlays",
    )
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--tile", type=int, default=640, help="0 = whole-image")
    parser.add_argument("--tile-overlap", type=int, default=96)
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="Also draw Braille-cell boxes around grouped dots",
    )
    parser.add_argument("--link-distance", type=float, default=15.0)
    args = parser.parse_args()

    args.input.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in args.input.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        print(f"No images found in:\n  {args.input.resolve()}")
        print("Drop .jpg / .jpeg / .png files there, then re-run this command.")
        return

    weights = args.weights or _default_weights()
    if not weights.exists():
        raise SystemExit(
            f"Weights not found: {weights}\n"
            "Train first or pass --weights path/to/best.pt"
        )

    repo_root = here.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from yolo_dot_detect.detect_dots import YoloDotDetector

    detector = YoloDotDetector(
        weights=weights,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        tile=args.tile or None,
        tile_overlap=args.tile_overlap,
    )

    cluster_fn = None
    if args.cluster:
        from braille_cnn.dot_detect import cluster_into_cells

        cluster_fn = cluster_into_cells

    print(f"Weights : {weights}")
    print(f"Input   : {args.input.resolve()}  ({len(images)} image(s))")
    print(f"Output  : {args.output.resolve()}")
    print("-" * 50)

    for img_path in images:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"  SKIP (unreadable): {img_path.name}")
            continue

        detections = detector.detect_boxes(bgr)
        clusters = None
        if cluster_fn is not None:
            centers = (
                np.array([d["center"] for d in detections], dtype=np.float64)
                if detections
                else np.zeros((0, 2), dtype=np.float64)
            )
            clusters = cluster_fn(centers, link_distance=args.link_distance)

        overlay = _draw(bgr, detections, clusters)
        out_path = args.output / f"{img_path.stem}_dots.png"
        cv2.imwrite(str(out_path), overlay)

        extra = ""
        if clusters is not None:
            n_ok = sum(1 for c in clusters if not c["merged"])
            extra = f", {n_ok} cells"
        print(f"  {img_path.name} -> {out_path.name}  ({len(detections)} dots{extra})")

    print("-" * 50)
    print(f"Done. Open marked images in:\n  {args.output.resolve()}")


if __name__ == "__main__":
    main()
