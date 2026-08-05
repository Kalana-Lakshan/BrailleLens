"""Step 3 — Run the trained YOLOv8 dot detector on a page image.

Draws detected embossed-dot boxes, optionally clusters them into Braille
cells (reuses braille_cnn.dot_detect.cluster_into_cells), and can compare
against the classical peak detector.

Usage:
    py -3.11 -m yolo_dot_detect.infer --image test-img.jpeg
    py -3.11 -m yolo_dot_detect.infer --image test-img.jpeg --compare-classical
    py -3.11 -m yolo_dot_detect.infer --image test-img.jpeg --cluster --link-distance 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


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


def _draw_detections(bgr, detections, color=(0, 200, 80)):
    out = bgr.copy()
    for d in detections:
        x0, y0, x1, y1 = (int(round(v)) for v in d["xyxy"])
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 1)
        cx, cy = (int(round(v)) for v in d["center"])
        cv2.circle(out, (cx, cy), 2, color, -1)
        cv2.putText(
            out,
            f"{d['conf']:.2f}",
            (x0, max(y0 - 2, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def _draw_clusters(bgr, clusters):
    out = bgr.copy()
    for cl in clusters:
        x0, y0, x1, y1 = (int(round(v)) for v in cl["bbox"])
        color = (0, 0, 255) if cl["merged"] else (255, 140, 0)
        cv2.rectangle(out, (x0 - 4, y0 - 4), (x1 + 4, y1 + 4), color, 2)
        for px, py in cl["points"]:
            cv2.circle(out, (int(px), int(py)), 3, color, -1)
    return out


def main():
    parser = argparse.ArgumentParser(description="Infer Braille dots with YOLOv8")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Overlay PNG path (default: <image>_yolo_dots.png)",
    )
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="Group YOLO dots into Braille cells via classical clustering",
    )
    parser.add_argument("--link-distance", type=float, default=15.0)
    parser.add_argument(
        "--tile",
        type=int,
        default=640,
        help="Tiled inference size (match training tile); 0 disables tiling",
    )
    parser.add_argument("--tile-overlap", type=int, default=96)
    parser.add_argument(
        "--compare-classical",
        action="store_true",
        help="Also run braille_cnn classical detector and save side-by-side",
    )
    args = parser.parse_args()

    if not args.image.exists():
        raise SystemExit(f"Image not found: {args.image}")

    weights = args.weights or _default_weights()
    # Allow falling back to a freshly downloaded base model only for smoke tests
    # — real detection needs fine-tuned weights.
    if not weights.exists():
        raise SystemExit(
            f"Weights not found: {weights}\n"
            "Train first: py -3.11 -m yolo_dot_detect.train"
        )

    # ensure repo root is importable when run as module
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from yolo_dot_detect.detect_dots import YoloDotDetector

    bgr = cv2.imread(str(args.image))
    if bgr is None:
        raise SystemExit(f"Failed to read image: {args.image}")

    detector = YoloDotDetector(
        weights=weights,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        tile=args.tile or None,
        tile_overlap=args.tile_overlap,
    )
    detections = detector.detect_boxes(bgr)
    centers = (
        np.array([d["center"] for d in detections], dtype=np.float64)
        if detections
        else np.zeros((0, 2), dtype=np.float64)
    )
    mode = f"tiled@{args.tile}" if args.tile else "whole-image"
    print(f"YOLO detections ({mode}): {len(detections)} dots")

    overlay = _draw_detections(bgr, detections)

    if args.cluster:
        from braille_cnn.dot_detect import cluster_into_cells

        clusters = cluster_into_cells(centers, link_distance=args.link_distance)
        n_ok = sum(1 for c in clusters if not c["merged"])
        n_merged = sum(1 for c in clusters if c["merged"])
        print(f"Clusters: {n_ok} cells, {n_merged} merged/ambiguous")
        overlay = _draw_clusters(overlay, clusters)

    if args.compare_classical:
        from braille_cnn.dot_detect import detect_dot_centers

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        classical = detect_dot_centers(gray)
        print(f"Classical detections: {len(classical)} peaks")
        classical_vis = bgr.copy()
        for x, y in classical:
            cv2.circle(classical_vis, (int(x), int(y)), 3, (255, 0, 255), -1)
        h = max(overlay.shape[0], classical_vis.shape[0])

        def _pad(img, h):
            if img.shape[0] == h:
                return img
            pad = np.zeros((h - img.shape[0], img.shape[1], 3), dtype=img.dtype)
            return np.vstack([img, pad])

        overlay = np.hstack([_pad(overlay, h), _pad(classical_vis, h)])
        cv2.putText(
            overlay, "YOLO", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 80), 2
        )
        cv2.putText(
            overlay,
            "Classical",
            (overlay.shape[1] // 2 + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 255),
            2,
        )

    out_path = args.out or args.image.with_name(args.image.stem + "_yolo_dots.png")
    cv2.imwrite(str(out_path), overlay)
    print(f"Saved overlay -> {out_path}")


if __name__ == "__main__":
    main()
