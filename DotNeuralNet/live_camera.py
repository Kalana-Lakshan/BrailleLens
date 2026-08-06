"""Live IP Webcam / webcam preview with DotNeuralNet YOLO cell boxes.

Run from the BrailleLens project root (Python 3.11):

    py -3.11 DotNeuralNet/live_camera.py --source http://192.168.x.x:8080/video

Controls (preview window):
    Q   quit
    S   force inference on current frame
    D   toggle box drawing
    + / -   raise / lower confidence (step 0.05)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_DNN = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_DEFAULT_WEIGHTS = _DNN / "weights" / "yolov8_braille.pt"


def _open_source(source: str) -> cv2.VideoCapture:
    try:
        idx = int(source)
        cap = cv2.VideoCapture(idx)
    except (ValueError, TypeError):
        cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera source: {source!r}\n"
            "For IP Webcam use: http://PHONE_IP:8080/video\n"
            "Phone and PC must be on the same Wi-Fi."
        )
    return cap


def _fit(frame: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame, 1.0
    scale = max_width / w
    return (
        cv2.resize(
            frame,
            (max_width, max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        ),
        scale,
    )


def main():
    p = argparse.ArgumentParser(description="Live DotNeuralNet Braille cell boxes")
    p.add_argument(
        "--source",
        required=True,
        help="IP Webcam URL (http://IP:8080/video) or webcam index (0)",
    )
    p.add_argument(
        "--weights",
        type=Path,
        default=_DEFAULT_WEIGHTS,
        help="Path to yolov8_braille.pt",
    )
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument(
        "--infer-interval",
        type=float,
        default=0.8,
        help="Seconds between YOLO runs (lower = more live, heavier CPU/GPU)",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default="cpu", help="cpu or 0 for CUDA")
    p.add_argument("--display-width", type=int, default=960)
    p.add_argument(
        "--labels",
        action="store_true",
        help="Draw class/conf text on each box (can clutter dense pages)",
    )
    args = p.parse_args()

    if not args.weights.exists():
        raise SystemExit(f"Weights not found: {args.weights}")

    from ultralytics import YOLO

    print(f"Loading {args.weights} ...", flush=True)
    model = YOLO(str(args.weights))
    print(f"Opening {args.source} ...", flush=True)
    cap = _open_source(args.source)
    print("Camera opened. Waiting for first frame...", flush=True)

    conf = args.conf
    show_boxes = True
    show_labels = args.labels
    last_boxes = None
    last_names = None
    last_n = 0
    last_infer_t = 0.0
    last_infer_ms = 0.0
    force = False

    win = "DotNeuralNet — Live (Q quit, S infer, D boxes, +/- conf)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("Live. Press Q to quit.", flush=True)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Frame grab failed — check IP Webcam URL / Wi-Fi.")
                time.sleep(0.3)
                continue

            now = time.time()
            should_infer = force or (now - last_infer_t) >= args.infer_interval
            if should_infer:
                force = False
                t0 = time.time()
                res = model.predict(
                    source=frame,
                    conf=conf,
                    imgsz=args.imgsz,
                    device=args.device,
                    verbose=False,
                )
                last_infer_ms = (time.time() - t0) * 1000.0
                last_infer_t = now
                if res and res[0].boxes is not None:
                    last_boxes = res[0].boxes
                    last_names = res[0].names
                    last_n = len(last_boxes)
                else:
                    last_boxes = None
                    last_n = 0

            preview, scale = _fit(frame, args.display_width)
            if show_boxes and last_boxes is not None:
                # redraw using stored boxes in original frame coords
                xyxy = last_boxes.xyxy.cpu().numpy()
                confs = last_boxes.conf.cpu().numpy()
                clss = last_boxes.cls.cpu().numpy().astype(int)
                for (x0, y0, x1, y1), c, cls_id in zip(xyxy, confs, clss):
                    sx0, sy0 = int(round(x0 * scale)), int(round(y0 * scale))
                    sx1, sy1 = int(round(x1 * scale)), int(round(y1 * scale))
                    cv2.rectangle(preview, (sx0, sy0), (sx1, sy1), (0, 200, 80), 1)
                    if show_labels:
                        name = (
                            last_names.get(int(cls_id), str(cls_id))
                            if isinstance(last_names, dict)
                            else str(cls_id)
                        )
                        cv2.putText(
                            preview,
                            f"{name} {c:.2f}",
                            (sx0, max(sy0 - 2, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            (0, 200, 80),
                            1,
                            cv2.LINE_AA,
                        )

            hud = [
                f"cells={last_n}  conf={conf:.2f}  infer={last_infer_ms:.0f}ms",
                f"interval={args.infer_interval:.1f}s  boxes={'ON' if show_boxes else 'OFF'}",
            ]
            y = 28
            for line in hud:
                cv2.putText(
                    preview,
                    line,
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 120),
                    2,
                    cv2.LINE_AA,
                )
                y += 28

            cv2.imshow(win, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("s"), ord("S")):
                force = True
            if key in (ord("d"), ord("D")):
                show_boxes = not show_boxes
            if key in (ord("+"), ord("=")):
                conf = min(0.95, conf + 0.05)
            if key in (ord("-"), ord("_")):
                conf = max(0.05, conf - 0.05)
            if key in (ord("l"), ord("L")):
                show_labels = not show_labels
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Stopped.")


if __name__ == "__main__":
    main()
