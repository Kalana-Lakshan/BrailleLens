"""Live fingertip YOLO overlay (PC webcam / IP Webcam). No MediaPipe.

    finger_cell_track\\.venv\\Scripts\\python.exe finger_cell_track/tip_track.py --source 0
    finger_cell_track\\.venv\\Scripts\\python.exe finger_cell_track/tip_track.py --source http://PHONE_IP:8080/video
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from camera_source import open_source  # noqa: E402
from tip_backends import create_tip_backend  # noqa: E402



def main() -> None:
    p = argparse.ArgumentParser(description="YOLO fingertip live test")
    p.add_argument("--source", default="0")
    p.add_argument("--tip-weights", type=Path, default=None)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="cpu")
    p.add_argument("--display-width", type=int, default=960)
    args = p.parse_args()

    tipper = create_tip_backend(
        "yolo",
        tip_weights=args.tip_weights,
        tip_conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
    )
    print(f"Tip weights: {tipper.weights}", flush=True)
    cap = open_source(args.source)
    win = "tip_track — YOLO fingertip (Q quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    t0, n = time.time(), 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            tip, box, conf = tipper.detect(frame)
            out = frame.copy()
            if box is not None:
                x1, y1, x2, y2 = box
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if tip is not None:
                cv2.circle(out, tip, 8, (0, 255, 255), -1)
                cv2.putText(
                    out,
                    f"tip {conf:.2f}",
                    (tip[0] + 12, tip[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
            n += 1
            if n % 30 == 0:
                fps = n / max(time.time() - t0, 1e-6)
                print(f"fps~{fps:.1f} tip={'Y' if tip else 'N'}", flush=True)
            h, w = out.shape[:2]
            if w > args.display_width:
                s = args.display_width / w
                out = cv2.resize(out, (args.display_width, int(h * s)))
            cv2.imshow(win, out)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
