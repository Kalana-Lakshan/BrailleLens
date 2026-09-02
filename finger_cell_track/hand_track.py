"""CLI demo: MediaPipe Hands index-tip overlay (PC webcam / IP Webcam).

Tip detectors live in separate modules — see tip_backends.py:
  tip_yolo.py       TipYOLO          (YOLO26n default)
  tip_skin.py       SkinContourTip   (fallback)
  tip_mediapipe.py  MediaPipeTip

From BrailleLens repo root:

    finger_cell_track\\.venv\\Scripts\\python.exe finger_cell_track/hand_track.py --source 0
    finger_cell_track\\.venv\\Scripts\\python.exe finger_cell_track/hand_track.py --source http://IP:8080/video

Index fingertip = landmark 8 (yellow). Q = quit.

Backward-compatible re-exports (prefer tip_backends / tip_skin / …):
    from hand_track import SkinContourTip, MediaPipeTip, FallbackTip, open_source
"""

from __future__ import annotations

import argparse
import time

import cv2

from camera_source import open_source
from tip_backends import FallbackTip, MediaPipeTip, SkinContourTip  # noqa: F401
from tip_mediapipe import INDEX_FINGERTIP, _mediapipe, process_frame

# Re-export for older imports
__all__ = [
    "open_source",
    "SkinContourTip",
    "MediaPipeTip",
    "FallbackTip",
    "INDEX_FINGERTIP",
    "process_frame",
]


def main() -> None:
    p = argparse.ArgumentParser(description="MediaPipe Hands index-tip live tracker")
    p.add_argument(
        "--source",
        default="0",
        help="Webcam index (0) or IP Webcam URL",
    )
    p.add_argument("--max-hands", type=int, default=1)
    p.add_argument("--detection-conf", type=float, default=0.6)
    p.add_argument("--tracking-conf", type=float, default=0.5)
    p.add_argument("--display-width", type=int, default=960)
    args = p.parse_args()

    print(f"Opening {args.source!r} ...", flush=True)
    cap = open_source(args.source)
    mp_hands, _, _ = _mediapipe()
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=args.max_hands,
        model_complexity=1,
        min_detection_confidence=args.detection_conf,
        min_tracking_confidence=args.tracking_conf,
    )

    win = "finger_cell_track — Hand tip (Q quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("Live. Show your hand (palm visible). Press Q to quit.", flush=True)

    t_prev = time.time()
    fps = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Frame grab failed — check camera / IP Webcam.", flush=True)
                time.sleep(0.2)
                continue

            out, tip = process_frame(frame, hands)
            now = time.time()
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

            hud = f"FPS={fps:.1f}  tip={'yes' if tip else 'no'}  landmark={INDEX_FINGERTIP}"
            cv2.putText(
                out,
                hud,
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 120),
                2,
                cv2.LINE_AA,
            )

            h, w = out.shape[:2]
            if w > args.display_width:
                scale = args.display_width / w
                out = cv2.resize(
                    out,
                    (args.display_width, max(1, int(h * scale))),
                    interpolation=cv2.INTER_AREA,
                )

            cv2.imshow(win, out)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q"), 27):
                break
    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Stopped.", flush=True)


if __name__ == "__main__":
    main()
