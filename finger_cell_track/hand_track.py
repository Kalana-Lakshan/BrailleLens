"""Live MediaPipe Hands — index fingertip overlay (PC webcam / IP Webcam).

From BrailleLens repo root:

    finger_cell_track\\.venv\\Scripts\\python.exe finger_cell_track/hand_track.py --source 0
    finger_cell_track\\.venv\\Scripts\\python.exe finger_cell_track/hand_track.py --source http://IP:8080/video

Index fingertip = landmark 8 (yellow). Q = quit.
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Hands landmark index for index fingertip
INDEX_FINGERTIP = 8

_mp_hands = mp.solutions.hands
_mp_draw = mp.solutions.drawing_utils
_mp_styles = mp.solutions.drawing_styles


def open_source(source: str) -> cv2.VideoCapture:
    try:
        idx = int(source)
        cap = cv2.VideoCapture(idx)
    except (ValueError, TypeError):
        cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera source: {source!r}\n"
            "Use --source 0 for PC webcam or http://PHONE_IP:8080/video for IP Webcam."
        )
    return cap


def index_tip_px(
    hand_landmarks,
    width: int,
    height: int,
) -> tuple[int, int]:
    lm = hand_landmarks.landmark[INDEX_FINGERTIP]
    return int(lm.x * width), int(lm.y * height)


def process_frame(
    frame_bgr: np.ndarray,
    hands: _mp_hands.Hands,
) -> tuple[np.ndarray, Optional[tuple[int, int]]]:
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    result = hands.process(rgb)
    rgb.flags.writeable = True
    out = frame_bgr
    tip: Optional[tuple[int, int]] = None

    if result.multi_hand_landmarks:
        for hand_lms in result.multi_hand_landmarks:
            _mp_draw.draw_landmarks(
                out,
                hand_lms,
                _mp_hands.HAND_CONNECTIONS,
                _mp_styles.get_default_hand_landmarks_style(),
                _mp_styles.get_default_hand_connections_style(),
            )
            tip = index_tip_px(hand_lms, w, h)
            cv2.circle(out, tip, 10, (0, 255, 255), -1)  # yellow tip
            cv2.circle(out, tip, 12, (0, 128, 255), 2)
            cv2.putText(
                out,
                f"tip={tip[0]},{tip[1]}",
                (tip[0] + 14, tip[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            break  # first hand only
    return out, tip


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
    hands = _mp_hands.Hands(
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
