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

# MediaPipe Hands landmark index for index fingertip / MCP
INDEX_FINGERTIP = 8
INDEX_MCP = 5

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


class MediaPipeTip:
    """Reusable index-finger tip. Landmark 8, optionally pulled back toward MCP 5."""

    def __init__(
        self,
        max_hands: int = 1,
        detection_conf: float = 0.6,
        tracking_conf: float = 0.5,
        contact_offset: float = 0.18,
    ) -> None:
        self.contact_offset = contact_offset
        self._hands = _mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=1,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self.hand_visible = False

    def detect(self, frame_bgr: np.ndarray):
        """Return ((x, y), None, 1.0) or (None, None, 0.0). Same shape as TipYOLO."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._hands.process(rgb)
        rgb.flags.writeable = True
        if not result.multi_hand_landmarks:
            self.hand_visible = False
            return None, None, 0.0
        self.hand_visible = True
        lm = result.multi_hand_landmarks[0].landmark
        tip = np.array([lm[INDEX_FINGERTIP].x * w, lm[INDEX_FINGERTIP].y * h], dtype=np.float32)
        mcp = np.array([lm[INDEX_MCP].x * w, lm[INDEX_MCP].y * h], dtype=np.float32)
        contact = tip - self.contact_offset * (tip - mcp)
        xy = (int(round(contact[0])), int(round(contact[1])))
        return xy, None, 1.0

    def close(self) -> None:
        self._hands.close()


class SkinContourTip:
    """Classical fallback: largest skin blob, tip = point farthest from the wrist edge."""

    def __init__(self, min_area: int = 800) -> None:
        self.min_area = min_area
        self.hand_visible = False

    def detect(self, frame_bgr: np.ndarray):
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
        mask = cv2.medianBlur(mask, 5)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.hand_visible = False
            return None, None, 0.0
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < self.min_area:
            self.hand_visible = False
            return None, None, 0.0
        self.hand_visible = True
        pts = contour.reshape(-1, 2)
        # Wrist is the side of the blob closest to a frame border.
        h, w = frame_bgr.shape[:2]
        border = np.array(
            [
                pts[:, 0],
                w - 1 - pts[:, 0],
                pts[:, 1],
                h - 1 - pts[:, 1],
            ]
        ).min(axis=0)
        wrist = pts[int(np.argmin(border))]
        tip = pts[int(np.argmax(np.sum((pts - wrist) ** 2, axis=1)))]
        xy = (int(tip[0]), int(tip[1]))
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        return xy, (int(x0), int(y0), int(x1), int(y1)), 1.0

    def close(self) -> None:
        return None


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
