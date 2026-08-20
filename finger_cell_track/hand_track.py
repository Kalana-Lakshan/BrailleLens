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
import numpy as np

# MediaPipe Hands landmark index for index fingertip / MCP
INDEX_FINGERTIP = 8
INDEX_MCP = 5

_mp_hands = None
_mp_draw = None
_mp_styles = None


def _mediapipe():
    """Lazy import so --tip-backend skin does not need MediaPipe installed."""
    global _mp_hands, _mp_draw, _mp_styles
    if _mp_hands is None:
        import mediapipe as mp

        _mp_hands = mp.solutions.hands
        _mp_draw = mp.solutions.drawing_utils
        _mp_styles = mp.solutions.drawing_styles
    return _mp_hands, _mp_draw, _mp_styles


def open_source(source: str) -> cv2.VideoCapture:
    try:
        idx = int(source)
        cap = cv2.VideoCapture(idx)
    except (ValueError, TypeError):
        url = str(source)
        # Prefer default backend first — CAP_FFMPEG-only often hangs ~30s on IP Webcam.
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera source: {source!r}\n"
            "Use --source 0 for PC webcam or http://PHONE_IP:8080/video for IP Webcam.\n"
            "For fingertip-only checks prefer tip_dot_test.py (MJPEG reader)."
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
        mp_hands, _, _ = _mediapipe()
        self._hands = mp_hands.Hands(
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
    """Contact point from the largest skin blob that enters the frame.

    Used as the live-app default. MediaPipe fails on top-down Braille footage
    (palm cropped); this path does not need the palm.

    Rejects page-coloured blobs, thin edge strips, and corner ghosts.
    Contact is the point deepest into the page (away from the entry border),
    pulled slightly back toward the wrist so it sits on the pad, not the nail.
    """

    def __init__(
        self,
        min_area: int = 1200,
        max_area_frac: float = 0.22,
        contact_offset: float = 0.22,
        y_max: int = 200,
        border_px: int = 12,
        min_thickness: int = 28,
        min_solidity: float = 0.35,
        corner_reject_px: int = 48,
        corner_min_area: int = 5000,
    ) -> None:
        self.min_area = min_area
        self.max_area_frac = max_area_frac
        self.contact_offset = contact_offset
        self.y_max = y_max
        self.border_px = border_px
        self.min_thickness = min_thickness
        self.min_solidity = min_solidity
        self.corner_reject_px = corner_reject_px
        self.corner_min_area = corner_min_area
        self.hand_visible = False

    def _mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        # Cap Y so cream Braille paper is not treated as skin.
        mask = cv2.inRange(ycrcb, (40, 133, 77), (self.y_max, 173, 127))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
        return mask

    def _reject_blob(self, contour, area: float, x: int, y: int, bw: int, bh: int, w: int, h: int) -> bool:
        """True = reject. Thin edge strips and tiny corner ghosts."""
        if min(bw, bh) < self.min_thickness:
            return True
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        if hull_area > 1.0 and (area / hull_area) < self.min_solidity:
            return True
        # A strip glued to one edge with almost no depth into the page.
        b = self.border_px
        on_left = x <= b
        on_right = (x + bw) >= (w - b)
        on_top = y <= b
        on_bottom = (y + bh) >= (h - b)
        edge_count = int(on_left) + int(on_right) + int(on_top) + int(on_bottom)
        if edge_count == 1:
            if on_left or on_right:
                depth = bw
            else:
                depth = bh
            if depth < self.min_thickness * 2 and area < self.corner_min_area:
                return True
        return False

    def detect(self, frame_bgr: np.ndarray):
        h, w = frame_bgr.shape[:2]
        mask = self._mask(frame_bgr)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = self.max_area_frac * w * h
        best = None
        best_score = -1.0
        best_area = 0.0
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area or area > max_area:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            b = self.border_px
            touches = x <= b or y <= b or (x + bw) >= (w - b) or (y + bh) >= (h - b)
            if not touches:
                continue
            if self._reject_blob(contour, area, x, y, bw, bh, w, h):
                continue
            # Prefer larger blobs that reach deeper into the frame (any border).
            cx = x + 0.5 * bw
            cy = y + 0.5 * bh
            depth = min(cx, w - 1 - cx, cy, h - 1 - cy)
            score = area * (1.0 + depth / max(min(w, h), 1))
            if score > best_score:
                best_score = score
                best = contour
                best_area = area

        if best is None:
            self.hand_visible = False
            return None, None, 0.0

        self.hand_visible = True
        pts = best.reshape(-1, 2).astype(np.float32)
        if pts.shape[0] < 5:
            self.hand_visible = False
            return None, None, 0.0

        b = self.border_px
        on_border = (
            (pts[:, 0] <= b)
            | (pts[:, 1] <= b)
            | (pts[:, 0] >= (w - 1 - b))
            | (pts[:, 1] >= (h - 1 - b))
        )
        if on_border.any():
            wrist = pts[on_border].mean(axis=0)
        else:
            edge_dist = np.minimum.reduce(
                [pts[:, 0], w - 1 - pts[:, 0], pts[:, 1], h - 1 - pts[:, 1]]
            )
            wrist = pts[int(np.argmin(edge_dist))]

        # Contact = deepest into the page (far from frame edges), not nail tip.
        # Score blends inward depth with distance from the entry (wrist).
        inward = np.minimum.reduce(
            [pts[:, 0], w - 1 - pts[:, 0], pts[:, 1], h - 1 - pts[:, 1]]
        )
        from_wrist = np.linalg.norm(pts - wrist.reshape(1, 2), axis=1)
        pad_score = 0.65 * inward + 0.35 * from_wrist
        tip = pts[int(np.argmax(pad_score))]
        contact = tip - self.contact_offset * (tip - wrist)
        xy = (int(round(float(contact[0]))), int(round(float(contact[1]))))

        # Corner ghost: tip near two edges unless the blob is large enough.
        c = self.corner_reject_px
        near_corner = (
            (xy[0] <= c or xy[0] >= w - 1 - c)
            and (xy[1] <= c or xy[1] >= h - 1 - c)
        )
        if near_corner and best_area < self.corner_min_area:
            self.hand_visible = False
            return None, None, 0.0

        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        return xy, (int(x0), int(y0), int(x1), int(y1)), 1.0

    def close(self) -> None:
        return None


class FallbackTip:
    """MediaPipe first; skin-contour if the palm is out of frame."""

    def __init__(self, primary, fallback) -> None:
        self.primary = primary
        self.fallback = fallback
        self.hand_visible = False
        self.last_backend = "none"

    def detect(self, frame_bgr: np.ndarray):
        tip, box, conf = self.primary.detect(frame_bgr)
        self.hand_visible = bool(getattr(self.primary, "hand_visible", tip is not None))
        if tip is not None:
            self.last_backend = "primary"
            return tip, box, conf
        if self.fallback is None:
            self.last_backend = "none"
            return None, None, 0.0
        out = self.fallback.detect(frame_bgr)
        fb_visible = bool(getattr(self.fallback, "hand_visible", out[0] is not None))
        self.hand_visible = self.hand_visible or fb_visible
        self.last_backend = "fallback" if out[0] is not None else "none"
        return out

    def close(self) -> None:
        for det in (self.primary, self.fallback):
            if det is not None and hasattr(det, "close"):
                det.close()


def index_tip_px(
    hand_landmarks,
    width: int,
    height: int,
) -> tuple[int, int]:
    lm = hand_landmarks.landmark[INDEX_FINGERTIP]
    return int(lm.x * width), int(lm.y * height)


def process_frame(
    frame_bgr: np.ndarray,
    hands,
) -> tuple[np.ndarray, Optional[tuple[int, int]]]:
    mp_hands, mp_draw, mp_styles = _mediapipe()
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    result = hands.process(rgb)
    rgb.flags.writeable = True
    out = frame_bgr
    tip: Optional[tuple[int, int]] = None

    if result.multi_hand_landmarks:
        for hand_lms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                out,
                hand_lms,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
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
