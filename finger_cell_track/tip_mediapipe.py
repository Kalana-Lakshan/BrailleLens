"""Fingertip method 2/3 — MediaPipe Hands (landmark 8 = index tip).

Needs a visible palm. Weak on tip-only / over-page Braille views.
Wire via tip_backends.create_tip_backend("mediapipe").
"""

from __future__ import annotations

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
        self.name = "mediapipe"
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
        tip = np.array(
            [lm[INDEX_FINGERTIP].x * w, lm[INDEX_FINGERTIP].y * h], dtype=np.float32
        )
        mcp = np.array([lm[INDEX_MCP].x * w, lm[INDEX_MCP].y * h], dtype=np.float32)
        contact = tip - self.contact_offset * (tip - mcp)
        xy = (int(round(contact[0])), int(round(contact[1])))
        return xy, None, 1.0

    def close(self) -> None:
        self._hands.close()


def index_tip_px(hand_landmarks, width: int, height: int) -> tuple[int, int]:
    lm = hand_landmarks.landmark[INDEX_FINGERTIP]
    return int(lm.x * width), int(lm.y * height)


def process_frame(
    frame_bgr: np.ndarray,
    hands,
) -> tuple[np.ndarray, Optional[tuple[int, int]]]:
    """Draw hand landmarks + yellow tip (used by hand_track.py CLI demo)."""
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
