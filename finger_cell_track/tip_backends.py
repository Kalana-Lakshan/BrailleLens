"""Wire the three fingertip detectors behind one factory.

Modules:
  tip_skin.py       → SkinContourTip   (default classical CV)
  tip_mediapipe.py  → MediaPipeTip     (landmark 8)
  tip_yolo.py       → TipYOLO          (fine-tuned YOLO26n)

Usage:
  from tip_backends import create_tip_backend
  tipper = create_tip_backend("skin")
  tip, box, conf = tipper.detect(frame_bgr)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from tip_mediapipe import MediaPipeTip
from tip_skin import SkinContourTip

# TipYOLO imported lazily inside create_tip_backend("yolo") so skin/mediapipe
# runs do not require ultralytics / tip weights.


class FallbackTip:
    """Try primary first (usually MediaPipe); if it fails, use fallback (skin)."""

    def __init__(self, primary, fallback) -> None:
        self.primary = primary
        self.fallback = fallback
        self.hand_visible = False
        self.last_backend = "none"

    def detect(self, frame_bgr):
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


def create_tip_backend(
    name: str = "skin",
    *,
    tip_weights: Optional[Path] = None,
    tip_conf: float = 0.25,
    imgsz: int = 640,
    device: str = "cpu",
) -> Any:
    """Build a tip detector. name: skin | mediapipe | yolo | auto."""
    key = (name or "skin").lower().strip()
    if key == "skin":
        return SkinContourTip()
    if key == "mediapipe":
        return MediaPipeTip()
    if key == "auto":
        return FallbackTip(MediaPipeTip(), SkinContourTip())
    if key == "yolo":
        from tip_yolo import TipYOLO

        return TipYOLO(
            weights=tip_weights,
            conf=tip_conf,
            imgsz=imgsz,
            device=device,
        )
    raise ValueError(
        f"Unknown tip backend {name!r}. Use: skin | mediapipe | yolo | auto"
    )


# Re-exports so callers can `from tip_backends import SkinContourTip, ...`
__all__ = [
    "SkinContourTip",
    "MediaPipeTip",
    "FallbackTip",
    "create_tip_backend",
]
