"""Wire the three fingertip detectors behind one factory.

Modules:
  tip_yolo.py       → TipYOLO          (YOLO26n; live-app default)
  tip_skin.py       → SkinContourTip   (classical CV fallback)
  tip_mediapipe.py  → MediaPipeTip     (landmark 8; optional)

Usage:
  from tip_backends import create_tip_backend
  tipper = create_tip_backend("auto")   # YOLO26, SkinContour if YOLO misses
  tip, box, conf = tipper.detect(frame_bgr)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from tip_mediapipe import MediaPipeTip
from tip_skin import SkinContourTip

# TipYOLO imported lazily so skin/mediapipe runs do not require ultralytics.


def _detector_name(det: Any) -> str:
    return str(getattr(det, "name", det.__class__.__name__))


class FallbackTip:
    """Try primary first (YOLO26 by default); if it misses, use fallback (skin)."""

    def __init__(self, primary, fallback) -> None:
        self.primary = primary
        self.fallback = fallback
        self.hand_visible = False
        self.last_backend = "none"

    def detect(self, frame_bgr):
        tip, box, conf = self.primary.detect(frame_bgr)
        self.hand_visible = bool(getattr(self.primary, "hand_visible", tip is not None))
        if tip is not None:
            self.last_backend = _detector_name(self.primary)
            return tip, box, conf
        if self.fallback is None:
            self.last_backend = "none"
            return None, None, 0.0
        out = self.fallback.detect(frame_bgr)
        fb_visible = bool(getattr(self.fallback, "hand_visible", out[0] is not None))
        self.hand_visible = self.hand_visible or fb_visible
        self.last_backend = (
            _detector_name(self.fallback) if out[0] is not None else "none"
        )
        return out

    def close(self) -> None:
        for det in (self.primary, self.fallback):
            if det is not None and hasattr(det, "close"):
                det.close()


def _try_tip_yolo(
    *,
    tip_weights: Optional[Path],
    tip_conf: float,
    imgsz: int,
    device: str,
    required: bool,
) -> Any:
    """Load TipYOLO. If required=False, missing weights/import return None."""
    try:
        from tip_yolo import TipYOLO

        return TipYOLO(
            weights=tip_weights,
            conf=tip_conf,
            imgsz=imgsz,
            device=device,
        )
    except (FileNotFoundError, ImportError) as exc:
        if required:
            raise
        print(f"TipYOLO unavailable ({exc}); using SkinContourTip.", flush=True)
        return None


def create_tip_backend(
    name: str = "auto",
    *,
    tip_weights: Optional[Path] = None,
    tip_conf: float = 0.25,
    imgsz: int = 640,
    device: str = "cpu",
) -> Any:
    """Build a tip detector. name: auto | yolo | skin | mediapipe.

    auto  — YOLO26 fingertip model, SkinContourTip if YOLO returns no tip
    yolo  — YOLO26 only (error if weights are missing)
    skin  — classical YCrCb contour (no neural net)
    mediapipe — Hands landmark 8 (needs a visible palm)
    """
    key = (name or "auto").lower().strip()
    if key == "skin":
        return SkinContourTip()
    if key == "mediapipe":
        return MediaPipeTip()
    if key == "yolo":
        return _try_tip_yolo(
            tip_weights=tip_weights,
            tip_conf=tip_conf,
            imgsz=imgsz,
            device=device,
            required=True,
        )
    if key == "auto":
        yolo = _try_tip_yolo(
            tip_weights=tip_weights,
            tip_conf=tip_conf,
            imgsz=imgsz,
            device=device,
            required=False,
        )
        if yolo is None:
            return SkinContourTip()
        return FallbackTip(yolo, SkinContourTip())
    raise ValueError(
        f"Unknown tip backend {name!r}. Use: auto | yolo | skin | mediapipe"
    )


__all__ = [
    "SkinContourTip",
    "MediaPipeTip",
    "FallbackTip",
    "create_tip_backend",
]
