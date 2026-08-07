"""YOLO26 single-class fingertip detector (tip-only / no-palm friendly)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from ultralytics import YOLO

_HERE = Path(__file__).resolve().parent
DEFAULT_TIP_WEIGHTS = _HERE / "weights" / "yolo26n_fingertip_best.pt"
# Fallback if user left Colab download in package root
_ALT_TIP_WEIGHTS = _HERE / "yolo26n_fingertip_best.pt"


def resolve_tip_weights(path: Path | None = None) -> Path:
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Tip weights not found: {p}")
        return p
    for cand in (DEFAULT_TIP_WEIGHTS, _ALT_TIP_WEIGHTS):
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"Tip weights not found. Place best.pt at:\n  {DEFAULT_TIP_WEIGHTS}\n"
        f"or pass --tip-weights PATH"
    )


class TipYOLO:
    """Returns the highest-confidence fingertip center in pixel coords."""

    def __init__(
        self,
        weights: Path | None = None,
        conf: float = 0.25,
        imgsz: int = 640,
        device: str = "cpu",
    ) -> None:
        self.weights = resolve_tip_weights(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.model = YOLO(str(self.weights))

    def detect(
        self, bgr: np.ndarray
    ) -> tuple[Optional[tuple[int, int]], Optional[tuple[int, int, int, int]], float]:
        """
        Returns (center_xy, xyxy_box, conf). center/box are None if no detection.
        """
        r = self.model.predict(
            source=bgr,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]
        if r.boxes is None or len(r.boxes) == 0:
            return None, None, 0.0

        confs = r.boxes.conf.cpu().numpy()
        i = int(np.argmax(confs))
        xyxy = r.boxes.xyxy[i].cpu().numpy()
        x1, y1, x2, y2 = map(int, xyxy.tolist())
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        return (cx, cy), (x1, y1, x2, y2), float(confs[i])
