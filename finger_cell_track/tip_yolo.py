"""Fingertip method — TipYOLO (fine-tuned YOLO26n; live-app default).

Single-class fingertip box → center. Tip-only / no-palm friendly.
Wire via tip_backends.create_tip_backend("auto") (SkinContourTip fallback)
or create_tip_backend("yolo") (YOLO only).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from ultralytics import YOLO

_HERE = Path(__file__).resolve().parent
# Prefer the Braille-domain fine-tune; fall back to the generic fingertip weights.
_DOMAIN_TIP_WEIGHTS = _HERE / "weights" / "yolo26n_fingertip_braille_best.pt"
DEFAULT_TIP_WEIGHTS = _HERE / "weights" / "yolo26n_fingertip_best.pt"
_ALT_TIP_WEIGHTS = _HERE / "yolo26n_fingertip_best.pt"


def resolve_tip_weights(path: Path | None = None) -> Path:
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Tip weights not found: {p}")
        return p
    for cand in (_DOMAIN_TIP_WEIGHTS, DEFAULT_TIP_WEIGHTS, _ALT_TIP_WEIGHTS):
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"Tip weights not found. Place best.pt at:\n  {_DOMAIN_TIP_WEIGHTS}\n"
        f"or {DEFAULT_TIP_WEIGHTS}\nor pass --tip-weights PATH"
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
        self.name = "yolo"
        self.weights = resolve_tip_weights(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.hand_visible = False
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
            self.hand_visible = False
            return None, None, 0.0

        confs = r.boxes.conf.cpu().numpy()
        i = int(np.argmax(confs))
        xyxy = r.boxes.xyxy[i].cpu().numpy()
        x1, y1, x2, y2 = map(int, xyxy.tolist())
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        self.hand_visible = True
        return (cx, cy), (x1, y1, x2, y2), float(confs[i])
