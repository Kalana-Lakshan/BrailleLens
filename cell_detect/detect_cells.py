"""Stage 4a inference API: find Braille cell boxes on a page.

Mirrors yolo_dot_detect.detect_dots.YoloDotDetector, but returns *cell*
boxes (not dot centres). Used by braille_cnn.recognize.recognize_page().

    from cell_detect import CellDetector
    boxes = CellDetector().detect_boxes(image)   # list[{xyxy, conf, center}]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = HERE / "weights" / "braille_cell_best.pt"


def _default_weights() -> Path:
    if DEFAULT_WEIGHTS.exists():
        return DEFAULT_WEIGHTS
    runs = HERE / "runs" / "detect"
    for name in ("braille_cell_yolo26", "smoke_test"):
        cand = runs / name / "weights" / "best.pt"
        if cand.exists():
            return cand
    return DEFAULT_WEIGHTS


def _to_bgr(image):
    import cv2

    if isinstance(image, (str, Path)):
        arr = cv2.imread(str(image))
        if arr is None:
            raise FileNotFoundError(f"Could not read image: {image}")
        return arr
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image
    arr = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


class CellDetector:
    """Lazy-loads the Stage 4a checkpoint and returns cell boxes."""

    def __init__(
        self,
        weights: str | Path | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 1280,
        device: str = "cpu",
        max_det: int = 800,
    ) -> None:
        self.weights = Path(weights) if weights else _default_weights()
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.max_det = max_det
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        if not self.weights.exists():
            raise FileNotFoundError(
                f"Cell-detector weights not found: {self.weights}\n"
                "Train on Colab, then copy best.pt to cell_detect/weights/"
                "braille_cell_best.pt\n"
                "See cell_detect/COLAB_SETUP.md"
            )
        from ultralytics import YOLO

        self._model = YOLO(str(self.weights))

    def detect_boxes(self, image) -> list[dict]:
        """Return [{xyxy, conf, center}, ...] in page-pixel coordinates."""
        self._ensure_model()
        bgr = _to_bgr(image)
        results = self._model.predict(
            source=bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            max_det=self.max_det,
            verbose=False,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []
        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        out = []
        for box, conf in zip(xyxy, confs):
            x0, y0, x1, y1 = (float(v) for v in box)
            out.append(
                {
                    "xyxy": (x0, y0, x1, y1),
                    "conf": float(conf),
                    "center": ((x0 + x1) / 2.0, (y0 + y1) / 2.0),
                }
            )
        return out

    def detect(self, image) -> np.ndarray:
        """(N, 4) float64 array of xyxy boxes. Empty (0, 4) if none."""
        dets = self.detect_boxes(image)
        if not dets:
            return np.zeros((0, 4), dtype=np.float64)
        return np.array([d["xyxy"] for d in dets], dtype=np.float64)


def detect_cells(image, **kwargs) -> list[dict]:
    """Convenience wrapper around CellDetector.detect_boxes."""
    return CellDetector(**kwargs).detect_boxes(image)
