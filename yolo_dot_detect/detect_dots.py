"""YOLOv8 Braille-dot detector API (drop-in centers for classical clustering).

detect_dot_centers_yolo() returns an (N, 2) float array of (x, y) pixel
centers — same shape as braille_cnn.dot_detect.detect_dot_centers — so you
can reuse cluster_into_cells() unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_DEFAULT_WEIGHTS = (
    Path(__file__).resolve().parent
    / "runs"
    / "detect"
    / "braille_dot_yolov8"
    / "weights"
    / "best.pt"
)


class YoloDotDetector:
    """Lazy-loads a fine-tuned YOLOv8 checkpoint and runs embossed-dot detection."""

    def __init__(
        self,
        weights: str | Path | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str = "cpu",
        max_det: int = 3000,
    ):
        self.weights = Path(weights) if weights else _DEFAULT_WEIGHTS
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
                f"YOLOv8 weights not found: {self.weights}\n"
                "Train first: py -3.11 -m yolo_dot_detect.train"
            )
        from ultralytics import YOLO

        self._model = YOLO(str(self.weights))

    def detect(self, image) -> np.ndarray:
        """Detect raised Braille dots.

        image: path, BGR ndarray (OpenCV), RGB ndarray, or PIL Image.
        Returns (N, 2) float64 array of (x, y) centers. Empty (0, 2) if none.
        """
        self._ensure_model()
        results = self._model.predict(
            source=image,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            max_det=self.max_det,
            verbose=False,
        )
        if not results:
            return np.zeros((0, 2), dtype=np.float64)

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 2), dtype=np.float64)

        xyxy = boxes.xyxy.cpu().numpy()
        cx = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
        cy = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
        return np.stack([cx, cy], axis=1).astype(np.float64)

    def detect_boxes(self, image):
        """Return list of dicts: {xyxy, conf, center} for each detection."""
        self._ensure_model()
        results = self._model.predict(
            source=image,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            max_det=self.max_det,
            verbose=False,
        )
        out = []
        if not results or results[0].boxes is None:
            return out
        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for box, conf in zip(xyxy, confs):
            x0, y0, x1, y1 = map(float, box)
            out.append(
                {
                    "xyxy": (x0, y0, x1, y1),
                    "conf": float(conf),
                    "center": ((x0 + x1) / 2.0, (y0 + y1) / 2.0),
                }
            )
        return out


def detect_dot_centers_yolo(
    image,
    weights: str | Path | None = None,
    conf: float = 0.25,
    device: str = "cpu",
) -> np.ndarray:
    """Convenience wrapper matching classical detect_dot_centers signature."""
    detector = YoloDotDetector(weights=weights, conf=conf, device=device)
    return detector.detect(image)
