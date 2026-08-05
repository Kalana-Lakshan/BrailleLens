"""YOLO Braille-dot detector API (drop-in centers for classical clustering).

Works with YOLO26 / YOLOv8 fine-tuned weights. detect_dot_centers_yolo()
returns (N, 2) float (x, y) centers — same shape as
braille_cnn.dot_detect.detect_dot_centers — so cluster_into_cells() works unchanged.

Models trained on tiles (see tile_dataset.py) must be run with tiled inference
so dots keep the pixel size they had during training. Pass tile=640.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _default_weights() -> Path:
    here = Path(__file__).resolve().parent / "runs" / "detect"
    for name in (
        "braille_dot_yolo26_tiled",
        "braille_dot_yolo26",
        "braille_dot_yolov8",
    ):
        cand = here / name / "weights" / "best.pt"
        if cand.exists():
            return cand
    return here / "braille_dot_yolo26_tiled" / "weights" / "best.pt"


_DEFAULT_WEIGHTS = _default_weights()


def _to_bgr(image):
    """Normalize supported inputs to an HxWx3 BGR ndarray."""
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
    # PIL Image
    arr = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _tile_origins(total: int, tile: int, stride: int):
    if total <= tile:
        return [0]
    origins = list(range(0, total - tile + 1, stride))
    if origins[-1] != total - tile:
        origins.append(total - tile)
    return origins


def _dedupe(detections, min_distance: float):
    """Greedy suppression of duplicate dots from overlapping tile seams."""
    if not detections:
        return []
    from scipy.spatial import cKDTree

    order = sorted(range(len(detections)), key=lambda i: -detections[i]["conf"])
    centers = np.array([d["center"] for d in detections], dtype=np.float64)
    tree = cKDTree(centers)
    suppressed = np.zeros(len(detections), dtype=bool)
    kept = []
    for i in order:
        if suppressed[i]:
            continue
        kept.append(detections[i])
        for j in tree.query_ball_point(centers[i], min_distance):
            if j != i:
                suppressed[j] = True
    return kept


class YoloDotDetector:
    """Lazy-loads a fine-tuned YOLO checkpoint and runs embossed-dot detection."""

    def __init__(
        self,
        weights: str | Path | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str = "cpu",
        max_det: int = 3000,
        tile: int | None = None,
        tile_overlap: int = 96,
    ):
        self.weights = Path(weights) if weights else _DEFAULT_WEIGHTS
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.max_det = max_det
        self.tile = tile
        self.tile_overlap = tile_overlap
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        if not self.weights.exists():
            raise FileNotFoundError(
                f"YOLO weights not found: {self.weights}\n"
                "Train first: py -3.11 -m yolo_dot_detect.train"
            )
        from ultralytics import YOLO

        self._model = YOLO(str(self.weights))

    def _predict_boxes(self, source, offset=(0, 0)):
        ox, oy = offset
        results = self._model.predict(
            source=source,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            max_det=self.max_det,
            verbose=False,
        )
        out = []
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return out
        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for box, conf in zip(xyxy, confs):
            x0, y0, x1, y1 = (float(v) for v in box)
            x0, x1 = x0 + ox, x1 + ox
            y0, y1 = y0 + oy, y1 + oy
            out.append(
                {
                    "xyxy": (x0, y0, x1, y1),
                    "conf": float(conf),
                    "center": ((x0 + x1) / 2.0, (y0 + y1) / 2.0),
                }
            )
        return out

    def detect_boxes(self, image):
        """Return list of dicts: {xyxy, conf, center} for each detection."""
        self._ensure_model()

        if not self.tile:
            return self._predict_boxes(image)

        bgr = _to_bgr(image)
        h, w = bgr.shape[:2]
        tile = self.tile
        stride = max(tile - self.tile_overlap, 1)

        detections = []
        for oy in _tile_origins(h, tile, stride):
            for ox in _tile_origins(w, tile, stride):
                crop = bgr[oy : oy + min(tile, h), ox : ox + min(tile, w)]
                detections.extend(self._predict_boxes(crop, offset=(ox, oy)))

        # dots inside one cell sit ~10px apart, so suppress well below that
        return _dedupe(detections, min_distance=max(self.tile_overlap * 0.05, 4.0))

    def detect(self, image) -> np.ndarray:
        """Detect raised Braille dots.

        image: path, BGR ndarray (OpenCV), grayscale ndarray, or PIL Image.
        Returns (N, 2) float64 array of (x, y) centers. Empty (0, 2) if none.
        """
        detections = self.detect_boxes(image)
        if not detections:
            return np.zeros((0, 2), dtype=np.float64)
        return np.array([d["center"] for d in detections], dtype=np.float64)


def detect_dot_centers_yolo(
    image,
    weights: str | Path | None = None,
    conf: float = 0.25,
    device: str = "cpu",
    tile: int | None = 640,
) -> np.ndarray:
    """Convenience wrapper matching classical detect_dot_centers signature."""
    detector = YoloDotDetector(
        weights=weights, conf=conf, device=device, tile=tile
    )
    return detector.detect(image)
