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


def _iou(a: tuple, b: tuple) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = (ax1 - ax0) * (ay1 - ay0)
    b_area = (bx1 - bx0) * (by1 - by0)
    return inter / (a_area + b_area - inter)


def _merge_detections(dets: list[dict], iou_thresh: float = 0.5) -> list[dict]:
    """Greedy NMS merge across two detection passes (e.g. full page + a
    zoomed-in strip), highest confidence first."""
    order = sorted(range(len(dets)), key=lambda i: -dets[i]["conf"])
    keep = []
    used = [False] * len(dets)
    for i in order:
        if used[i]:
            continue
        keep.append(dets[i])
        for j in order:
            if used[j] or j == i:
                continue
            if _iou(dets[i]["xyxy"], dets[j]["xyxy"]) > iou_thresh:
                used[j] = True
        used[i] = True
    return keep


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

    def _predict_raw(self, bgr) -> list[dict]:
        """Run the model on one already-prepared BGR array. Boxes come back
        in that array's own pixel coordinates -- caller remaps if needed."""
        self._ensure_model()
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
            out.append({"xyxy": (x0, y0, x1, y1), "conf": float(conf)})
        return out

    def detect_boxes(
        self,
        image,
        spine_boost: bool = False,
        spine_strip_frac: float = 0.45,
        spine_upscale: float = 2.0,
    ) -> list[dict]:
        """Return [{xyxy, conf, center}, ...] in page-pixel coordinates.

        spine_boost=True adds a second detection pass on an upscaled
        spine-proximal strip (the left spine_strip_frac of the page) and
        merges it with the full-page pass by NMS. On an open-book photo,
        page curvature near the spine makes cells there ~6-7% smaller than
        elsewhere, which measurably suppresses detection confidence -- not
        a brightness/contrast effect (ruled out), not fixable by a lower or
        size-adaptive confidence threshold alone (tried, net worse: more
        false positives than recovered true positives). Detecting that strip
        again at higher effective resolution recovers some of that lost
        confidence instead. Validated on held-out gold test pages: F1
        0.742 -> 0.760 at the defaults here (see the "Failure analysis"
        section of reports/eval/gold_cell_detector_finetune.md for the full
        diagnosis and a sweep over strip_frac/upscale). Only worth using on
        genuine open-book-spread photos -- it repeats work for no benefit on
        a flat scan or single loose page.
        """
        bgr = _to_bgr(image)
        base = self._predict_raw(bgr)
        if spine_boost:
            h, w = bgr.shape[:2]
            strip_w = max(int(w * spine_strip_frac), 1)
            strip = bgr[:, :strip_w]
            sw, sh = int(strip_w * spine_upscale), int(h * spine_upscale)
            if sw > 0 and sh > 0:
                import cv2

                strip_up = cv2.resize(strip, (sw, sh), interpolation=cv2.INTER_CUBIC)
                strip_dets = self._predict_raw(strip_up)
                for d in strip_dets:
                    x0, y0, x1, y1 = d["xyxy"]
                    d["xyxy"] = (x0 / spine_upscale, y0 / spine_upscale, x1 / spine_upscale, y1 / spine_upscale)
                base = _merge_detections(base + strip_dets)
        return [
            {**d, "center": ((d["xyxy"][0] + d["xyxy"][2]) / 2.0, (d["xyxy"][1] + d["xyxy"][3]) / 2.0)}
            for d in base
        ]

    def detect(self, image) -> np.ndarray:
        """(N, 4) float64 array of xyxy boxes. Empty (0, 4) if none."""
        dets = self.detect_boxes(image)
        if not dets:
            return np.zeros((0, 4), dtype=np.float64)
        return np.array([d["xyxy"] for d in dets], dtype=np.float64)


def detect_cells(image, **kwargs) -> list[dict]:
    """Convenience wrapper around CellDetector.detect_boxes."""
    return CellDetector(**kwargs).detect_boxes(image)
