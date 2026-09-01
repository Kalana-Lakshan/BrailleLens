"""Pre-scan: run the chosen detection+classification pipeline ONCE on a
clear, unoccluded photo of a page, producing a lookup table of every
cell's position and character. This is what makes reading under finger
occlusion possible at all -- a cell hidden by the reading finger in the
live frame was already read during the pre-scan, before anything covered
it, so the live loop only ever needs to know "which cell is the finger
over", not "what does the covered cell look like right now".

Uses the pipeline the team has adopted: yolo_dot_detect (learned dot
detection) -> braille_cnn.dot_detect.cluster_into_cells (classical
clustering, unchanged) -> braille_cnn's SimpleBrailleCNN (classification,
unchanged). See yolo_dot_detect/README.md's "Relation to the rest of
BrailleLens" section -- this only swaps the detection stage.

The page photo is preprocessed (CLAHE contrast, best-effort deskew --
cell_detect/preprocess.py, shared with CellDetector) before any of that
runs; see scan_page()'s docstring for why no coordinate remapping is
needed here even though it is in cell_detect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from braille_cnn.dot_detect import cluster_into_cells
from braille_cnn.infer_page import _classify, _cluster_crop_box, _estimate_cell_size, load_model
from braille_cnn.labels import code_to_label
from cell_detect.preprocess import apply_clahe, deskew_page
from yolo_dot_detect.detect_dots import YoloDotDetector


@dataclass
class Cell:
    center: tuple[float, float]  # (x, y) pixel position in the reference (pre-scan) frame
    code: int
    label: str
    confidence: float


@dataclass
class PageScan:
    """Reference frame + its cell lookup table. `reference_gray` is kept
    for registration.py to match live frames against."""
    reference_gray: np.ndarray
    cells: list[Cell] = field(default_factory=list)

    def nearest_cell(self, point: tuple[float, float], max_distance: float | None = None) -> Cell | None:
        """Cell whose pre-scan center is closest to `point` (already in
        this scan's reference-frame coordinates -- see registration.py to
        get there from a live frame). Returns None if nothing is within
        max_distance (when given)."""
        if not self.cells:
            return None
        centers = np.array([c.center for c in self.cells])
        d = np.linalg.norm(centers - np.array(point), axis=1)
        i = int(np.argmin(d))
        if max_distance is not None and d[i] > max_distance:
            return None
        return self.cells[i]


def scan_page(
    image_path: str | Path,
    braille_checkpoint: str | Path,
    yolo_dot_weights: str | Path | None = None,
    device: str | None = None,
    dot_conf: float = 0.25,
    link_distance: float | None = None,
    margin_scale: float = 0.8,
    img_size: int = 64,
    apply_clahe_preproc: bool = True,
    clahe_clip_limit: float = 2.5,
    deskew: bool = True,
    deskew_min_area_frac: float = 0.25,
) -> PageScan:
    """Runs the full pre-scan pipeline on one page photo and returns its
    lookup table. Call this once per page/session, before reading starts --
    not per-frame.

    apply_clahe_preproc / clahe_clip_limit / deskew mirror cell_detect's
    CellDetector preprocessing (see cell_detect/preprocess.py) -- both
    default ON. Unlike CellDetector, no box-remapping step is needed here:
    detection, cell cropping, and the reference frame stored in
    PageScan.reference_gray all run on the SAME (possibly deskewed/CLAHE'd)
    image, so live_loop.py's registration + nearest_cell lookup stays
    internally consistent regardless of whether that frame matches the raw
    photo's own pixel coordinates -- nothing outside this module reads
    reference_gray or Cell.center against the original, unprocessed photo.
    CAUTION: apply_clahe_preproc was measured to hurt cell detection on a
    real phone photo (see cell_detect/preprocess.py's docstring) -- being
    on by default here is so it can be checked on more real photos, not a
    claim it will help. Pass apply_clahe_preproc=False if it hurts on yours.
    """
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    image = Image.open(image_path).convert("L")
    gray = np.asarray(image, dtype=np.uint8)

    if deskew:
        gray, _inverse_matrix = deskew_page(gray, min_area_frac=deskew_min_area_frac)
        # No remap needed (see docstring above) -- discard the inverse
        # matrix; everything downstream stays in this same frame.
    if apply_clahe_preproc:
        gray = apply_clahe(gray, clip_limit=clahe_clip_limit)
    image = Image.fromarray(gray)

    detector = YoloDotDetector(weights=yolo_dot_weights, conf=dot_conf, device=str(device_t))
    points = detector.detect(image)

    # This branch's cluster_into_cells has no auto-estimate mode (unlike
    # the improve-cell-detection branch) -- its own 15.0px default is used
    # unless overridden.
    cluster_kwargs = {} if link_distance is None else {"link_distance": link_distance}
    clusters = cluster_into_cells(points, **cluster_kwargs)
    valid = [c for c in clusters if not c["merged"]]

    cell_w, cell_h = _estimate_cell_size(clusters)
    w, h = image.size
    boxes = [
        _cluster_crop_box(c["center"], cell_w, cell_h, margin_scale, w, h)
        for c in valid
    ]
    crops = [
        image.crop(tuple(int(round(v)) for v in box)).resize((img_size, img_size), Image.Resampling.BICUBIC)
        for box in boxes
    ]

    model = load_model(str(braille_checkpoint), device_t)
    preds, confidences = _classify(crops, model, device_t)

    cells = [
        Cell(
            center=tuple(c["center"]),
            code=int(preds[i].item()),
            label=code_to_label(int(preds[i].item())),
            confidence=float(confidences[i].item()),
        )
        for i, c in enumerate(valid)
    ]

    return PageScan(reference_gray=gray.astype(np.uint8), cells=cells)
