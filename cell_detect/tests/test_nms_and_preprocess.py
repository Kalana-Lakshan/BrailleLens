"""CD-* cell-detector geometry (SRS FR 5 / FR 9). No trained weights required."""

from __future__ import annotations

import numpy as np

from cell_detect.detect_cells import _iou, _merge_detections
from cell_detect.preprocess import apply_clahe, deskew_page, remap_boxes, _order_corners


def test_cd01_iou_identical_and_disjoint():
    box = (0.0, 0.0, 10.0, 10.0)
    assert _iou(box, box) == 1.0
    assert _iou(box, (20.0, 20.0, 30.0, 30.0)) == 0.0


def test_cd02_iou_partial_overlap():
    a = (0.0, 0.0, 10.0, 10.0)
    b = (5.0, 0.0, 15.0, 10.0)
    # intersection 50, union 150
    assert abs(_iou(a, b) - 50.0 / 150.0) < 1e-9


def test_cd03_merge_detections_keeps_higher_conf():
    dets = [
        {"xyxy": (0.0, 0.0, 10.0, 10.0), "conf": 0.4},
        {"xyxy": (1.0, 1.0, 11.0, 11.0), "conf": 0.9},
        {"xyxy": (50.0, 50.0, 60.0, 60.0), "conf": 0.8},
    ]
    kept = _merge_detections(dets, iou_thresh=0.5)
    confs = sorted(d["conf"] for d in kept)
    assert confs == [0.8, 0.9]
    assert len(kept) == 2


def test_cd04_empty_image_has_no_page_quad():
    blank = np.full((120, 160, 3), 128, dtype=np.uint8)
    warped, inv = deskew_page(blank, min_area_frac=0.25)
    assert inv is None
    assert warped is blank


def test_cd05_clahe_preserves_shape():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[8:24, 8:24] = 200
    out = apply_clahe(img, clip_limit=2.0)
    assert out.shape == img.shape


def test_cd06_order_corners_tl_tr_br_bl():
    pts = np.array([[10, 0], [0, 0], [10, 10], [0, 10]], dtype=np.float32)
    ordered = _order_corners(pts)
    assert list(ordered[0]) == [0.0, 0.0]
    assert list(ordered[1]) == [10.0, 0.0]
    assert list(ordered[2]) == [10.0, 10.0]
    assert list(ordered[3]) == [0.0, 10.0]


def test_cd07_remap_boxes_identity():
    ident = np.eye(3, dtype=np.float32)
    boxes = [(2.0, 4.0, 8.0, 12.0)]
    out = remap_boxes(boxes, ident)
    assert len(out) == 1
    x0, y0, x1, y1 = out[0]
    assert abs(x0 - 2.0) < 1e-4 and abs(y1 - 12.0) < 1e-4
