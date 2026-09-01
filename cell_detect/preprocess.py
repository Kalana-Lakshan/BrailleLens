"""Stage 4a step 0 - image preprocessing before the cell detector.

The cell detector's own training augmentation (cell_detect/configs/cells.yaml)
bakes in tolerance for *some* amount of real-world capture variation --
hsv_v=0.40 exposure jitter ("scanner vs phone vs glasses camera"),
degrees=3.0 page tilt, perspective=0.0005 -- but that tolerance only covers
the range the augmentation swept, and nothing at inference time corrects an
image that falls outside it. A phone photo taken in uneven room lighting, or
at a real handheld angle, can exceed that range.

Two corrective steps are available, both wired into CellDetector.detect_boxes
but **off by default**:

    apply_clahe   Local contrast normalization. Purely photometric (doesn't
                  touch image geometry, so box coordinates need no
                  remapping) -- but measured on a real handheld phone photo
                  of an open Braille book (test-img3.jpeg, raking side-light
                  casting shadows off the raised dots), it made detection
                  markedly WORSE, monotonically with clip_limit: 183 boxes
                  at baseline down to 65/34/16/7/5 at clip_limit
                  1.0/1.5/2.0/2.5/3.0. Best guess why: the detector's HSV
                  training jitter (cells.yaml's hsv_v=0.40) varies overall
                  exposure, which is a global, roughly linear shift CLAHE
                  does NOT reproduce -- CLAHE instead locally re-stretches
                  contrast per tile, which can flatten or invert exactly the
                  raised-dot / shadow contrast pattern the model's conv
                  filters learned to key on, especially under directional
                  (non-uniform) lighting. Left in as an opt-in tool and NOT
                  removed, because a milder setting or a different capture
                  condition (flat, evenly-lit scan) may still benefit -- but
                  do not flip this default on without measuring it first,
                  the way this one measurement already contradicted the
                  original hypothesis.

    deskew_page   Best-effort perspective correction: finds the page's own
                  quadrilateral outline and warps it flat before detection.
                  Detected boxes are then mapped back through the inverse
                  transform (remap_boxes) so callers still get boxes in the
                  *original* image's coordinate frame -- recognize_page()
                  crops classifier input from the original image, not the
                  warped one, so this remapping is required for correctness,
                  not an optional nicety. Untested end-to-end (on
                  test-img3.jpeg it correctly found no confident quad --
                  it's an open two-page spread against a cluttered
                  background -- and safely no-op'd rather than guessing).
                  Depends on the page forming one clear, confidently-
                  detectable quadrilateral against a distinguishable
                  background, which a real open-book photo often will not
                  give it. Same "off until validated on real photos" posture
                  as spine_boost in detect_cells.py -- flip the default once
                  it's been checked against a real phone-photo set, ideally
                  the Gold Dataset the way spine_boost and drop_ruler_lines
                  were (see reports/eval/gold_cell_detector_finetune.md).
"""

from __future__ import annotations

import cv2
import numpy as np


def apply_clahe(
    bgr: np.ndarray, clip_limit: float = 2.5, tile_grid: tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Contrast-Limited Adaptive Histogram Equalization on luminance only.

    Operates in LAB space so only the L (lightness) channel is touched --
    pixel geometry is unchanged, so box coordinates measured on the output
    are already valid on the input, unlike deskew_page below. clip_limit
    bounds how much any one tile's histogram can be stretched, which keeps
    flat/blank regions (a lot of a Braille page) from having sensor noise
    amplified into fake contrast -- the same concern normalize_crop's
    std_floor addresses per-cell-crop, applied here per-page before
    detection instead. tile_grid trades locality (small tiles adapt to
    tight shadows/glare) against noise (very small tiles overfit to a few
    pixels); 8x8 is OpenCV's own default and a reasonable starting point for
    a full-page photo.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_chan = clahe.apply(l_chan)
    lab = cv2.merge((l_chan, a_chan, b_chan))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Sort 4 (x, y) points into [top-left, top-right, bottom-right, bottom-left],
    independent of the order cv2 happened to trace the contour in."""
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def find_page_quad(bgr: np.ndarray, min_area_frac: float = 0.25) -> np.ndarray | None:
    """Best-effort page-boundary quadrilateral, or None if not confidently found.

    Looks for the largest convex 4-point contour after edge detection.
    Requires it to cover at least min_area_frac of the frame so a random
    rectangular object in the background (a book spine highlight, a table
    edge, another book) can't be mistaken for the page -- returns None
    rather than guessing when nothing qualifies, so the caller can fall back
    to the original image untouched instead of warping it on a bad guess.
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_area = float(h * w)
    best = None
    best_area = 0.0
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        area = cv2.contourArea(approx)
        if area / frame_area < min_area_frac:
            continue
        if area > best_area:
            best_area = area
            best = approx.reshape(4, 2).astype(np.float32)
    return best


def deskew_page(
    bgr: np.ndarray, min_area_frac: float = 0.25
) -> tuple[np.ndarray, np.ndarray | None]:
    """Warp the page flat if a confident quadrilateral outline is found.

    Returns (processed_bgr, inverse_matrix). inverse_matrix is None when no
    quad was found -- processed_bgr is then just the input, unchanged --
    which callers use to know whether detected box coordinates need mapping
    back through remap_boxes(). When a quad IS found, processed_bgr is a new,
    differently-sized image (the straightened page), so anything measured on
    it (boxes) is only valid in ITS coordinate frame until remapped.
    """
    quad = find_page_quad(bgr, min_area_frac=min_area_frac)
    if quad is None:
        return bgr, None

    top_left, top_right, bottom_right, bottom_left = _order_corners(quad)
    width_top = np.linalg.norm(top_right - top_left)
    width_bottom = np.linalg.norm(bottom_right - bottom_left)
    height_left = np.linalg.norm(bottom_left - top_left)
    height_right = np.linalg.norm(bottom_right - top_right)
    out_w = int(max(width_top, width_bottom))
    out_h = int(max(height_left, height_right))
    if out_w < 2 or out_h < 2:
        return bgr, None

    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(bgr, matrix, (out_w, out_h))
    inverse = np.linalg.inv(matrix)
    return warped, inverse


def remap_boxes(
    boxes: list[tuple[float, float, float, float]], inverse_matrix: np.ndarray
) -> list[tuple[float, float, float, float]]:
    """Map xyxy boxes detected on a deskewed image back to the original
    image's coordinate frame.

    A box that was axis-aligned in the warped frame is generally NOT
    axis-aligned anymore once mapped back through the inverse perspective
    transform, so this returns the tightest axis-aligned box (in original
    coordinates) that contains all 4 mapped corners -- consistent with every
    other box in this codebase being represented as plain xyxy.
    """
    if not boxes:
        return []
    remapped = []
    for x0, y0, x1, y1 in boxes:
        corners = np.array(
            [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32
        ).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(corners, inverse_matrix).reshape(-1, 2)
        remapped.append(
            (
                float(mapped[:, 0].min()),
                float(mapped[:, 1].min()),
                float(mapped[:, 0].max()),
                float(mapped[:, 1].max()),
            )
        )
    return remapped
