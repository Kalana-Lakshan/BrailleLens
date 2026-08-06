"""Reading-order layout for DotNeuralNet cell detections.

Improves on the original ``parse_xywh_and_class`` (fixed y-threshold) by:

1. **Line breaks** — Otsu on y-center gaps (same idea as ``braille_cnn.infer_page``)
2. **Word spaces** — median inter-cell gap × factor within each line
3. **Left-to-right** sort within each line

Input cells are dicts::

    {
      "xyxy": (x0, y0, x1, y1),
      "center": (cx, cy),
      "conf": float,
      "cls_id": int,
      "pattern": str,   # optional 6-bit name
      "code": int,      # BrailleLens code
    }
"""

from __future__ import annotations

import cv2
import numpy as np


def _line_gap_threshold(sorted_ys: list[float], cell_h_med: float) -> float:
    diffs = np.diff(sorted_ys)
    if len(diffs) < 2 or float(diffs.max()) <= 0:
        return cell_h_med * 1.5
    scaled = np.clip(diffs / diffs.max() * 255, 0, 255).astype(np.uint8)
    otsu_val, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = (otsu_val / 255.0) * float(diffs.max())
    return max(threshold, cell_h_med * 1.3)


def group_into_lines(cells: list[dict]) -> list[list[dict]]:
    """Group cells into reading-order lines (top→bottom, left→right within)."""
    if not cells:
        return []

    heights = [c["xyxy"][3] - c["xyxy"][1] for c in cells]
    cell_h_med = float(np.median(heights)) if heights else 20.0

    by_y = sorted(cells, key=lambda c: c["center"][1])
    line_gap = _line_gap_threshold([c["center"][1] for c in by_y], cell_h_med)

    lines: list[list[dict]] = [[by_y[0]]]
    for c in by_y[1:]:
        if c["center"][1] - lines[-1][-1]["center"][1] > line_gap:
            lines.append([c])
        else:
            lines[-1].append(c)

    for line in lines:
        line.sort(key=lambda c: c["center"][0])
    lines.sort(key=lambda line: float(np.mean([c["center"][1] for c in line])))
    return lines


def word_gap_threshold(line: list[dict], factor: float = 1.8) -> tuple[list[float], float]:
    """Horizontal gaps between consecutive boxes and the word-break threshold."""
    gaps = []
    for i in range(len(line) - 1):
        # gap = next.left - curr.right
        gaps.append(line[i + 1]["xyxy"][0] - line[i]["xyxy"][2])
    thresh = float(np.median(gaps)) * factor if gaps else 0.0
    return gaps, thresh


def codes_with_word_spaces(
    line: list[dict],
    conf_threshold: float = 0.0,
) -> tuple[list[int], list[float]]:
    """Return codes/confs for one line, inserting code 0 (space) at word gaps.

    Low-confidence cells are still included (caller may blank them); word
    gaps always insert an empty-cell code so ``decode_sequence`` emits spaces.
    """
    if not line:
        return [], []

    gaps, word_gap = word_gap_threshold(line)
    codes: list[int] = []
    confs: list[float] = []

    for i, cell in enumerate(line):
        if i > 0 and gaps[i - 1] > max(word_gap, 1.0):
            codes.append(0)
            confs.append(1.0)
        codes.append(int(cell["code"]))
        confs.append(float(cell["conf"]))
    return codes, confs


def boxes_to_cells(boxes, names: dict | list) -> list[dict]:
    """Convert an ultralytics Boxes object into layout cell dicts."""
    from .pattern_code import class_id_to_code

    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)

    cells = []
    for (x0, y0, x1, y1), conf, cls_id in zip(xyxy, confs, clss):
        code = class_id_to_code(int(cls_id), names)
        if isinstance(names, dict):
            pattern = str(names.get(int(cls_id), names.get(str(cls_id), "")))
        else:
            pattern = str(names[int(cls_id)])
        cells.append(
            {
                "xyxy": (float(x0), float(y0), float(x1), float(y1)),
                "center": ((float(x0) + float(x1)) / 2.0, (float(y0) + float(y1)) / 2.0),
                "conf": float(conf),
                "cls_id": int(cls_id),
                "pattern": pattern,
                "code": code,
            }
        )
    return cells
