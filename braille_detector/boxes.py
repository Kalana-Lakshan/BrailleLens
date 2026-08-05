"""Box/label utilities shared by the detector's data, model, loss, and
inference code. Self-contained (no imports from braille_cnn/) by design --
this whole folder is a separate, from-scratch implementation of the Ovodov
"Optical Braille Recognition Using Object Detection CNN" approach (single-
stage detector finds+classifies whole cells directly, no dot-detection/
grid-fitting stage at all), kept isolated from the existing grid-fitting
pipeline in braille_cnn/ rather than modifying it.

Code convention (matches the rest of the project, confirmed bit-identical
across DBSI/Angelina/braille_cnn's own dots_to_code): bit (i-1) of a 0-63
code is 1 if dot i is raised. Dot layout in a 2x3 cell:
    1 4
    2 5
    3 6
i.e. bits 0,1,2 = left column (dots 1,2,3), bits 3,4,5 = right column
(dots 4,5,6). Code 0 (no dots) is never a detection target -- a blank cell
has nothing for the detector to find.
"""

import numpy as np

NUM_CLASSES = 63  # codes 1..63; class index = code - 1


def mirror_code(code):
    """Code for the same character after a horizontal flip of the image --
    swaps the left and right dot columns (dot1<->dot4, dot2<->dot5,
    dot3<->dot6), i.e. swap the low 3 bits with the high 3 bits."""
    left = code & 0b000111
    right = (code & 0b111000) >> 3
    return (left << 3) | right


def box_iou(boxes1, boxes2):
    """boxes1: (N,4), boxes2: (M,4), each (x0,y0,x1,y1). Returns (N,M) IOU."""
    boxes1 = np.asarray(boxes1, dtype=np.float64)
    boxes2 = np.asarray(boxes2, dtype=np.float64)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    x0 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y0 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x1 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y1 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    union = area1[:, None] + area2[None, :] - inter
    return inter / np.clip(union, 1e-9, None)


def nms(boxes, scores, iou_threshold=0.02):
    """Greedy NMS. boxes: (N,4), scores: (N,). Returns kept indices.

    iou_threshold is deliberately very low (0.02, matching the paper) --
    real Braille characters never overlap, so any detected overlap is
    assumed to be a duplicate detection of the same character, not two
    genuinely adjacent objects, and can be suppressed aggressively.
    """
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int64)
    order = np.argsort(-scores)
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        ious = box_iou(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_threshold]
    return np.array(keep, dtype=np.int64)
