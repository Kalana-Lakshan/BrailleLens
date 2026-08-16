"""Stage 4a - the cell detector: "where are the Braille cells on this page?"

Single-class YOLO over whole Braille cells. This is the first half of the
two-stage recogniser; braille_cnn/ is the second half and answers "what is each
cell?".

Detecting cells rather than dots removes the grid-fitting step that was the
main source of error on handheld photos, and it produces the cell boxes the
live app needs for the finger hit-test anyway.

Not to be confused with yolo_dot_detect/, which detects individual raised
*dots* and remains the fallback path and a baseline for comparison.
"""

from .detect_cells import CellDetector, detect_cells  # noqa: F401
