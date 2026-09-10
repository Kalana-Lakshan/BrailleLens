"""DOT-* fallback tiling / seam dedupe (SRS FR 5 fallback). No weights required."""

from __future__ import annotations

import pytest

from yolo_dot_detect.detect_dots import _tile_origins


def test_dot01_tile_origins_single_when_smaller_than_tile():
    """Check an image smaller than the tile size has a single origin at 0."""
    assert _tile_origins(400, tile=640, stride=320) == [0]


def test_dot02_tile_origins_covers_last_edge():
    """Check tile starts include 0 and a last origin on the image edge so the last strip is not skipped."""
    origins = _tile_origins(1000, tile=640, stride=320)
    assert origins[0] == 0
    assert origins[-1] == 1000 - 640
    assert all(o >= 0 for o in origins)


def test_dot03_dedupe_keeps_highest_conf_near_seam():
    """Check two dots 1 px apart keep only the 0.9 detection; a far 0.7 dot is kept as a second point."""
    pytest.importorskip("scipy")
    from yolo_dot_detect.detect_dots import _dedupe

    dets = [
        {"center": (10.0, 10.0), "conf": 0.4},
        {"center": (11.0, 10.0), "conf": 0.9},
        {"center": (80.0, 80.0), "conf": 0.7},
    ]
    kept = _dedupe(dets, min_distance=5.0)
    assert len(kept) == 2
    confs = {round(d["conf"], 2) for d in kept}
    assert confs == {0.9, 0.7}
