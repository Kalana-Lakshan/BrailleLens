"""CC-* camera helpers without opening a device (SRS supporting capture)."""

from __future__ import annotations

import numpy as np

from camera_capture.camera import (
    _fit_for_display,
    _motion_score,
    _overlay_status_lines,
    _scale_box,
)


def test_cc01_first_frame_motion_is_zero():
    curr = np.zeros((16, 16), dtype=np.uint8)
    assert _motion_score(None, curr) == 0.0


def test_cc02_changed_frame_has_positive_motion():
    a = np.zeros((16, 16), dtype=np.uint8)
    b = np.full((16, 16), 255, dtype=np.uint8)
    assert _motion_score(a, b) > 0.0


def test_cc03_fit_for_display_no_upscale():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    out, scale = _fit_for_display(frame, max_width=400)
    assert scale == 1.0
    assert out.shape[1] == 200


def test_cc04_fit_for_display_downscales():
    frame = np.zeros((100, 400, 3), dtype=np.uint8)
    out, scale = _fit_for_display(frame, max_width=200)
    assert abs(scale - 0.5) < 1e-9
    assert out.shape[1] == 200


def test_cc05_scale_box_and_empty_status():
    assert _scale_box((10, 20, 30, 40), 0.5) == (5, 10, 15, 20)
    lines = _overlay_status_lines(None, 1.0)
    assert "waiting" in lines[0].lower() or "Hold camera" in lines[1]
