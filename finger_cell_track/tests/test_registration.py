"""FT registration: live tip mapped into prescan frame (SRS FR 7)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_FCT = Path(__file__).resolve().parents[1]
if str(_FCT) not in sys.path:
    sys.path.insert(0, str(_FCT))

cv2 = pytest.importorskip("cv2")

from registration import FrameRegistration  # noqa: E402


def test_ft10_identity_homography_maps_point_unchanged():
    """Check with H = identity a point (20, 25) maps to itself (no camera motion after prescan)."""
    gray = np.zeros((64, 64), dtype=np.uint8)
    gray[10:40, 10:40] = 255
    reg = FrameRegistration(gray)
    reg.assume_identity()
    mapped = FrameRegistration.transform_point((20.0, 25.0), np.eye(3, dtype=np.float32))
    assert abs(mapped[0] - 20.0) < 1e-4
    assert abs(mapped[1] - 25.0) < 1e-4


def test_ft11_synthetic_drift_maps_within_two_pixels():
    """Check a warped synthetic page maps a known point back within 2 px of the reference location."""
    rng = np.random.default_rng(0)
    ref = (rng.random((480, 640)) * 255).astype(np.uint8)
    cv2.rectangle(ref, (100, 100), (300, 250), 255, -1)
    cv2.rectangle(ref, (350, 300), (500, 420), 0, -1)
    cv2.circle(ref, (450, 100), 40, 200, -1)

    M = cv2.getRotationMatrix2D((320, 240), 6, 1.05)
    M[0, 2] += 15
    M[1, 2] += -8
    live = cv2.warpAffine(ref, M, (640, 480), borderValue=128)

    reg = FrameRegistration(ref)
    H = reg.estimate_homography(live)
    assert H is not None
    true_ref = np.array([200.0, 150.0, 1.0])
    live_pt = M @ true_ref
    mapped = reg.transform_point((live_pt[0], live_pt[1]), H)
    err = float(np.hypot(mapped[0] - 200.0, mapped[1] - 150.0))
    assert err < 2.0
