"""Estimates a photo's own dot pitch (in pixels) directly from its raised-
dot highlights, so an unfamiliar photo (no known dataset, no ground truth)
can be rescaled to the detector's expected ~16px pitch automatically,
instead of relying on a hardcoded per-dataset constant that only covers
DBSI/Angelina.

Deliberately much simpler than braille_cnn/dot_detect.py's full grid-fitting
(no periodicity/phase/line model at all -- just a nearest-neighbor spacing
estimate over raw brightness peaks). That's the point: this only needs a
rough overall SCALE, not precise per-cell positions, so it doesn't need to
get skew, line pitch, or column phase right at all -- exactly the kind of
per-line/per-page fitting that kept going wrong for real photos this
session. A biased or noisy pitch estimate just means a slightly-off scale
that the detector's own scale-jitter training robustness (see data.py) can
absorb; it doesn't corrupt anything downstream the way a bad grid fit did
in the old pipeline.
"""

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


def _raw_dot_peaks(gray, smooth_sigma=1.5, background_sigma=25, local_norm_sigma=20.0,
                    z_threshold=1.5, footprint=9, border_margin=20):
    """z_threshold=1.5 (not the more typical ~3.0) is deliberate: swept
    empirically across DBSI/Angelina/test-img3 -- at 3.0 too few peaks
    survive on a full DBSI page, and the resulting sparse point cloud's
    nearest-neighbor distance jumps from the true intra-cell pitch (~21px)
    to the much larger cell-to-cell pitch (~50px) since many points'
    real intra-cell neighbor goes undetected. 1.5-2.0 gave consistent,
    accurate pitch estimates on all three test images; this function has
    no downstream verification step (unlike braille_cnn/'s DotPatchCNN) so
    a noisier point set here is fine -- only the nearest-neighbor median is
    used, which is robust to a fair amount of noise.
    """
    img = np.asarray(gray, dtype=np.float32)
    smooth = cv2.GaussianBlur(img, (0, 0), sigmaX=smooth_sigma)
    background = cv2.GaussianBlur(img, (0, 0), sigmaX=background_sigma)
    diff = smooth - background
    local_mean = cv2.GaussianBlur(diff, (0, 0), sigmaX=local_norm_sigma)
    local_sqmean = cv2.GaussianBlur(diff * diff, (0, 0), sigmaX=local_norm_sigma)
    local_std = np.sqrt(np.clip(local_sqmean - local_mean ** 2, 1e-6, None))
    z = (diff - local_mean) / (local_std + 1.0)
    local_max = ndimage.maximum_filter(z, size=footprint)
    peaks = (z == local_max) & (z > z_threshold)
    if border_margin > 0:
        peaks[:border_margin, :] = False
        peaks[-border_margin:, :] = False
        peaks[:, :border_margin] = False
        peaks[:, -border_margin:] = False
    ys, xs = np.nonzero(peaks)
    return np.stack([xs, ys], axis=1).astype(np.float64)


def estimate_dot_pitch(gray, min_points=30, fallback=16.0):
    """Median 1st-nearest-neighbor distance over raw dot-highlight peaks --
    a reasonable proxy for the tighter intra-cell dot pitch (mirrors
    dot_detect.estimate_link_distance's same reasoning in braille_cnn/, but
    that function is not imported here -- this folder stays self-contained).
    Returns `fallback` if too few peaks are found to trust the estimate.
    """
    points = _raw_dot_peaks(gray)
    if len(points) < min_points:
        return fallback
    tree = cKDTree(points)
    nn_dist, _ = tree.query(points, k=2)
    pitch = float(np.median(nn_dist[:, 1]))
    return pitch if pitch > 0 else fallback


def estimate_scale(gray, target_dx=16.0, min_scale=0.3, max_scale=4.0):
    """Scale factor to rescale this image so its own dot pitch matches
    target_dx (the detector's expected pitch, see model.py's STRIDE).
    Clamped to a sane range in case the pitch estimate is degenerate (e.g.
    a blank or non-Braille image)."""
    pitch = estimate_dot_pitch(gray)
    scale = target_dx / pitch
    return float(np.clip(scale, min_scale, max_scale))
