"""Maps a point in a live camera frame back into the pre-scan's reference
frame, so a fingertip position can be looked up against the pre-scan's
cell table even if the page or camera has shifted since the scan was
taken (tilt, translation, the user's head moving while reading).

Uses ORB feature matching + a RANSAC homography -- a global, whole-frame
alignment, not local patch tracking around the fingertip itself. That
matters here specifically: the fingertip is, by definition, near a region
that may be partially occluded by the hand, so anything relying on local
texture right around the point being tracked is exactly the region most
likely to be unreliable. Matching on keypoints spread across the whole
page (most of which the hand isn't covering at any given moment) is far
more robust to that.
"""

from __future__ import annotations

import cv2
import numpy as np


class FrameRegistration:
    def __init__(
        self,
        reference_gray: np.ndarray,
        n_features: int = 2000,
        min_matches: int = 12,
        ransac_reproj_threshold: float = 5.0,
        match_ratio: float = 0.75,
    ):
        self.min_matches = min_matches
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.match_ratio = match_ratio

        self._orb = cv2.ORB_create(nfeatures=n_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.ref_keypoints, self.ref_descriptors = self._orb.detectAndCompute(reference_gray, None)

        self._last_homography: np.ndarray | None = None
        self._last_inliers: int = 0

    def estimate_homography(self, live_gray: np.ndarray) -> np.ndarray | None:
        """Returns the 3x3 homography mapping live-frame points -> reference-
        frame points, or None if not enough confident matches were found
        (caller should fall back to the last known-good homography, e.g.
        via `homography_or_last`, rather than treat a single bad frame as
        "lost" -- momentary motion blur/glare on one frame is common)."""
        if self.ref_descriptors is None:
            return None
        live_kp, live_desc = self._orb.detectAndCompute(live_gray, None)
        if live_desc is None or len(live_kp) < self.min_matches:
            return None

        raw_matches = self._matcher.knnMatch(live_desc, self.ref_descriptors, k=2)
        good = [m for m, n in raw_matches if len(raw_matches) and m.distance < self.match_ratio * n.distance]
        if len(good) < self.min_matches:
            return None

        src_pts = np.float32([live_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([self.ref_keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_reproj_threshold)
        if H is None:
            return None

        inliers = int(mask.sum()) if mask is not None else 0
        if inliers < self.min_matches:
            return None

        self._last_homography = H
        self._last_inliers = inliers
        return H

    def homography_or_last(self, live_gray: np.ndarray) -> np.ndarray | None:
        """Like estimate_homography, but falls back to the last known-good
        homography if this frame's matching fails -- avoids dropping
        tracking on a single noisy frame. Still returns None if no
        homography has ever been found."""
        H = self.estimate_homography(live_gray)
        return H if H is not None else self._last_homography

    @staticmethod
    def transform_point(point: tuple[float, float], homography: np.ndarray) -> tuple[float, float]:
        pts = np.array([[point]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pts, homography)
        return float(mapped[0, 0, 0]), float(mapped[0, 0, 1])

    def map_point(self, live_gray: np.ndarray, point: tuple[float, float]) -> tuple[float, float] | None:
        """Convenience one-shot: live-frame point -> reference-frame point,
        or None if no homography (current or last known-good) is available."""
        H = self.homography_or_last(live_gray)
        if H is None:
            return None
        return self.transform_point(point, H)
