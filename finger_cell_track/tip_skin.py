"""Fingertip method 1/3 — SkinContourTip (classical CV; live-app default).

YCrCb skin mask → contours → distal pad contact. No neural net.
Wire via tip_backends.create_tip_backend("skin") or TipEMA in live_app.
"""

from __future__ import annotations

import cv2
import numpy as np


class SkinContourTip:
    """Contact point from the largest skin blob that enters the frame.

    STUDY WALKTHROUGH (default tip method — classical CV, not a neural net):
      Step A  _mask()         → binary skin pixels (white=skin, black=other)
      Step B  findContours    → candidate finger-shaped blobs
      Step C  filter + score  → keep border-entering blobs; pick best
      Step D  distal pad      → contact (x,y) on the pad, not nail/knuckle
      Step E  reject ghosts   → drop corner / shallow false tips

    Used as the live-app default. MediaPipe fails on top-down Braille footage
    (palm cropped); this path does not need the palm.
    """

    def __init__(
        self,
        min_area: int = 1200,  # ignore tiny speckles (noise)
        max_area_frac: float = 0.22,  # ignore huge regions (whole page / arm)
        contact_offset: float = 0.22,  # pull tip back toward wrist → pad, not nail
        y_max: int = 200,  # max brightness Y; cream paper is brighter → excluded
        border_px: int = 12,  # how thick the "frame edge" band is
        min_thickness: int = 28,  # reject skinny strips (page borders)
        min_solidity: float = 0.35,  # area/hull; reject ragged noise blobs
        corner_reject_px: int = 48,  # tip too close to a corner = ghost
        corner_min_area: int = 5000,  # large blobs near corners are still OK
        min_reach_frac: float = 0.10,  # finger must reach this far into the page
        max_edge_aspect: float = 4.0,  # tall thin left/right strip = decoration
    ) -> None:
        self.min_area = min_area
        self.max_area_frac = max_area_frac
        self.contact_offset = contact_offset
        self.y_max = y_max
        self.border_px = border_px
        self.min_thickness = min_thickness
        self.min_solidity = min_solidity
        self.corner_reject_px = corner_reject_px
        self.corner_min_area = corner_min_area
        self.min_reach_frac = min_reach_frac
        self.max_edge_aspect = max_edge_aspect
        self.hand_visible = False  # True when detect() found a tip this frame

    def _mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        # --- Step A: build a binary skin mask ---
        # YCrCb separates brightness (Y) from chroma (Cr, Cb); skin lives in a
        # known Cr/Cb band. Capping Y stops cream Braille paper matching "skin".
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        # Pixels inside this range → 255 (skin); outside → 0.
        mask = cv2.inRange(ycrcb, (40, 133, 77), (self.y_max, 173, 127))
        # Ellipse kernel for morphology.
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        # OPEN = erode then dilate → remove salt noise (tiny white dots).
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        # CLOSE = dilate then erode → fill small holes inside the finger.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
        return mask  # HxW uint8: 255 = candidate skin

    def _reject_blob(
        self, contour, area: float, x: int, y: int, bw: int, bh: int, w: int, h: int
    ) -> bool:
        """True = discard this contour (not a real finger)."""
        # Too thin in either direction → edge strip / noise, not a finger.
        if min(bw, bh) < self.min_thickness:
            return True
        # Solidity = filled area / convex-hull area. Low = spiky junk.
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        if hull_area > 1.0 and (area / hull_area) < self.min_solidity:
            return True

        # Which frame edges does the bounding box touch?
        b = self.border_px
        on_left = x <= b
        on_right = (x + bw) >= (w - b)
        on_top = y <= b
        on_bottom = (y + bh) >= (h - b)
        edge_count = int(on_left) + int(on_right) + int(on_top) + int(on_bottom)

        # "Reach" = how far the blob sticks into the page from its entry edge.
        if on_left and not on_right:
            reach = bw
        elif on_right and not on_left:
            reach = bw
        elif on_top and not on_bottom:
            reach = bh
        elif on_bottom and not on_top:
            reach = bh
        else:
            reach = max(bw, bh)

        # Shallow small blobs on the margin = ghosts, not a reading finger.
        min_reach = self.min_reach_frac * float(min(w, h))
        if reach < min_reach and area < self.corner_min_area:
            return True

        # Tall thin column glued to left/right = decorative Braille page border.
        if edge_count == 1 and (on_left or on_right):
            aspect = bh / max(bw, 1)
            if aspect >= self.max_edge_aspect and bw < 0.22 * w:
                return True
            if bw < self.min_thickness * 2.5 and area < self.corner_min_area:
                return True
        # Same idea for a wide shallow strip on top/bottom.
        if edge_count == 1 and (on_top or on_bottom):
            aspect = bw / max(bh, 1)
            if aspect >= self.max_edge_aspect and bh < 0.22 * h:
                return True
        return False  # keep this blob

    def detect(self, frame_bgr: np.ndarray): 
        """Return (tip_xy, bbox_xyxy, conf) or (None, None, 0.0). Same API as TipYOLO."""
        h, w = frame_bgr.shape[:2]

        # --- Step B: skin mask → outer contours (each = one blob) ---
        mask = self._mask(frame_bgr)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # --- Step C: keep only finger-like border-entering blobs; score them ---
        max_area = self.max_area_frac * w * h
        best = None
        best_score = -1.0
        best_area = 0.0
        for contour in contours:
            area = float(cv2.contourArea(contour))
            # Skip too-small (noise) or too-large (page/arm) blobs.
            if area < self.min_area or area > max_area:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            b = self.border_px
            # Finger must enter from a frame edge (we never see a floating palm).
            touches = x <= b or y <= b or (x + bw) >= (w - b) or (y + bh) >= (h - b)
            if not touches:
                continue
            if self._reject_blob(contour, area, x, y, bw, bh, w, h):
                continue
            # Prefer larger blobs whose center sits deeper into the page.
            cx = x + 0.5 * bw
            cy = y + 0.5 * bh
            depth = min(cx, w - 1 - cx, cy, h - 1 - cy)
            score = area * (1.0 + depth / max(min(w, h), 1))
            if score > best_score:
                best_score = score
                best = contour
                best_area = area

        if best is None:
            self.hand_visible = False
            return None, None, 0.0  # no finger this frame

        self.hand_visible = True
        # Contour vertices as Nx2 float points.
        pts = best.reshape(-1, 2).astype(np.float32)
        if pts.shape[0] < 5:
            self.hand_visible = False
            return None, None, 0.0

        # --- Step D: estimate "wrist" (entry) then distal pad contact ---
        b = self.border_px
        # Points that lie on the frame border ≈ where the finger enters.
        on_border = (
            (pts[:, 0] <= b)
            | (pts[:, 1] <= b)
            | (pts[:, 0] >= (w - 1 - b))
            | (pts[:, 1] >= (h - 1 - b))
        )
        if on_border.any():
            wrist = pts[on_border].mean(axis=0)  # average entry point
        else:
            # Fallback: point closest to any frame edge.
            edge_dist = np.minimum.reduce(
                [pts[:, 0], w - 1 - pts[:, 0], pts[:, 1], h - 1 - pts[:, 1]]
            )
            wrist = pts[int(np.argmin(edge_dist))]

        # Only look at the far 20% of the contour (away from wrist) = fingertip region.
        # This stops the contact locking onto a knuckle mid-finger.
        from_wrist = np.linalg.norm(pts - wrist.reshape(1, 2), axis=1)
        far_cut = float(np.percentile(from_wrist, 80.0))
        far_mask = from_wrist >= far_cut
        if not np.any(far_mask):
            far_mask = from_wrist >= from_wrist.max() * 0.85
        far_pts = pts[far_mask]
        # Among far points, pick the one deepest into the page (max min-distance to edges).
        inward_far = np.minimum.reduce(
            [
                far_pts[:, 0],
                w - 1 - far_pts[:, 0],
                far_pts[:, 1],
                h - 1 - far_pts[:, 1],
            ]
        )
        tip = far_pts[int(np.argmax(inward_far))]  # geometric "fingertip"
        # Move slightly toward wrist so contact sits on the pad, not the nail tip.
        contact = tip - self.contact_offset * (tip - wrist)
        xy = (int(round(float(contact[0]))), int(round(float(contact[1]))))

        # --- Step E: final ghost checks on the contact point itself ---
        # Contact must sit far enough into the page (not stuck on the margin).
        reach_from_border = float(
            np.min([xy[0], w - 1 - xy[0], xy[1], h - 1 - xy[1]])
        )
        if (
            reach_from_border < self.min_reach_frac * 0.5 * min(w, h)
            and best_area < self.corner_min_area
        ):
            self.hand_visible = False
            return None, None, 0.0

        # Tip near two edges at once = corner ghost unless blob is large.
        c = self.corner_reject_px
        near_corner = (xy[0] <= c or xy[0] >= w - 1 - c) and (
            xy[1] <= c or xy[1] >= h - 1 - c
        )
        if near_corner and best_area < self.corner_min_area:
            self.hand_visible = False
            return None, None, 0.0

        # Bounding box of the whole finger blob (for debug overlay).
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        # conf=1.0: classical method has no probability score.
        return xy, (int(x0), int(y0), int(x1), int(y1)), 1.0

    def close(self) -> None:
        return None  # no neural-net / MediaPipe resources to free
