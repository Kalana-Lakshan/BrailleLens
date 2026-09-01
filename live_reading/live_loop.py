"""Ties pre_scan.py + registration.py + yolo_finger_detect together into
the live reading loop: for each camera frame, find the fingertip, map it
into the pre-scan's reference frame, and look up which cell it's over.

    pre-scan (once)  ->  PageScan (cell lookup table)
    each live frame  ->  fingertip (x,y) --registration--> reference (x,y)
                                                                  |
                                                                  v
                                                    PageScan.nearest_cell()
                                                                  |
                                                                  v
                                                          Cell (character)

Debounced so the same cell isn't re-emitted every single frame while the
finger sits still on it -- only emits when the matched cell actually
changes, matching how a screen-reader-style output should behave.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from yolo_finger_detect.detect_fingertip import FingertipDetector

from .pre_scan import Cell, PageScan
from .registration import FrameRegistration


@dataclass
class ReadEvent:
    cell: Cell | None          # None if the finger isn't over any known cell
    fingertip_live: tuple[float, float] | None
    fingertip_reference: tuple[float, float] | None


class LiveReader:
    def __init__(
        self,
        page_scan: PageScan,
        fingertip_weights: str | Path | None = None,
        fingertip_conf: float = 0.25,
        device: str = "cpu",
        max_lookup_distance: float | None = None,
    ):
        self.page_scan = page_scan
        self.fingertip_detector = FingertipDetector(weights=fingertip_weights, conf=fingertip_conf, device=device)
        self.registration = FrameRegistration(page_scan.reference_gray)
        # Default: half the median inter-cell spacing, so the finger has to
        # be genuinely close to a cell's own center to match it, not just
        # "somewhere on the page" -- avoids confidently reporting a wrong
        # neighboring cell when the finger is between two of them.
        self.max_lookup_distance = max_lookup_distance or self._default_lookup_radius()
        self._last_cell: Cell | None = None

    def _default_lookup_radius(self) -> float:
        if len(self.page_scan.cells) < 2:
            return 50.0
        centers = np.array([c.center for c in self.page_scan.cells])
        from scipy.spatial import cKDTree

        tree = cKDTree(centers)
        nn_dist, _ = tree.query(centers, k=2)
        return float(np.median(nn_dist[:, 1])) * 0.6

    def process_frame(self, live_bgr: np.ndarray) -> ReadEvent:
        fingertip = self.fingertip_detector.detect_best(live_bgr)
        if fingertip is None:
            return ReadEvent(cell=None, fingertip_live=None, fingertip_reference=None)

        live_gray = cv2.cvtColor(live_bgr, cv2.COLOR_BGR2GRAY) if live_bgr.ndim == 3 else live_bgr
        ref_point = self.registration.map_point(live_gray, fingertip)
        if ref_point is None:
            # Couldn't register this frame (or any prior one) against the
            # reference -- can't know which cell the fingertip is over.
            return ReadEvent(cell=None, fingertip_live=fingertip, fingertip_reference=None)

        cell = self.page_scan.nearest_cell(ref_point, max_distance=self.max_lookup_distance)
        return ReadEvent(cell=cell, fingertip_live=fingertip, fingertip_reference=ref_point)

    def process_frame_debounced(self, live_bgr: np.ndarray) -> Cell | None:
        """Like process_frame, but only returns a Cell when it's DIFFERENT
        from the last one returned (None the rest of the time) -- use this
        for driving speech/braille-cell output so the same character isn't
        re-announced every frame while the finger holds still."""
        event = self.process_frame(live_bgr)
        if event.cell is None:
            return None
        if self._last_cell is not None and event.cell.center == self._last_cell.center:
            return None
        self._last_cell = event.cell
        return event.cell


def run_on_video(reader: LiveReader, source, quiet: bool = False):
    """source: camera index (int), video file path, or IP-camera URL --
    anything cv2.VideoCapture accepts."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {source}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cell = reader.process_frame_debounced(frame)
            if cell is not None and not quiet:
                print(f"-> {cell.label}  (code={cell.code}, conf={cell.confidence:.2f})")
    finally:
        cap.release()


def main():
    import argparse

    from .pre_scan import scan_page

    p = argparse.ArgumentParser(description="Live Braille reading with finger tracking")
    p.add_argument("--page-image", type=Path, required=True, help="Clear, unoccluded pre-scan photo")
    p.add_argument("--braille-checkpoint", type=Path, required=True)
    p.add_argument("--yolo-dot-weights", type=Path, default=None)
    p.add_argument("--fingertip-weights", type=Path, default=None)
    p.add_argument("--source", default=0, help="Camera index, video file, or IP camera URL")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--clahe", action=argparse.BooleanOptionalAction, default=True,
                    help="CLAHE contrast correction on the pre-scan photo (see cell_detect/preprocess.py). On by default -- CAUTION: measured to HURT cell detection on one real phone photo; pass --no-clahe if it hurts on yours")
    p.add_argument("--clahe-clip-limit", type=float, default=2.5)
    p.add_argument("--deskew", action=argparse.BooleanOptionalAction, default=True,
                    help="Best-effort perspective deskew on the pre-scan photo. On by default; safe no-op when no confident page quad is found")
    args = p.parse_args()

    print(f"Pre-scanning {args.page_image} ...")
    scan = scan_page(
        args.page_image,
        braille_checkpoint=args.braille_checkpoint,
        yolo_dot_weights=args.yolo_dot_weights,
        device=args.device,
        apply_clahe_preproc=args.clahe,
        clahe_clip_limit=args.clahe_clip_limit,
        deskew=args.deskew,
    )
    print(f"  {len(scan.cells)} cells found.")

    reader = LiveReader(scan, fingertip_weights=args.fingertip_weights, device=args.device)

    source = args.source
    try:
        source = int(source)
    except (TypeError, ValueError):
        pass
    run_on_video(reader, source)


if __name__ == "__main__":
    main()
