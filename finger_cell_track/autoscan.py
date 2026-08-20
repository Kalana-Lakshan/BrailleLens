"""Stage 5.1 — hands-free page capture.

PageWatcher watches the live stream and fires a prescan when the camera is
still, no hand is covering the page, and a cooldown has elapsed. R in
live_app.py stays as a manual override.

    watcher = PageWatcher(scan_fn=lambda frame: recognize_or_prescan(frame))
    event = watcher.update(frame_bgr, hand_visible=bool(tip_or_hand), now=time.time())
    if event:
        print(event.message)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

SEARCHING = "SEARCHING"
CAPTURING = "CAPTURING"
TRACKING = "TRACKING"


def motion_score(prev: Optional[np.ndarray], curr: np.ndarray) -> float:
    """Mean absolute pixel difference. Same idea as camera_capture.camera."""
    if prev is None:
        return 0.0
    a = prev if prev.ndim == 2 else cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    b = curr if curr.ndim == 2 else cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    return float(cv2.absdiff(a, b).mean())


@dataclass
class PageEvent:
    kind: str  # searching | hand | capturing | captured | rejected | ready | lost
    message: str
    cell_count: int = 0
    mean_conf: float = 0.0
    elapsed_s: float = 0.0


class PageWatcher:
    """Four-state machine: SEARCHING → CAPTURING → TRACKING → SEARCHING."""

    def __init__(
        self,
        scan_fn: Callable,
        motion_threshold: float = 8.0,
        stable_frames: int = 10,
        rescan_cooldown: float = 3.0,
        min_cells: int = 20,
        min_page_conf: float = 0.5,
        lost_seconds: float = 2.0,
        hand_clear_frames: int = 8,
    ) -> None:
        self.scan_fn = scan_fn
        self.motion_threshold = motion_threshold
        self.stable_frames = stable_frames
        self.rescan_cooldown = rescan_cooldown
        self.min_cells = min_cells
        self.min_page_conf = min_page_conf
        self.lost_seconds = lost_seconds
        self.hand_clear_frames = hand_clear_frames

        self.state = SEARCHING
        self._prev_gray: Optional[np.ndarray] = None
        self._still_streak = 0
        self._no_hand_streak = 0
        self._last_capture_t = 0.0
        self._lost_since: Optional[float] = None
        self._tracking_since: Optional[float] = None
        self._last_kind: Optional[str] = None
        self.cell_map = None

    def reset_to_search(self) -> None:
        self.state = SEARCHING
        self._still_streak = 0
        self._no_hand_streak = 0
        self._lost_since = None
        self._tracking_since = None
        self.cell_map = None

    def _emit(self, kind: str, message: str, **kwargs) -> Optional[PageEvent]:
        if kind == self._last_kind:
            return None
        self._last_kind = kind
        return PageEvent(kind=kind, message=message, **kwargs)

    def update(
        self,
        frame_bgr: np.ndarray,
        hand_visible: bool = False,
        registration_lost: bool = False,
        now: Optional[float] = None,
    ) -> Optional[PageEvent]:
        now = time.time() if now is None else now
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        motion = motion_score(self._prev_gray, gray)
        self._prev_gray = gray

        if hand_visible:
            self._no_hand_streak = 0
        else:
            self._no_hand_streak += 1

        if motion < self.motion_threshold:
            self._still_streak += 1
        else:
            self._still_streak = 0

        if self.state == TRACKING:
            if self._tracking_since is None:
                self._tracking_since = now
            if registration_lost:
                if self._lost_since is None:
                    self._lost_since = now
                elif now - self._lost_since >= self.lost_seconds:
                    self.reset_to_search()
                    return self._emit("lost", "[PAGE] lost - searching for the page again")
            else:
                self._lost_since = None
            # Don't dump the CellMap the moment the finger enters (motion).
            grace = (now - self._tracking_since) < 2.0
            if not grace and motion > 6.0 * self.motion_threshold:
                self.reset_to_search()
                return self._emit("lost", "[PAGE] lost - searching for the page again")
            return None

        if hand_visible:
            return self._emit("hand", "[PAGE] hand in view - lift your hand so the page can be scanned")

        searching = self._emit("searching", "[PAGE] searching - hold the camera steady over the page")

        ready = (
            self._still_streak >= self.stable_frames
            and self._no_hand_streak >= self.hand_clear_frames
            and (now - self._last_capture_t) >= self.rescan_cooldown
        )
        if not ready:
            return searching

        self.state = CAPTURING
        started = time.time()
        # Always notify immediately — live_app only receives one event per update,
        # and the final CAPTURED event would otherwise hide this line.
        print("[PAGE] capturing...", flush=True)
        self._emit("capturing", "[PAGE] capturing...")
        result = self.scan_fn(frame_bgr)
        elapsed = time.time() - started
        self._last_capture_t = now

        cells = getattr(result, "cells", result)
        n = len(cells) if cells is not None else 0
        confs = [getattr(c, "conf", 1.0) for c in (cells or [])]
        mean_conf = float(np.mean(confs)) if confs else 0.0

        if n < self.min_cells or mean_conf < self.min_page_conf:
            self.state = SEARCHING
            self._still_streak = 0
            return self._emit(
                "rejected",
                f"[PAGE] rejected - {n} cells, mean conf {mean_conf:.2f} "
                f"(need ≥{self.min_cells} cells and conf ≥{self.min_page_conf})",
                cell_count=n,
                mean_conf=mean_conf,
                elapsed_s=elapsed,
            )

        self.cell_map = result
        self.state = TRACKING
        self._lost_since = None
        self._tracking_since = now
        # Force a fresh "captured" notify even if the last kind was also captured.
        self._last_kind = None
        return self._emit(
            "captured",
            f"[PAGE] CAPTURED - {n} cells, mean conf {mean_conf:.2f} ({elapsed:.1f}s)",
            cell_count=n,
            mean_conf=mean_conf,
            elapsed_s=elapsed,
        )
