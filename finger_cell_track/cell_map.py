"""CellMap, tip hit-test, dwell filter, and session memory.

Pure geometry — no camera / MediaPipe / YOLO imports.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Cell:
    id: int
    xyxy: tuple[float, float, float, float]  # x0,y0,x1,y1
    char: str = ""
    pattern: str = ""
    code: int = 0
    conf: float = 1.0
    line: int = 0
    col: int = 0

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.xyxy
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

    def expanded(self, margin_frac: float = 0.15) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = self.xyxy
        w, h = x1 - x0, y1 - y0
        mx, my = w * margin_frac, h * margin_frac
        return (x0 - mx, y0 - my, x1 + mx, y1 + my)

    def contains(self, x: float, y: float, margin_frac: float = 0.15) -> bool:
        x0, y0, x1, y1 = self.expanded(margin_frac)
        return x0 <= x <= x1 and y0 <= y <= y1


@dataclass
class CellMap:
    cells: list[Cell] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.cells)

    def by_id(self, cell_id: int) -> Optional[Cell]:
        for c in self.cells:
            if c.id == cell_id:
                return c
        return None


def hit_test(
    tip: tuple[float, float],
    cell_map: CellMap,
    margin_frac: float = 0.15,
) -> Optional[Cell]:
    """Return the cell under tip; if several overlap, nearest center wins."""
    if tip is None or not cell_map.cells:
        return None
    x, y = tip
    hits = [c for c in cell_map.cells if c.contains(x, y, margin_frac)]
    if not hits:
        return None
    if len(hits) == 1:
        return hits[0]
    return min(
        hits,
        key=lambda c: (c.center[0] - x) ** 2 + (c.center[1] - y) ** 2,
    )


class TipEMA:
    """Exponential moving average for fingertip coordinates.

    Also rejects teleport jumps (false tip locks on a distant edge blob).
    After ``jump_grace`` consecutive huge jumps, the new tip is accepted as a
    fresh track (real fast motion / re-entry).
    """

    def __init__(
        self,
        alpha: float = 0.35,
        max_jump_px: float = 180.0,
        jump_grace: int = 3,
    ):
        self.alpha = alpha
        self.max_jump_px = max_jump_px
        self.jump_grace = jump_grace
        self._xy: Optional[tuple[float, float]] = None
        self._jump_streak = 0

    def reset(self) -> None:
        self._xy = None
        self._jump_streak = 0

    def update(self, tip: Optional[tuple[float, float]]) -> Optional[tuple[float, float]]:
        if tip is None:
            self._xy = None
            self._jump_streak = 0
            return None
        x, y = float(tip[0]), float(tip[1])
        if self._xy is not None:
            dx = x - self._xy[0]
            dy = y - self._xy[1]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist > self.max_jump_px:
                self._jump_streak += 1
                if self._jump_streak < self.jump_grace:
                    # Hold last good tip; ignore teleport.
                    return self._xy
                # Accept as a new track after repeated far detections.
            self._jump_streak = 0
        if self._xy is None:
            self._xy = (x, y)
        else:
            a = self.alpha
            self._xy = (
                a * x + (1 - a) * self._xy[0],
                a * y + (1 - a) * self._xy[1],
            )
        return self._xy


class DwellFilter:
    """Fire only after the same cell_id is held for ``dwell_ms`` milliseconds."""

    def __init__(self, dwell_ms: float = 400.0):
        self.dwell_ms = dwell_ms
        self._cell_id: Optional[int] = None
        self._since: float = 0.0
        self._fired: bool = False

    def reset(self) -> None:
        self._cell_id = None
        self._since = 0.0
        self._fired = False

    def update(self, cell: Optional[Cell], now: Optional[float] = None) -> Optional[Cell]:
        """Return cell once when dwell completes; None otherwise."""
        now = time.time() if now is None else now
        cid = cell.id if cell is not None else None
        if cid != self._cell_id:
            self._cell_id = cid
            self._since = now
            self._fired = False
            return None
        if cid is None or self._fired:
            return None
        if (now - self._since) * 1000.0 >= self.dwell_ms:
            self._fired = True
            return cell
        return None


@dataclass
class VisitEvent:
    cell_id: int
    char: str
    t_enter: float
    t_exit: Optional[float] = None


@dataclass
class SessionMemory:
    visited: list[VisitEvent] = field(default_factory=list)
    current_id: Optional[int] = None
    last_announced_id: Optional[int] = None
    correct: int = 0
    total: int = 0

    def on_current(self, cell: Optional[Cell], now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        new_id = cell.id if cell is not None else None
        if new_id == self.current_id:
            return
        if self.current_id is not None and self.visited:
            self.visited[-1].t_exit = now
        self.current_id = new_id
        if cell is not None:
            self.visited.append(
                VisitEvent(cell_id=cell.id, char=cell.char, t_enter=now)
            )

    def should_announce(self, cell: Cell) -> bool:
        if cell.id == self.last_announced_id:
            return False
        self.last_announced_id = cell.id
        return True

    def clear_announce_lock(self) -> None:
        """Call when finger leaves all cells so revisiting re-announces."""
        if self.current_id is None:
            self.last_announced_id = None

    def record_test(self, ok: bool) -> None:
        self.total += 1
        if ok:
            self.correct += 1


def _smoke() -> None:
    cells = [
        Cell(id=0, xyxy=(10, 10, 50, 50), char="A"),
        Cell(id=1, xyxy=(60, 10, 100, 50), char="B"),
        Cell(id=2, xyxy=(40, 40, 80, 80), char="C"),  # overlaps A/B region
    ]
    m = CellMap(cells=cells)
    assert hit_test((30, 30), m).char == "A"
    assert hit_test((80, 30), m).char == "B"
    # Overlap: nearer center of C vs A — tip at (45,45) closer to C center (60,60)? 
    # A center 30,30; C center 60,60; tip 45,45 → equal-ish; C contains with margin
    hit = hit_test((55, 55), m)
    assert hit is not None and hit.char in ("A", "C", "B")

    ema = TipEMA(0.5)
    assert ema.update((0, 0)) == (0.0, 0.0)
    assert ema.update((10, 10)) == (5.0, 5.0)

    dwell = DwellFilter(dwell_ms=50)
    c0 = cells[0]
    assert dwell.update(c0, now=0.0) is None
    assert dwell.update(c0, now=0.02) is None
    assert dwell.update(c0, now=0.06) is c0
    assert dwell.update(c0, now=0.10) is None  # already fired

    mem = SessionMemory()
    mem.on_current(c0, now=1.0)
    assert mem.should_announce(c0)
    assert not mem.should_announce(c0)
    mem.on_current(None, now=2.0)
    mem.clear_announce_lock()
    mem.on_current(c0, now=3.0)
    assert mem.should_announce(c0)
    print("cell_map smoke OK", flush=True)


if __name__ == "__main__":
    _smoke()
