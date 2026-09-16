"""FT / FO cases for CellMap, TipEMA, DwellFilter, Learning/Testing (SRS FR 5-8)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_FCT = Path(__file__).resolve().parents[1]
if str(_FCT) not in sys.path:
    sys.path.insert(0, str(_FCT))

from cell_map import Cell, CellMap, DwellFilter, SessionMemory, TipEMA, hit_test  # noqa: E402
from modes import LearningMode
from modes import TestingMode as TestSessionMode  # noqa: E402
from tip_backends import FallbackTip  # noqa: E402


def _map() -> CellMap:
    """Three overlapping cells (ක, ත, ග) used as a tiny CellMap fixture."""
    return CellMap(
        cells=[
            Cell(id=0, xyxy=(10, 10, 50, 50), char="ක", code=19),
            Cell(id=1, xyxy=(60, 10, 100, 50), char="ත", code=6),
            Cell(id=2, xyxy=(40, 40, 80, 80), char="ග", code=18),
        ]
    )


def test_ft01_hit_test_inside_and_miss():
    """Check a tip inside the first box returns ක; far tip, None tip, and empty map return no cell."""
    m = _map()
    assert hit_test((30, 30), m).char == "ක"
    assert hit_test((500, 500), m) is None
    assert hit_test(None, m) is None
    assert hit_test((30, 30), CellMap()) is None


def test_ft02_hit_test_overlap_nearest_center():
    """Check a tip in overlapping boxes still returns one of those cells (nearest-centre), never None."""
    m = _map()
    hit = hit_test((55, 55), m)
    assert hit is not None
    assert hit.char in ("ක", "ත", "ග")


def test_ft03_tip_ema_smooths_and_rejects_teleport():
    """Check EMA moves to the midpoint (5,5) then ignores a 400 px jump (ghost / teleport reject)."""
    ema = TipEMA(alpha=0.5, max_jump_px=50.0, lost_frames_to_retarget=8, coast_frames=2)
    assert ema.update((0, 0)) == (0.0, 0.0)
    assert ema.update((10, 10)) == (5.0, 5.0)
    held = ema.update((400, 400))
    assert held == (5.0, 5.0)


def test_ft04_tip_ema_coasts_then_clears():
    """Check missing frames still return the last tip for coast_frames, then the track clears."""
    ema = TipEMA(alpha=0.5, coast_frames=2, lost_frames_to_retarget=3)
    ema.update((1, 1))
    assert ema.update(None) == (1.0, 1.0)
    assert ema.update(None) == (1.0, 1.0)
    assert ema.update(None) is None  # past coast
    assert ema.update(None) is None  # retarget threshold clears track


def test_ft05_dwell_fires_once():
    """Check holding the same cell fires exactly once after 50 ms, then stays silent (no repeated TTS)."""
    dwell = DwellFilter(dwell_ms=50)
    c0 = Cell(id=0, xyxy=(0, 0, 10, 10), char="A")
    assert dwell.update(c0, now=0.0) is None
    assert dwell.update(c0, now=0.02) is None
    assert dwell.update(c0, now=0.06) is c0
    assert dwell.update(c0, now=0.10) is None


def test_ft06_dwell_resets_on_cell_change():
    """Check moving from cell A to B restarts the dwell timer so B fires after its own 50 ms."""
    dwell = DwellFilter(dwell_ms=50)
    a = Cell(id=0, xyxy=(0, 0, 10, 10), char="A")
    b = Cell(id=1, xyxy=(20, 0, 30, 10), char="B")
    dwell.update(a, now=0.0)
    assert dwell.update(b, now=0.06) is None
    assert dwell.update(b, now=0.12) is b


def test_ft07_learning_announces_once_until_leave():
    """Check LearningMode announces a cell once; after on_leave() the same cell can be announced again."""
    mode = LearningMode()
    cell = Cell(id=0, xyxy=(0, 0, 10, 10), char="ක", code=19)
    ev = mode.on_dwell(cell)
    assert ev is not None and ev.kind == "announce"
    assert mode.on_dwell(cell) is None
    mode.on_leave()
    ev2 = mode.on_dwell(cell)
    assert ev2 is not None and ev2.kind == "announce"


def test_ft08_testing_match_and_mismatch():
    """Check TestingMode scores 'ක' as correct and 'ත' as wrong; score is 1/2."""
    mode = TestSessionMode()
    cell = Cell(id=1, xyxy=(0, 0, 10, 10), char="ක")
    prompt = mode.on_dwell(cell)
    assert prompt.kind == "prompt"
    ok = mode.submit_answer("ක")
    assert ok.ok is True
    mode.on_dwell(cell)
    bad = mode.submit_answer("ත")
    assert bad.ok is False
    assert mode.memory.correct == 1
    assert mode.memory.total == 2


def test_fo01_yolo_miss_falls_back_to_skin():
    """Check when YOLO returns no tip, FallbackTip uses SkinContourTip and records that backend."""
    class Primary:
        name = "TipYOLO"

        def detect(self, _frame):
            return None, None, 0.0

    class Fallback:
        name = "SkinContourTip"

        def detect(self, _frame):
            return (12.0, 34.0), (1, 2, 3, 4), 0.5

    fb = FallbackTip(Primary(), Fallback())
    tip, box, conf = fb.detect(object())
    assert tip == (12.0, 34.0)
    assert fb.last_backend == "SkinContourTip"
    assert conf == pytest.approx(0.5)


def test_ft09_session_memory_does_not_mutate_cellmap():
    """Check recording a visit/score does not add/remove cells or change the stored character."""
    m = _map()
    n = len(m.cells)
    mem = SessionMemory()
    mem.on_current(m.cells[0], now=1.0)
    mem.record_test(True)
    assert len(m.cells) == n
    assert m.by_id(0).char == "ක"
