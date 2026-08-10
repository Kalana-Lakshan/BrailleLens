"""Learning and Testing mode state machines for fingertip-selected cells."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from cell_map import Cell, SessionMemory


@dataclass
class ModeEvent:
    kind: str  # announce | prompt | score | info
    message: str
    cell: Optional[Cell] = None
    ok: Optional[bool] = None


@dataclass
class LearningMode:
    memory: SessionMemory = field(default_factory=SessionMemory)

    def on_dwell(self, cell: Cell) -> Optional[ModeEvent]:
        self.memory.on_current(cell)
        if not self.memory.should_announce(cell):
            return None
        ch = cell.char or "?"
        return ModeEvent(
            kind="announce",
            message=f"[LEARN] cell {cell.id} → {ch}",
            cell=cell,
        )

    def on_leave(self) -> None:
        self.memory.on_current(None)
        self.memory.clear_announce_lock()


@dataclass
class TestingMode:
    memory: SessionMemory = field(default_factory=SessionMemory)
    pending: Optional[Cell] = None
    awaiting_answer: bool = False

    def on_dwell(self, cell: Cell) -> Optional[ModeEvent]:
        self.memory.on_current(cell)
        if self.awaiting_answer:
            return None
        self.pending = cell
        self.awaiting_answer = True
        ch_hint = cell.char  # expected (hidden from user in real UI)
        return ModeEvent(
            kind="prompt",
            message=(
                f"[TEST] cell {cell.id}: type the character then Enter "
                f"(debug expected={ch_hint!r})"
            ),
            cell=cell,
        )

    def submit_answer(self, answer: str) -> ModeEvent:
        cell = self.pending
        self.awaiting_answer = False
        expected = (cell.char if cell else "").strip().lower()
        got = answer.strip().lower()
        ok = bool(expected) and got == expected
        self.memory.record_test(ok)
        self.pending = None
        status = "CORRECT" if ok else "WRONG"
        return ModeEvent(
            kind="score",
            message=(
                f"[TEST] {status}: you={got!r} expected={expected!r} "
                f"score={self.memory.correct}/{self.memory.total}"
            ),
            cell=cell,
            ok=ok,
        )

    def on_leave(self) -> None:
        self.memory.on_current(None)
        # keep awaiting_answer so user can still type
