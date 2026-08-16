"""Tiny helper so every Stage 6 script writes the same kind of markdown report."""

from __future__ import annotations

from pathlib import Path


def write_eval_report(path: Path, title: str, lines: list[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [f"# {title}", ""] + lines + [""]
    path.write_text("\n".join(body), encoding="utf-8")
    print(f"wrote {path}")
    return path
