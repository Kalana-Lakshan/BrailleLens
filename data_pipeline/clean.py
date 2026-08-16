"""Stage 2b - Cleaning: every rule logs what it removed and why.

Reads  manifest_raw.csv
Writes manifest_clean.csv  +  reports/cleaning_log.md

The log is the point. A cleaning step that silently drops 8% of a dataset is
indistinguishable from a bug, so each rule reports its own count and the report
is a deliverable, not a debug print.

Usage (from repo root):
    py -3.11 -m data_pipeline.clean
    py -3.11 -m data_pipeline.clean --drop-rulers
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import (
    MARKOUT_CODE,
    dot_string_to_code,
    read_manifest,
    repo_root,
    summarize_manifest,
    validate_manifest,
    write_manifest,
)

ROOT = repo_root()
DEFAULT_IN = ROOT / "data_pipeline" / "manifests" / "manifest_raw.csv"
DEFAULT_OUT = ROOT / "data_pipeline" / "manifests" / "manifest_clean.csv"
DEFAULT_LOG = ROOT / "reports" / "cleaning_log.md"

# A decorative divider row is a long run of cells that nearly all carry the
# same code - real Braille text almost never does that.
RULER_MIN_CELLS = 15
RULER_SAME_CODE_FRACTION = 0.80

# Boxes may stick out of the page by this fraction of their own size before
# being treated as an annotation error rather than a rounding artifact.
OUT_OF_BOUNDS_TOLERANCE = 0.5


class DropLog:
    """Collects one line per rule so the report matches what actually ran."""

    def __init__(self, total: int) -> None:
        self.start_total = total
        self.entries: list[tuple[str, int, str]] = []

    def record(self, rule: str, removed: int, note: str = "") -> None:
        self.entries.append((rule, removed, note))

    def to_markdown(self, final_total: int) -> str:
        lines = [
            "# Stage 2b cleaning log",
            "",
            f"Input rows: **{self.start_total:,}**",
            f"Output rows: **{final_total:,}** "
            f"({final_total / max(self.start_total, 1):.2%} retained)",
            "",
            "| Rule | Rows removed | Note |",
            "|---|---:|---|",
        ]
        for rule, removed, note in self.entries:
            lines.append(f"| {rule} | {removed:,} | {note} |")
        return "\n".join(lines) + "\n"

    def print_console(self) -> None:
        for rule, removed, note in self.entries:
            suffix = f"  ({note})" if note else ""
            print(f"  {rule:38s} -{removed:6,d}{suffix}")


def _assign_rows(frame: pd.DataFrame) -> pd.Series:
    """Cluster cells on each page into reading-order rows by y-centre.

    Used only by the ruler-line rule. A simple pitch-based threshold is enough
    here: cells on one line share a y-centre to well within one dot pitch.
    """
    y_centre = (frame["y0"] + frame["y1"]) / 2.0
    out = pd.Series(-1, index=frame.index, dtype=int)
    for _, idx in frame.groupby("image_path").groups.items():
        sub = y_centre.loc[idx].sort_values()
        pitch = float(frame.loc[idx, "dot_pitch_px"].median())
        threshold = max(pitch * 1.5, 1.0)
        row_id = 0
        previous = None
        for i, value in sub.items():
            if previous is not None and (value - previous) > threshold:
                row_id += 1
            out.loc[i] = row_id
            previous = value
    return out


def _ruler_mask(frame: pd.DataFrame) -> pd.Series:
    """True for cells belonging to a decorative divider row."""
    rows = _assign_rows(frame)
    key = frame["image_path"].astype(str) + "#" + rows.astype(str)
    mask = pd.Series(False, index=frame.index)
    for _, idx in frame.groupby(key).groups.items():
        codes = frame.loc[idx, "code"]
        if len(codes) < RULER_MIN_CELLS:
            continue
        dominant = codes.value_counts().iloc[0] / len(codes)
        if dominant >= RULER_SAME_CODE_FRACTION:
            mask.loc[idx] = True
    return mask


def clean(frame: pd.DataFrame, drop_rulers: bool, check_images: bool) -> tuple[pd.DataFrame, DropLog]:
    log = DropLog(len(frame))

    # 1. Angelina reserves all-six-dots for "XX" markout, so it is not a real
    #    cell there. For DSBI and Gold a 6-dot cell is legitimate and stays.
    before = len(frame)
    markout = (frame["source"] == "angelina") & (frame["code"] == MARKOUT_CODE)
    frame = frame[~markout]
    log.record(
        "Angelina markout (code 63)",
        before - len(frame),
        "already filtered by _read_csv_annotation, kept as a guard",
    )

    # 2. Geometry that cannot be cropped.
    before = len(frame)
    degenerate = (frame["x1"] <= frame["x0"]) | (frame["y1"] <= frame["y0"])
    frame = frame[~degenerate]
    log.record("Zero-area or inverted boxes", before - len(frame))

    # 3. Boxes far outside the page. A small overhang is normal because DSBI
    #    boxes are grid-extent plus a margin, so tolerate it and clip later.
    before = len(frame)
    width = frame["x1"] - frame["x0"]
    height = frame["y1"] - frame["y0"]
    out_of_bounds = (
        (frame["x1"] < -OUT_OF_BOUNDS_TOLERANCE * width)
        | (frame["y1"] < -OUT_OF_BOUNDS_TOLERANCE * height)
        | (frame["x0"] > frame["img_w"] + OUT_OF_BOUNDS_TOLERANCE * width)
        | (frame["y0"] > frame["img_h"] + OUT_OF_BOUNDS_TOLERANCE * height)
    )
    frame = frame[~out_of_bounds]
    log.record("Boxes outside the page", before - len(frame),
               f"tolerance {OUT_OF_BOUNDS_TOLERANCE:.0%} of box size")

    # 4. Exact duplicate annotations of the same cell.
    before = len(frame)
    frame = frame.drop_duplicates(subset=["image_path", "x0", "y0", "x1", "y1", "code"])
    log.record("Duplicate cell annotations", before - len(frame))

    # 5. Absurd cell shapes - a Braille cell is roughly 2 wide by 3 tall, so an
    #    aspect ratio far from that means the annotation is wrong.
    before = len(frame)
    aspect = (frame["x1"] - frame["x0"]) / (frame["y1"] - frame["y0"])
    frame = frame[aspect.between(0.15, 6.0)]
    log.record("Implausible aspect ratio", before - len(frame), "kept 0.15 to 6.0")

    # 6. Decorative divider rows: reported always, dropped only on request,
    #    because on some pages they are genuine content.
    ruler = _ruler_mask(frame)
    if drop_rulers:
        before = len(frame)
        frame = frame[~ruler]
        log.record("Decorative divider rows", before - len(frame), "--drop-rulers was set")
    else:
        log.record("Decorative divider rows", 0,
                   f"{int(ruler.sum()):,} flagged but kept; pass --drop-rulers to remove")

    # 7. Rows pointing at an image that is not on disk.
    if check_images:
        before = len(frame)
        existing = {p: (ROOT / p).exists() for p in frame["image_path"].unique()}
        frame = frame[frame["image_path"].map(existing)]
        log.record("Missing page image", before - len(frame))
    else:
        log.record("Missing page image", 0, "skipped (--no-check-images)")

    # 8. Consistency between the two label columns.
    before = len(frame)
    frame = frame[frame["dots"].map(dot_string_to_code) == frame["code"]]
    log.record("dots / code disagreement", before - len(frame))

    return frame.reset_index(drop=True), log


def leakage_report(frame: pd.DataFrame) -> list[str]:
    """Page groups appearing in more than one split. Must be empty."""
    spans = frame.groupby("page_group")["split"].nunique()
    return sorted(spans[spans > 1].index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2b - clean the cell manifest")
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--drop-rulers", action="store_true",
                        help="Remove decorative divider rows instead of only flagging them")
    parser.add_argument("--no-check-images", action="store_true",
                        help="Skip verifying every page image exists (faster)")
    args = parser.parse_args()

    frame = read_manifest(args.in_path)
    print(f"Read {len(frame):,} rows from {args.in_path}")

    cleaned, log = clean(frame, args.drop_rulers, not args.no_check_images)
    print("\nDrop rules:")
    log.print_console()

    leaked = leakage_report(cleaned)
    if leaked:
        raise SystemExit(
            "\nLEAKAGE: these page groups appear in more than one split, so any "
            "accuracy measured on this manifest would be inflated:\n  "
            + "\n  ".join(leaked[:20])
        )
    print("\nLeakage check: no page group spans two splits.")

    problems = validate_manifest(cleaned)
    if problems:
        print("\nRemaining validation problems:")
        for p in problems:
            print(f"  - {p}")

    write_manifest(cleaned, args.out)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(log.to_markdown(len(cleaned)), encoding="utf-8")

    print(f"\nWrote {args.out}")
    print(f"Wrote {args.log}")
    print(summarize_manifest(cleaned))
    print("\nNext: py -3.11 -m data_pipeline.analyze")


if __name__ == "__main__":
    main()
