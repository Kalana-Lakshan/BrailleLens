"""The one data contract shared by every stage: a cell-level manifest.

DSBI, Angelina and Gold store Braille geometry in three incompatible formats.
Rather than teaching every training script all three, Stage 2a converts all of
them into a single CSV with one row per Braille cell. Every stage after that
reads only this schema, so adding a fourth dataset later touches integrate.py
and nothing else.

Key design point: `page_group`
------------------------------
Splits are assigned per *page group*, never per file, because several files can
show the same physical page:

  * Gold  - "High quality/pg-3.jpeg" and "Low quality/pg-3.jpeg" are the same
            page shot twice under different lighting.
  * DSBI  - "+recto" and "+verso" are two sides of one sheet, scanned together.

Splitting on filename would put the same Braille content in both train and
test and silently inflate every accuracy number in the report. `page_group` is
the key that prevents it, and clean.py asserts no group spans two splits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import pandas as pd

# Column order is part of the contract; downstream code may rely on it.
MANIFEST_COLUMNS = [
    "source",  # dbsi | angelina | gold
    "image_path",  # repo-relative path to the page image
    "page_group",  # leakage key - see module docstring
    "book",
    "page",
    "side",  # recto | verso | "" when not applicable
    "split",  # train | val | test
    "x0",  # cell box in absolute pixels, inclusive-exclusive
    "y0",
    "x1",
    "y1",
    "code",  # 0-63, dot i -> bit (i-1)
    "dots",  # human-readable form of code, e.g. "1345"
    "img_w",
    "img_h",
    "dot_pitch_px",  # (y1 - y0) / 2, see note below
]

# dot_pitch_px is deliberately derived the same way for every source:
# (y1 - y0) / 2.0, because a cell box spans three dot rows and therefore two
# vertical dot pitches. DSBI could give a truer pitch straight from its grid
# lines, but mixing a measured pitch for one source with an estimated one for
# another in the same column is exactly how scale bugs get hidden. One
# definition, applied everywhere, and analyze.py plots the real distributions.

SOURCES = ("dbsi", "angelina", "gold")
SPLITS = ("train", "val", "test")

# Angelina reserves all-six-dots for "XX" markout (illegible / crossed out),
# so it is not a real 6-dot cell. See data Angelina/README.md.
MARKOUT_CODE = 63


@dataclass
class CellRow:
    """One Braille cell. Field order matches MANIFEST_COLUMNS."""

    source: str
    image_path: str
    page_group: str
    book: str
    page: str
    side: str
    split: str
    x0: float
    y0: float
    x1: float
    y1: float
    code: int
    dots: str
    img_w: int
    img_h: int
    dot_pitch_px: float

    def as_dict(self) -> dict:
        return asdict(self)


# Sanity check at import: dataclass and column list must not drift apart.
_DATACLASS_FIELDS = [f.name for f in fields(CellRow)]
if _DATACLASS_FIELDS != MANIFEST_COLUMNS:
    raise RuntimeError(
        "CellRow fields and MANIFEST_COLUMNS are out of sync:\n"
        f"  dataclass: {_DATACLASS_FIELDS}\n"
        f"  columns  : {MANIFEST_COLUMNS}"
    )


# --------------------------------------------------------------------------
# dot code helpers
#
# Convention used everywhere in this repo (matches braille_cnn.labels and
# Angelina's own label_tools.py `v = [1, 2, 4, 8, 16, 32]`):
#
#     dot1 -> bit 0        d1 d4
#     dot2 -> bit 1        d2 d5
#     dot3 -> bit 2        d3 d6
#     dot4 -> bit 3
#     dot5 -> bit 4
#     dot6 -> bit 5
# --------------------------------------------------------------------------


def dots_to_code(dots) -> int:
    """6 raised/not-raised flags in dot1..dot6 order -> code 0-63."""
    return sum((1 << i) for i, d in enumerate(dots) if int(d) == 1)


def code_to_dots(code: int) -> tuple[int, ...]:
    """Code 0-63 -> 6 flags in dot1..dot6 order."""
    return tuple((int(code) >> i) & 1 for i in range(6))


def dot_string_to_code(dot_string: str) -> int:
    """Annotation shorthand -> code. "1345" -> dots 1,3,4,5 raised.

    Accepts "" or "0" for a blank cell. Raises ValueError on any digit outside
    1-6 or on a repeated digit, both of which are annotation mistakes worth
    failing loudly on rather than silently mislabelling a cell.
    """
    s = str(dot_string).strip()
    if s in ("", "0", "-"):
        return 0
    seen = set()
    code = 0
    for ch in s:
        if ch not in "123456":
            raise ValueError(f"Bad dot string {dot_string!r}: {ch!r} is not a dot 1-6")
        if ch in seen:
            raise ValueError(f"Bad dot string {dot_string!r}: dot {ch} repeated")
        seen.add(ch)
        code |= 1 << (int(ch) - 1)
    return code


def code_to_dot_string(code: int) -> str:
    """Code -> annotation shorthand. 0 becomes "0" (blank cell)."""
    dots = [str(i + 1) for i in range(6) if (int(code) >> i) & 1]
    return "".join(dots) if dots else "0"


# --------------------------------------------------------------------------
# manifest IO
# --------------------------------------------------------------------------


def write_manifest(rows, path: str | Path) -> Path:
    """Write a manifest to CSV in contract column order.

    Accepts a DataFrame, or any iterable of CellRow objects or dicts, so the
    stage scripts can pass whichever they naturally hold.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, pd.DataFrame):
        frame = rows.reindex(columns=MANIFEST_COLUMNS)
    else:
        records = [r.as_dict() if isinstance(r, CellRow) else dict(r) for r in rows]
        frame = pd.DataFrame(records, columns=MANIFEST_COLUMNS)
    frame.to_csv(path, index=False, encoding="utf-8")
    return path


def read_manifest(path: str | Path) -> pd.DataFrame:
    """Read a manifest CSV and verify it matches the contract."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {path}\n"
            "Build it first: py -3.11 -m data_pipeline.integrate"
        )
    frame = pd.read_csv(path, encoding="utf-8", keep_default_na=False)
    missing = [c for c in MANIFEST_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Manifest {path} is missing columns: {missing}")
    # side/dots are strings; keep_default_na=False stops "" becoming NaN
    for col in ("source", "image_path", "page_group", "book", "page", "side", "split", "dots"):
        frame[col] = frame[col].astype(str)
    return frame[MANIFEST_COLUMNS]


def validate_manifest(frame: pd.DataFrame) -> list[str]:
    """Return a list of human-readable problems; empty list means valid."""
    problems: list[str] = []

    bad_source = sorted(set(frame["source"]) - set(SOURCES))
    if bad_source:
        problems.append(f"unknown source values: {bad_source}")

    bad_split = sorted(set(frame["split"]) - set(SPLITS))
    if bad_split:
        problems.append(f"unknown split values: {bad_split}")

    if not frame["code"].between(0, 63).all():
        n = int((~frame["code"].between(0, 63)).sum())
        problems.append(f"{n} rows have code outside 0-63")

    degenerate = ((frame["x1"] <= frame["x0"]) | (frame["y1"] <= frame["y0"])).sum()
    if degenerate:
        problems.append(f"{int(degenerate)} rows have a zero-area or inverted box")

    # the leakage check that matters most
    spans = frame.groupby("page_group")["split"].nunique()
    leaked = spans[spans > 1]
    if len(leaked):
        problems.append(
            f"{len(leaked)} page_group(s) appear in more than one split "
            f"(first few: {list(leaked.index[:5])})"
        )

    mismatched = frame[frame["dots"].map(dot_string_to_code) != frame["code"]]
    if len(mismatched):
        problems.append(f"{len(mismatched)} rows where 'dots' disagrees with 'code'")

    return problems


def summarize_manifest(frame: pd.DataFrame) -> str:
    """Compact text summary used by several CLIs."""
    lines = [f"{len(frame):,} cells from {frame['image_path'].nunique():,} images"]
    by_source = frame.groupby("source").agg(
        cells=("code", "size"),
        images=("image_path", "nunique"),
        groups=("page_group", "nunique"),
    )
    for source, row in by_source.iterrows():
        lines.append(
            f"  {source:9s} {row['cells']:8,d} cells  "
            f"{row['images']:4,d} images  {row['groups']:4,d} page groups"
        )
    counts = frame["split"].value_counts()
    split_bits = "  ".join(f"{s}={counts.get(s, 0):,}" for s in SPLITS)
    lines.append(f"  splits: {split_bits}")
    return "\n".join(lines)


def repo_root() -> Path:
    """Repo root, found by walking up to the folder holding braille_cnn/."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "braille_cnn").is_dir():
            return parent
    raise RuntimeError("Could not find BrailleLens repo root (no braille_cnn/)")
