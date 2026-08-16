"""Stage 2a - Integration: three dataset formats into one cell-level manifest.

    DSBI      grid lines + per-cell 6 dot bits        (flatbed scans, 200 dpi)
    Angelina  normalized cell boxes + code           (handheld phone photos)
    Gold      LabelMe rectangles + dot strings       (our Sinhala pages)
                        |
                        v
              manifest_raw.csv   one row per Braille cell

Parsing is delegated to the readers that already exist and are already trusted
by the training code, so there is exactly one implementation of each format:
`_parse_annotation` / `_cell_box` from braille_cnn.dbsi_dataset and
`_read_csv_annotation` from braille_cnn.angelina_dataset.

Usage (from repo root):
    py -3.11 -m data_pipeline.integrate
    py -3.11 -m data_pipeline.integrate --sources dbsi angelina --split-mode rebalance
    py -3.11 -m data_pipeline.integrate --sources gold        # after labelling
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image

from .contracts import (
    CellRow,
    code_to_dot_string,
    dot_string_to_code,
    read_manifest,
    repo_root,
    summarize_manifest,
    validate_manifest,
    write_manifest,
)

ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from braille_cnn.angelina_dataset import _read_csv_annotation  # noqa: E402
from braille_cnn.dbsi_dataset import (  # noqa: E402
    _cell_box,
    _dots_to_code,
    _parse_annotation,
    _parse_split_file,
)

DEFAULT_DBSI_ROOT = ROOT / "data DBSI" / "data"
DEFAULT_ANGELINA_ROOT = ROOT / "data Angelina" / "books"
DEFAULT_GOLD_ROOT = ROOT / "Gold Dataset"
DEFAULT_OUT = ROOT / "data_pipeline" / "manifests" / "manifest_raw.csv"

# DSBI annotations give the dot *extent* (the 2x3 lattice), while Angelina
# annotates a full cell. To make one box mean one thing across sources, the DSBI
# extent is expanded by this fraction of the within-cell dot pitch.
#
# Calibrated against Stage 3 EDA rather than guessed. A DSBI box is
# 2*dy + 2*margin*dy tall; scaling both datasets to a common page height, an
# Angelina box measures about 2.6 dot pitches, which needs margin 0.3.
# The existing DBSIDataset uses 0.8, giving 3.6 pitches - wider than the ~4
# pitch line spacing, so those crops contain dots from the lines above and
# below. That is survivable when training on DSBI alone, but it makes a DSBI
# crop a visibly different object from an Angelina crop, which is exactly the
# domain gap this pipeline exists to close.
DBSI_MARGIN_SCALE = 0.35

# Gold pages are shot twice under different lighting; both folders hold the
# same 12 physical pages, so they must share a page_group.
GOLD_VARIANTS = {"High quality dataset": "high", "Low quality dataset": "low"}

# Page-level split for the 12 unique Gold pages (Stage 1b). Both lighting
# variants of a page follow its number, so a page never spans two splits.
GOLD_PAGE_SPLITS = {
    1: "train", 2: "train", 3: "train", 4: "train",
    5: "train", 6: "train", 7: "train", 8: "train",
    9: "val", 10: "val",
    11: "test", 12: "test",
}

REBALANCE_RATIOS = (0.70, 0.15, 0.15)  # train, val, test


def _rel(path: Path) -> str:
    """Repo-relative POSIX path, so manifests are portable between machines."""
    return Path(path).resolve().relative_to(ROOT).as_posix()


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def _pitch(y0: float, y1: float) -> float:
    """A cell box spans 3 dot rows = 2 vertical pitches."""
    return max((y1 - y0) / 2.0, 1e-6)


def _stable_bucket(key: str) -> float:
    """Deterministic value in [0, 1) from a string, stable across machines.

    Python's hash() is salted per process, so it cannot be used for a split
    that must reproduce on Colab and on this laptop.
    """
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0x100000000


def _rebalanced_split(page_group: str) -> str:
    bucket = _stable_bucket(page_group)
    train_end = REBALANCE_RATIOS[0]
    val_end = train_end + REBALANCE_RATIOS[1]
    if bucket < train_end:
        return "train"
    if bucket < val_end:
        return "val"
    return "test"


# --------------------------------------------------------------------------
# DSBI
# --------------------------------------------------------------------------


def read_dbsi(root: Path, split_mode: str, sides=("recto", "verso")) -> list[CellRow]:
    """DSBI grid annotations -> CellRow list.

    The official split is inverted for training use (26 train pages against 88
    test), which is why --split-mode rebalance exists. The official split is
    still the one to quote for any comparison with published DSBI numbers.
    """
    rows: list[CellRow] = []
    official: dict[tuple[str, str], str] = {}
    for split_name, split_file in (("train", "train.txt"), ("test", "test.txt")):
        path = root / split_file
        if not path.exists():
            print(f"  ! missing DSBI split file: {path}")
            continue
        for book, base in _parse_split_file(path):
            official[(book, base)] = split_name

    for (book, base), split_name in sorted(official.items()):
        page_group = f"dbsi:{book}/{base}"
        split = split_name if split_mode == "official" else _rebalanced_split(page_group)
        for side in sides:
            txt_path = root / book / f"{base}+{side}.txt"
            img_path = root / book / f"{base}+{side}.jpg"
            if not txt_path.exists() or not img_path.exists():
                continue
            parsed = _parse_annotation(txt_path)
            if parsed is None:  # page has no dots on this side
                continue
            verticals, horizontals, cells = parsed
            img_w, img_h = _image_size(img_path)
            rel = _rel(img_path)
            for row_i, col_i, dots in cells:
                try:
                    box = _cell_box(verticals, horizontals, row_i, col_i, DBSI_MARGIN_SCALE)
                except IndexError:
                    # annotation references a grid position its own header does
                    # not define; clean.py would drop it anyway, skip early
                    continue
                x0, y0, x1, y1 = box
                code = _dots_to_code(dots)
                rows.append(
                    CellRow(
                        source="dbsi",
                        image_path=rel,
                        page_group=page_group,
                        book=book,
                        page=base,
                        side=side,
                        split=split,
                        x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1),
                        code=int(code),
                        dots=code_to_dot_string(code),
                        img_w=int(img_w), img_h=int(img_h),
                        dot_pitch_px=_pitch(y0, y1),
                    )
                )
    return rows


# --------------------------------------------------------------------------
# Angelina
# --------------------------------------------------------------------------


def read_angelina(root: Path, split_mode: str) -> list[CellRow]:
    """Angelina normalized boxes -> CellRow list.

    Angelina ships books/train.txt and books/val.txt and no test list, so under
    --split-mode official its val set doubles as the test set. Say so in the
    report rather than pretending there are three splits.
    """
    rows: list[CellRow] = []
    official: dict[Path, str] = {}
    for split_name, split_file in (("train", "train.txt"), ("val", "val.txt")):
        path = root / split_file
        if not path.exists():
            print(f"  ! missing Angelina split file: {path}")
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                rel_line = line.strip().replace("\\", "/")
                if rel_line:
                    official[root / rel_line] = split_name

    for img_path, split_name in sorted(official.items()):
        csv_path = img_path.parent / (img_path.stem + ".csv")
        if not img_path.exists() or not csv_path.exists():
            continue
        rects = _read_csv_annotation(csv_path)  # already skips markout code 63
        if not rects:
            continue
        book = img_path.parent.name
        page = img_path.stem
        page_group = f"angelina:{book}/{page}"
        split = split_name if split_mode == "official" else _rebalanced_split(page_group)
        img_w, img_h = _image_size(img_path)
        rel = _rel(img_path)
        for left, top, right, bottom, code in rects:
            x0, y0 = left * img_w, top * img_h
            x1, y1 = right * img_w, bottom * img_h
            rows.append(
                CellRow(
                    source="angelina",
                    image_path=rel,
                    page_group=page_group,
                    book=book,
                    page=page,
                    side="",
                    split=split,
                    x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1),
                    code=int(code),
                    dots=code_to_dot_string(code),
                    img_w=int(img_w), img_h=int(img_h),
                    dot_pitch_px=_pitch(y0, y1),
                )
            )
    return rows


# --------------------------------------------------------------------------
# Gold (Stage 1b - only produces rows once LabelMe JSONs exist)
# --------------------------------------------------------------------------


def _gold_page_number(stem: str) -> int | None:
    """"pg-7" -> 7. Returns None for anything not matching that pattern."""
    if not stem.lower().startswith("pg-"):
        return None
    try:
        return int(stem[3:])
    except ValueError:
        return None


def read_gold(root: Path, split_mode: str) -> list[CellRow]:
    """LabelMe rectangles with dot-string labels -> CellRow list.

    Labels are dot strings such as "1345", not Sinhala characters: the CNN's 64
    classes are dot codes, so a dot string converts losslessly with no lookup
    table, whereas Sinhala-to-code is many-to-one and ambiguous for the
    two-cell characters. The Sinhala glyph is still recoverable for display via
    braille_cnn.labels.CODE_TO_SINHALA.
    """
    rows: list[CellRow] = []
    for folder, variant in GOLD_VARIANTS.items():
        folder_path = root / folder
        if not folder_path.is_dir():
            continue
        for json_path in sorted(folder_path.glob("*.json")):
            img_path = next(
                (p for p in (json_path.with_suffix(ext) for ext in (".jpeg", ".jpg", ".png"))
                 if p.exists()),
                None,
            )
            if img_path is None:
                print(f"  ! {json_path.name} has no matching image, skipped")
                continue

            page_no = _gold_page_number(json_path.stem)
            if page_no is None:
                print(f"  ! cannot read a page number from {json_path.name}, skipped")
                continue

            # both lighting variants of one physical page share this key
            page_group = f"gold:pg-{page_no}"
            if split_mode == "official":
                split = GOLD_PAGE_SPLITS.get(page_no, "train")
            else:
                split = _rebalanced_split(page_group)

            with open(json_path, encoding="utf-8") as f:
                doc = json.load(f)
            img_w = int(doc.get("imageWidth") or 0)
            img_h = int(doc.get("imageHeight") or 0)
            if not img_w or not img_h:
                img_w, img_h = _image_size(img_path)
            rel = _rel(img_path)

            for shape in doc.get("shapes", []):
                if shape.get("shape_type") != "rectangle":
                    print(f"  ! {json_path.name}: skipping non-rectangle shape")
                    continue
                (px0, py0), (px1, py1) = shape["points"][:2]
                x0, x1 = sorted((float(px0), float(px1)))
                y0, y1 = sorted((float(py0), float(py1)))
                try:
                    code = dot_string_to_code(shape.get("label", ""))
                except ValueError as exc:
                    print(f"  ! {json_path.name}: {exc}")
                    continue
                rows.append(
                    CellRow(
                        source="gold",
                        image_path=rel,
                        page_group=page_group,
                        book=variant,
                        page=f"pg-{page_no}",
                        side="",
                        split=split,
                        x0=x0, y0=y0, x1=x1, y1=y1,
                        code=int(code),
                        dots=code_to_dot_string(code),
                        img_w=img_w, img_h=img_h,
                        dot_pitch_px=_pitch(y0, y1),
                    )
                )
    return rows


# --------------------------------------------------------------------------


READERS = {"dbsi": read_dbsi, "angelina": read_angelina, "gold": read_gold}

DEFAULT_ROOTS = {
    "dbsi": DEFAULT_DBSI_ROOT,
    "angelina": DEFAULT_ANGELINA_ROOT,
    "gold": DEFAULT_GOLD_ROOT,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2a - build one cell-level manifest from all sources"
    )
    parser.add_argument("--sources", nargs="+", default=["dbsi", "angelina"],
                        choices=sorted(READERS), help="Which datasets to integrate")
    parser.add_argument("--dbsi-root", type=Path, default=DEFAULT_DBSI_ROOT)
    parser.add_argument("--angelina-root", type=Path, default=DEFAULT_ANGELINA_ROOT)
    parser.add_argument("--gold-root", type=Path, default=DEFAULT_GOLD_ROOT)
    parser.add_argument("--split-mode", choices=["official", "rebalance"], default="official",
                        help="official = each dataset's own split files; "
                             "rebalance = deterministic 70/15/15 by page group")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    roots = {"dbsi": args.dbsi_root, "angelina": args.angelina_root, "gold": args.gold_root}

    all_rows: list[CellRow] = []
    for source in args.sources:
        root = roots[source]
        if not Path(root).exists():
            print(f"{source}: root not found, skipped ({root})")
            continue
        print(f"{source}: reading {root} ...")
        rows = READERS[source](Path(root), args.split_mode)
        print(f"{source}: {len(rows):,} cells")
        if source == "gold" and not rows:
            print("  (expected until Stage 1b labelling is done - see Gold Dataset/"
                  "ANNOTATION_GUIDELINES.md)")
        all_rows.extend(rows)

    if not all_rows:
        raise SystemExit("No cells read from any source; nothing written.")

    out_path = write_manifest(all_rows, args.out)
    frame = read_manifest(out_path)

    print(f"\nWrote {out_path}")
    print(summarize_manifest(frame))

    problems = validate_manifest(frame)
    if problems:
        print("\nValidation warnings (clean.py handles most of these):")
        for p in problems:
            print(f"  - {p}")
    else:
        print("\nValidation: clean.")
    print("\nNext: py -3.11 -m data_pipeline.clean")


if __name__ == "__main__":
    main()
