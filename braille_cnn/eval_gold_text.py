"""Stage 6 (approximate) -- page-text accuracy on the 6 hand-transcribed Gold
pages in "Gold Dataset/Text pages/pg-N.txt".

These are NOT the Stage 1b LabelMe cell boxes (see ANNOTATION_GUIDELINES.md)
-- they are a per-page, reading-order transcription where each number is one
printed symbol's ID in "Gold Dataset/symbols guide.jpeg" (a Braille alphabet
chart the labeller numbered by hand: a=1..z=26, space=29, capital=37,
digits=51-60, and ~20 punctuation marks in between).

Scope of this script, on purpose: only letters (ids 1-26) and space (id 29)
are decoded. Capital sign, number sign, digits, and punctuation ids are
skipped rather than guessed, because their exact raised-dot patterns would
have to be transcribed by eye from a phone photo of the chart, and
labels.py's CODE_TO_LETTER only carries dot patterns for a-z/space/
capital_sign/number_sign (the standard English Grade-1 table already in the
repo) -- nothing for punctuation. A wrong guess there would silently corrupt
the score, so this reports letters-only accuracy instead. Extend
CODE_TO_LETTER first (with dot patterns confirmed against the physical
chart, not a photo read) if punctuation coverage is ever needed.

Reports two numbers per page pair:
  - accuracy_with_spaces: normalized predicted text vs ground truth, as-is
  - accuracy_letters_only: same, with all whitespace stripped from both
    sides first (isolates recognition error from missing/extra word-gap
    detections -- the "cells" YOLO detector only proposes boxes where it
    sees raised dots, so it structurally cannot detect blank word-gap
    cells; accuracy_with_spaces will always look worse than
    accuracy_letters_only for that reason, not because letters are wrong).

    py -3.11 -m braille_cnn.eval_gold_text
"""

from __future__ import annotations

import argparse
import re
import string
from pathlib import Path

from .recognize import recognize_page
from .eval_report import write_eval_report

ROOT = Path(__file__).resolve().parent.parent
TEXT_DIR = ROOT / "Gold Dataset" / "Text pages"
IMAGE_DIR = ROOT / "Gold Dataset" / "High quality dataset"

# symbols guide.jpeg ids -> lowercase letter / space. Only the ids this repo
# already has verified dot patterns for (braille_cnn/labels.py LETTER_DOTS +
# SPECIAL_DOTS["space"]). Every other id (accent, apostrophe, letter-sign,
# ?!&,_.capital#;: "" () / @ + - * decimal number, digits) is intentionally
# left out -- see module docstring.
ID_TO_CHAR = {i: chr(ord("a") + i - 1) for i in range(1, 27)}
ID_TO_CHAR[29] = " "


def _decode_reference(path: Path) -> str:
    """pg-N.txt -> lowercase letters/spaces only, word groups preserved.

    Each physical line is "id id id   id id   id id id id" -- a run of 2+
    spaces separates words (matches how the labeller typed it), a single
    space separates letters within a word. Runs of dots (divider rows) and
    blank lines have no digits and contribute nothing.
    """
    words: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or not any(ch.isdigit() for ch in line):
            continue
        for group in re.split(r"\s{2,}", line):
            ids = [int(tok) for tok in group.split() if tok.isdigit()]
            letters = "".join(ID_TO_CHAR.get(i, "") for i in ids)
            if letters.strip():
                words.append(letters.strip())
    return " ".join(words)


def _decode_prediction(cells: list[dict]) -> str:
    """recognize_page() cells -> lowercase letters/spaces only, reading order.

    Cells come back in raw detection order; sort by (line, col) first. Any
    char that isn't a single a-z letter or the literal "space" token
    (capital_sign, number_sign, unknown "#<code>") is dropped, same scope
    limit as the reference decode.
    """
    ordered = sorted(cells, key=lambda c: (c["line"], c["col"]))
    out = []
    for c in ordered:
        ch = c["char"]
        if len(ch) == 1 and ch in string.ascii_lowercase:
            out.append(ch)
        elif ch == "space":
            out.append(" ")
    return "".join(out)


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(
                prev[j] + 1,        # deletion
                cur[j - 1] + 1,     # insertion
                prev[j - 1] + (ca != cb),  # substitution
            )
        prev = cur
    return prev[-1]


def _accuracy(gt: str, pred: str) -> tuple[float, int]:
    dist = _levenshtein(gt, pred)
    denom = max(len(gt), 1)
    return max(0.0, 1.0 - dist / denom), dist


def main() -> None:
    parser = argparse.ArgumentParser(description="Gold page-text accuracy (letters-only)")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--cell-conf", type=float, default=0.25)
    parser.add_argument("--checkpoint", default=None, help="CNN classifier checkpoint (default: recognize.py's own resolution order)")
    parser.add_argument("--cell-weights", default=None, help="YOLO cell-detector weights (default: cell_detect/weights/braille_cell_best.pt)")
    parser.add_argument("--spine-boost", action="store_true", help="Re-detect the spine-proximal strip at higher resolution and merge (see CellDetector.detect_boxes)")
    parser.add_argument("--drop-ruler-lines", action=argparse.BooleanOptionalAction, default=True,
                         help="Remove decorative divider/ruler rows the detector mistakes for cells (see recognize._drop_ruler_lines). On by default; pass --no-drop-ruler-lines to disable")
    args = parser.parse_args()

    pages = sorted(TEXT_DIR.glob("pg-*.txt"), key=lambda p: int(re.search(r"\d+", p.stem).group()))
    if not pages:
        raise SystemExit(f"No pg-N.txt files found in {TEXT_DIR}")

    rows = []
    total_gt_chars = total_gt_letters = 0
    total_dist_sp = total_dist_letters = 0

    for txt_path in pages:
        n = re.search(r"\d+", txt_path.stem).group()
        img_path = IMAGE_DIR / f"pg-{n}.jpeg"
        if not img_path.exists():
            print(f"skip pg-{n}: no matching image {img_path}")
            continue

        gt = _decode_reference(txt_path)
        cells = recognize_page(
            str(img_path), backend="cells", lang=args.lang, cell_conf=args.cell_conf,
            cnn_checkpoint=args.checkpoint, cell_weights=args.cell_weights,
            spine_boost=args.spine_boost, drop_ruler_lines=args.drop_ruler_lines,
        )
        pred = _decode_prediction(cells)

        acc_sp, dist_sp = _accuracy(gt, pred)
        gt_nosp, pred_nosp = gt.replace(" ", ""), pred.replace(" ", "")
        acc_letters, dist_letters = _accuracy(gt_nosp, pred_nosp)

        rows.append((f"pg-{n}", len(cells), len(gt), acc_sp, acc_letters))
        total_gt_chars += len(gt)
        total_gt_letters += len(gt_nosp)
        total_dist_sp += dist_sp
        total_dist_letters += dist_letters

        print(f"pg-{n}: {len(cells)} cells detected | gt {len(gt)} chars | "
              f"acc_with_spaces={acc_sp:.3f} acc_letters_only={acc_letters:.3f}")
        print(f"  gt:   {gt[:100]}{'...' if len(gt) > 100 else ''}")
        print(f"  pred: {pred[:100]}{'...' if len(pred) > 100 else ''}")

    overall_sp = max(0.0, 1.0 - total_dist_sp / max(total_gt_chars, 1))
    overall_letters = max(0.0, 1.0 - total_dist_letters / max(total_gt_letters, 1))
    print(f"\nOVERALL ({len(rows)} pages): acc_with_spaces={overall_sp:.3f} "
          f"acc_letters_only={overall_letters:.3f}")

    lines = [
        "Ground truth: `Gold Dataset/Text pages/pg-N.txt`, decoded via the ids "
        "in `Gold Dataset/symbols guide.jpeg` -- letters (a-z) and space only; "
        "capital/number/punctuation ids are skipped (unverified dot patterns, "
        "see eval_gold_text.py docstring), not counted as errors either way.",
        "",
        "Predicted: `recognize_page(backend=\"cells\", lang=\"en\")`, sorted by "
        "(line, col), same letters/space-only filter applied.",
        "",
        "| page | cells detected | gt chars | acc (with spaces) | acc (letters only) |",
        "|---|---|---|---|---|",
        *[f"| {p} | {n} | {g} | {a:.3f} | {b:.3f} |" for p, n, g, a, b in rows],
        "",
        f"**Overall: acc_with_spaces={overall_sp:.3f}  acc_letters_only={overall_letters:.3f}**",
        "",
        "acc_with_spaces is lower than acc_letters_only mostly because the "
        "cell-detector YOLO model only proposes boxes where it sees raised "
        "dots -- it has no mechanism to box a blank word-gap cell, so most "
        "spaces are structurally missing from the prediction regardless of "
        "classifier accuracy. That's a detection-recall gap, not a "
        "classification error.",
    ]
    write_eval_report(Path("reports/eval") / "gold_text.md", "Gold page-text accuracy (letters-only)", lines)


if __name__ == "__main__":
    main()
