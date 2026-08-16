"""Stage 6 — page-level: cells detected AND correctly classified.

Matches each predicted box to a ground-truth box by IoU, then checks the
code. Reports precision / recall / exact-match cell accuracy per source.

    py -3.11 -m braille_cnn.eval_end_to_end --backend cells --max-pages 20
    py -3.11 -m braille_cnn.eval_end_to_end --backend dots --sources angelina
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from data_pipeline.contracts import read_manifest, repo_root

from .recognize import recognize_page

ROOT = repo_root()


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    return inter / max(area_a + area_b - inter, 1e-6)


def _match(preds, truths, iou_thresh: float):
    """Greedy one-to-one match. Returns (tp_correct, tp_wrong, fp, fn)."""
    used = set()
    tp_ok = tp_bad = 0
    for pred in preds:
        best_i, best_iou = -1, 0.0
        for i, gt in enumerate(truths):
            if i in used:
                continue
            score = _iou(pred["xyxy"], (gt.x0, gt.y0, gt.x1, gt.y1))
            if score > best_iou:
                best_i, best_iou = i, score
        if best_iou < iou_thresh or best_i < 0:
            continue
        used.add(best_i)
        if int(pred["code"]) == int(truths[best_i].code):
            tp_ok += 1
        else:
            tp_bad += 1
    fp = len(preds) - tp_ok - tp_bad
    fn = len(truths) - len(used)
    return tp_ok, tp_bad, fp, fn


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end page evaluation")
    parser.add_argument("--manifest", default="data_pipeline/manifests/manifest_clean.csv")
    parser.add_argument("--split", default="val")
    parser.add_argument("--sources", nargs="+", default=None)
    parser.add_argument("--backend", choices=("cells", "dots"), default="dots")
    parser.add_argument("--iou", type=float, default=0.4)
    parser.add_argument("--max-pages", type=int, default=30,
                        help="Cap pages so a CPU run finishes")
    parser.add_argument("--lang", default="en")
    args = parser.parse_args()

    frame = read_manifest(ROOT / args.manifest)
    frame = frame[frame["split"] == args.split]
    if args.sources:
        frame = frame[frame["source"].isin(args.sources)]
    pages = list(frame.groupby("image_path"))
    if args.max_pages:
        pages = pages[: args.max_pages]
    if not pages:
        raise SystemExit(f"No pages in split={args.split}. Rebuild the manifest.")

    print(f"{len(pages)} pages  backend={args.backend}  split={args.split}")
    totals = defaultdict(int)

    for image_path, rows in pages:
        path = ROOT / str(image_path)
        if not path.exists():
            print(f"  missing {image_path}")
            continue
        try:
            preds = recognize_page(Image.open(path), backend=args.backend, lang=args.lang)
        except FileNotFoundError as exc:
            print(f"  skip {image_path}: {exc}")
            continue
        tp_ok, tp_bad, fp, fn = _match(preds, list(rows.itertuples(index=False)), args.iou)
        totals["tp_ok"] += tp_ok
        totals["tp_bad"] += tp_bad
        totals["fp"] += fp
        totals["fn"] += fn
        print(
            f"  {path.name:40s}  pred={len(preds):4d}  gt={len(rows):4d}  "
            f"ok={tp_ok}  wrong={tp_bad}  fp={fp}  fn={fn}"
        )

    detected = totals["tp_ok"] + totals["tp_bad"]
    pred_n = detected + totals["fp"]
    gt_n = detected + totals["fn"]
    prec = detected / max(pred_n, 1)
    rec = detected / max(gt_n, 1)
    cls = totals["tp_ok"] / max(detected, 1)
    print("\n=== end-to-end ===")
    print(f"  detection precision : {prec:.4f}")
    print(f"  detection recall    : {rec:.4f}")
    print(f"  classification | IoU≥{args.iou} : {cls:.4f}")
    print(f"  exact cells         : {totals['tp_ok']} / {gt_n} gt")
    from .eval_report import write_eval_report

    write_eval_report(
        Path("reports/eval") / f"end_to_end_{args.backend}_{args.split}.md",
        f"End-to-end page eval ({args.backend}, {args.split})",
        [
            f"Pages: {len(pages)}",
            f"Detection precision: **{prec:.4f}**",
            f"Detection recall: **{rec:.4f}**",
            f"Classification given IoU≥{args.iou}: **{cls:.4f}**",
            f"Exact matched cells: {totals['tp_ok']} / {gt_n} gt",
        ],
    )


if __name__ == "__main__":
    main()
