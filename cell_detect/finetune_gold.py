"""Stage 4a-gold -- short cell-detector fine-tune on labelled Gold pages.

Deliberately NOT built on data_pipeline.integrate / clean / reduce: that
pipeline pulls in every currently-labelled Gold page under its documented
12-page split (GOLD_PAGE_SPLITS in data_pipeline/integrate.py) and, for a
detector fine-tune, doesn't need DSBI/Angelina reprocessed at all. This
script reads a handful of LabelMe JSONs directly and builds a small
single-class YOLO dataset from exactly the pages given on the command line
-- fast and CPU-feasible (a few images, a few epochs resuming from an
existing checkpoint), unlike a from-scratch YOLO run.

    py -3.11 -m cell_detect.finetune_gold
    py -3.11 -m cell_detect.finetune_gold --train-pages 4 5 8 --val-page 9 --test-page 11

Default pages: train=[4,5,8], val=[9], test=[11] (val is used only for
Ultralytics' own best.pt / early-stopping bookkeeping during fine-tuning;
--test-page is never touched during training and is the number to trust).

Low-quality (different lighting) variants of the same 12 physical pages live
in "Gold Dataset/Low quality dataset/" and get annotated separately (not via
data_pipeline.transfer_gold_labels' homography transfer -- these are hand
labelled). Any low-quality page whose number is already in --train-pages,
--val-page, or --test-page is automatically added as an *extra* image
alongside its high-quality counterpart, in that same split (more images of
the same physical page's existing train/val/test assignment, never a new
split decision -- both lighting variants of a page always stay together)
-- pass --no-low-quality to disable. Doubling val/test this way isn't just
more data: with only ~2 images per split otherwise, held-out metrics are
noisy enough that a single page's quirks can swing the number -- see
reports/eval/gold_cell_detector_finetune.md's pg-10-vs-pg-11 discussion.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
GOLD_DIR = ROOT / "Gold Dataset" / "High quality dataset"
LOW_GOLD_DIR = ROOT / "Gold Dataset" / "Low quality dataset"
VARIANT_DIRS = {"high": GOLD_DIR, "low": LOW_GOLD_DIR}
DEFAULT_BASE_WEIGHTS = ROOT / "cell_detect" / "weights" / "braille_cell_best.pt"
DEFAULT_OUT_WEIGHTS = ROOT / "cell_detect" / "weights" / "braille_cell_gold.pt"
DATASET_ROOT = ROOT / "cell_detect" / "datasets" / "gold_finetune"

CLASS_ID = 0
CLASS_NAME = "braille_cell"


def _yolo_line(x0: float, y0: float, x1: float, y1: float, img_w: int, img_h: int) -> str | None:
    x0, y0 = max(x0, 0.0), max(y0, 0.0)
    x1, y1 = min(x1, float(img_w)), min(y1, float(img_h))
    if x1 - x0 < 1.0 or y1 - y0 < 1.0:
        return None
    xc, yc = (x0 + x1) / 2.0 / img_w, (y0 + y1) / 2.0 / img_h
    nw, nh = (x1 - x0) / img_w, (y1 - y0) / img_h
    if not (0.0 < nw <= 1.0 and 0.0 < nh <= 1.0):
        return None
    return f"{CLASS_ID} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}"


def _page_image(n: int, variant: str = "high") -> Path:
    variant_dir = VARIANT_DIRS[variant]
    for ext in (".jpeg", ".jpg", ".png"):
        p = variant_dir / f"pg-{n}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"No image for pg-{n} in {variant_dir}")


def _convert_page(n: int, split: str, out_root: Path, variant: str = "high") -> int:
    variant_dir = VARIANT_DIRS[variant]
    json_path = variant_dir / f"pg-{n}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"No LabelMe json for pg-{n}: {json_path}")
    img_path = _page_image(n, variant)
    doc = json.loads(json_path.read_text(encoding="utf-8"))
    img_w = int(doc.get("imageWidth") or 0)
    img_h = int(doc.get("imageHeight") or 0)

    lines = []
    for shape in doc.get("shapes", []):
        if shape.get("shape_type") != "rectangle":
            continue
        (px0, py0), (px1, py1) = shape["points"][:2]
        x0, x1 = sorted((float(px0), float(px1)))
        y0, y1 = sorted((float(py0), float(py1)))
        line = _yolo_line(x0, y0, x1, y1, img_w, img_h)
        if line:
            lines.append(line)

    img_dir = out_root / "images" / split
    lbl_dir = out_root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    stem = f"pg-{n}" if variant == "high" else f"pg-{n}-{variant}"
    shutil.copy2(img_path, img_dir / f"{stem}{img_path.suffix}")
    (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def build_dataset(train_pages, val_pages, test_pages, out_root: Path, low_quality_pages=()) -> tuple[Path, dict[str, list[int]]]:
    if out_root.exists():
        shutil.rmtree(out_root)
    counts = {"train": 0, "val": 0, "test": 0}
    split_for_page: dict[int, str] = {}
    for split, pages in (("train", train_pages), ("val", val_pages), ("test", test_pages)):
        for n in pages:
            counts[split] += _convert_page(n, split, out_root, variant="high")
            split_for_page[n] = split

    low_used: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for n in low_quality_pages:
        split = split_for_page.get(n)
        if split is None:
            continue  # page isn't in any split at all -- nothing to add it alongside
        counts[split] += _convert_page(n, split, out_root, variant="low")
        low_used[split].append(n)

    print(f"Gold fine-tune dataset: train pg-{train_pages} + low pg-{low_used['train']} "
          f"({counts['train']} boxes), "
          f"val pg-{val_pages} + low pg-{low_used['val']} ({counts['val']} boxes), "
          f"test pg-{test_pages} + low pg-{low_used['test']} ({counts['test']} boxes) "
          f"-- held out, never trained/monitored on")

    data = {
        "path": str(out_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {CLASS_ID: CLASS_NAME},
        "nc": 1,
    }
    yaml_path = out_root / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return yaml_path, low_used


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4a-gold -- fine-tune the cell detector on labelled Gold pages")
    parser.add_argument("--train-pages", type=int, nargs="+", default=[4, 5, 8])
    parser.add_argument("--val-page", type=int, nargs="+", default=[9])
    parser.add_argument("--test-page", type=int, nargs="+", default=[11])
    parser.add_argument("--base-weights", type=Path, default=DEFAULT_BASE_WEIGHTS)
    parser.add_argument("--out-weights", type=Path, default=DEFAULT_OUT_WEIGHTS)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--lr0", type=float, default=0.001, help="Low LR -- fine-tuning from a trained checkpoint on very few images")
    parser.add_argument("--mosaic", type=float, default=0.0, help="0.0 = off (default: too few train images for mosaic to help, not hurt -- per this script's docstring)")
    parser.add_argument("--shear", type=float, default=1.0,
                         help="Matches cell_detect/configs/cells.yaml's full-scale Job A value (was silently 0.0 -- ultralytics' own default -- before)")
    parser.add_argument("--perspective", type=float, default=0.0005,
                         help="Matches cell_detect/configs/cells.yaml's full-scale Job A value; real photos have real perspective distortion (see braille_cnn/PIPELINE.md's ~13-point measured drop)")
    parser.add_argument("--scale", type=float, default=0.20,
                         help="Failure analysis in reports/eval/gold_cell_detector_finetune.md found missed cells run ~6-7%% smaller than detected ones (perspective foreshortening near a book's spine) -- worth trying higher than the current default to see if it closes more of that gap")
    parser.add_argument("--no-low-quality", action="store_true",
                         help="Don't add low-quality-lighting variants of any split's pages as extra images, even if labelled")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False,
                         help="Automatic mixed precision -- off by default, it's a CUDA feature and unreliable on CPU")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip the before-fine-tune eval on --test-page")
    args = parser.parse_args()

    if not args.base_weights.exists():
        raise SystemExit(f"Base weights not found: {args.base_weights}")

    low_quality_pages = []
    if not args.no_low_quality:
        low_quality_pages = sorted(
            int(p.stem.split("-")[1]) for p in LOW_GOLD_DIR.glob("pg-*.json")
        ) if LOW_GOLD_DIR.is_dir() else []

    yaml_path, low_used = build_dataset(args.train_pages, args.val_page, args.test_page, DATASET_ROOT, low_quality_pages)

    from ultralytics import YOLO

    baseline = None
    if not args.skip_baseline:
        print("\n=== Baseline: braille_cell_best.pt on held-out test page (before fine-tune) ===")
        base_model = YOLO(str(args.base_weights))
        baseline = base_model.val(data=str(yaml_path), split="test", imgsz=args.imgsz, device=args.device, plots=False)

    print(f"\n=== Fine-tuning {args.base_weights.name} on pg-{args.train_pages} ===")
    model = YOLO(str(args.base_weights))
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        lr0=args.lr0,
        amp=args.amp,
        fliplr=0.0,  # tried 0.5 (safe in principle: nc=1, no dot patterns here)
        flipud=0.0,  # but on 12 train images it hurt badly (val mAP50 0.81->0.32,
                     # see reports/eval/gold_cell_detector_finetune.md) -- too much
                     # augmentation variance for this little real data to absorb
        mosaic=args.mosaic,
        degrees=3.0,
        translate=0.10,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        project=str(ROOT / "cell_detect" / "runs"),
        name="gold_finetune",
        exist_ok=True,
    )

    best = ROOT / "cell_detect" / "runs" / "gold_finetune" / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"Training finished but {best} is missing")
    args.out_weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, args.out_weights)
    print(f"\nSaved fine-tuned weights: {args.out_weights}")

    print(f"\n=== After fine-tune: {args.out_weights.name} on held-out test page ===")
    tuned_model = YOLO(str(args.out_weights))
    after = tuned_model.val(data=str(yaml_path), split="test", imgsz=args.imgsz, device=args.device, plots=False)

    def _row(label, m):
        if m is None:
            return f"| {label} | n/a | n/a | n/a |"
        return f"| {label} | {m.box.map50:.4f} | {m.box.mp:.4f} | {m.box.mr:.4f} |"

    print("\n| model | mAP50 | precision | recall |")
    print("|---|---|---|---|")
    print(_row(f"baseline ({args.base_weights.name})", baseline))
    print(_row(f"gold fine-tuned ({args.out_weights.name})", after))

    from braille_cnn.eval_report import write_eval_report

    def _pages_desc(pages, low):
        return f"pg-{pages}" + (f" + low-quality-lighting pg-{low}" if low else "")

    lines = [
        f"Train pages: {_pages_desc(args.train_pages, low_used['train'])} | "
        f"val (checkpoint selection only): {_pages_desc(args.val_page, low_used['val'])} | "
        f"**test (held out, never trained/monitored on): {_pages_desc(args.test_page, low_used['test'])}**",
        "",
        "| model | mAP50 | precision | recall |",
        "|---|---|---|---|",
        _row(f"baseline (`{args.base_weights.name}`)", baseline),
        _row(f"gold fine-tuned (`{args.out_weights.name}`)", after),
    ]
    write_eval_report(
        Path("reports/eval") / "gold_cell_detector_finetune.md",
        "Gold cell-detector fine-tune: before vs after (held-out test page)",
        lines,
    )


if __name__ == "__main__":
    main()
