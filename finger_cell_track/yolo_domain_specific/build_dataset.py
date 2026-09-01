"""Build Ultralytics YOLO dataset from LabelMe Braille_fingertip annotations.

Split: 48 train / 6 val / 6 test (seed 42).

Usage (from repo root)::

    finger_cell_track\\.venv\\Scripts\\python.exe finger_cell_track/yolo_domain_specific/build_dataset.py
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from labelme_to_yolo import convert_json, resolve_image_size

_REPO = _HERE.parents[1]
_DEFAULT_SOURCE = _REPO / "Gold Dataset" / "Braille_fingertip"
_DEFAULT_OUT = _HERE / "datasets" / "braille_fingertip_yolo"

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_TRAIN_N = 48
_VAL_N = 6
_TEST_N = 6
_SEED = 42


def write_data_yaml(out_root: Path) -> Path:
    yaml_path = out_root / "data.yaml"
    text = f"""# BrailleLens domain fingertip detector (single class)
path: {out_root.resolve().as_posix()}
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: fingertip
"""
    yaml_path.write_text(text, encoding="utf-8")
    return yaml_path


def _collect_labeled_images(source: Path) -> list[Path]:
    images = sorted(
        f for f in source.iterdir() if f.suffix.lower() in _IMG_EXTS and f.is_file()
    )
    return images


def build_dataset(
    source: Path,
    out_root: Path,
    *,
    clean: bool,
    seed: int,
) -> dict[str, int | list[str]]:
    if clean and out_root.exists():
        shutil.rmtree(out_root)

    for split in ("train", "val", "test"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    images = _collect_labeled_images(source)
    if len(images) != _TRAIN_N + _VAL_N + _TEST_N:
        print(
            f"WARNING: expected {_TRAIN_N + _VAL_N + _TEST_N} images, found {len(images)}"
        )

    rng = random.Random(seed)
    shuffled = images[:]
    rng.shuffle(shuffled)

    n_test = min(_TEST_N, len(shuffled))
    n_val = min(_VAL_N, max(0, len(shuffled) - n_test))
    n_train = len(shuffled) - n_val - n_test

    splits: dict[str, list[Path]] = {
        "test": shuffled[:n_test],
        "val": shuffled[n_test : n_test + n_val],
        "train": shuffled[n_test + n_val :],
    }

    warnings: list[str] = []
    counts = {"train": 0, "val": 0, "test": 0, "no_json": 0, "empty_label": 0, "with_box": 0}

    for split, subset in splits.items():
        for img_path in subset:
            json_path = source / f"{img_path.stem}.json"
            dst_img = out_root / "images" / split / img_path.name
            dst_lbl = out_root / "labels" / split / f"{img_path.stem}.txt"

            shutil.copy2(img_path, dst_img)

            if not json_path.exists():
                counts["no_json"] += 1
                warnings.append(f"no JSON: {img_path.name}")
                dst_lbl.write_text("", encoding="utf-8")
            else:
                w, h = resolve_image_size(json_path, source)
                lines = convert_json(json_path, w, h)
                dst_lbl.write_text(
                    "\n".join(lines) + ("\n" if lines else ""),
                    encoding="utf-8",
                )
                if lines:
                    counts["with_box"] += 1
                else:
                    counts["empty_label"] += 1
                    warnings.append(f"empty label: {img_path.name}")

            counts[split] += 1

    yaml_path = write_data_yaml(out_root)
    counts["yaml"] = str(yaml_path)  # type: ignore[assignment]
    counts["warnings"] = warnings  # type: ignore[assignment]
    return counts


def main() -> None:
    p = argparse.ArgumentParser(description="Build braille_fingertip_yolo dataset")
    p.add_argument("--source", type=Path, default=_DEFAULT_SOURCE)
    p.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    p.add_argument("--clean", action="store_true", help="Delete output dir first")
    p.add_argument("--seed", type=int, default=_SEED)
    args = p.parse_args()

    if not args.source.is_dir():
        raise SystemExit(f"Source not found: {args.source}")

    result = build_dataset(args.source, args.out, clean=args.clean, seed=args.seed)
    print(f"Output: {args.out}")
    print(f"  train: {result['train']}  val: {result['val']}  test: {result['test']}")
    print(f"  with box: {result['with_box']}  empty: {result['empty_label']}  no JSON: {result['no_json']}")
    print(f"  data.yaml: {result['yaml']}")

    warnings: list[str] = result["warnings"]  # type: ignore[assignment]
    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for w in warnings[:20]:
            print(f"  - {w}")
        if len(warnings) > 20:
            print(f"  ... and {len(warnings) - 20} more")
        print("\nFix labels in LabelMe, then re-run with --clean")
    elif result["with_box"] == 0:
        print("\nNo boxes found. Annotate in LabelMe first — see ANNOTATION_GUIDE.md")


if __name__ == "__main__":
    main()
